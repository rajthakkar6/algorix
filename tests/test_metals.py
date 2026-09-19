"""Tests for the gold/silver/currency track."""

from datetime import date

import pandas as pd
import pytest

from algorix.calendar import TradingCalendar
from algorix.exceptions import DataUnavailableError
from algorix.ingestion import bars_from_dataframe
from algorix.metals import (
    GOLD_ETF,
    GOLD_USD,
    METAL_INSTRUMENTS,
    SILVER_USD,
    USD_INR,
    MetalSnapshot,
    metal_snapshot,
    seed_metal_instruments,
)
from algorix.models import Bar, CalendarPolicy, Exchange, InstrumentType
from algorix.storage import BarRepository, Database, InstrumentRepository

NSE_HOLIDAY = date(2026, 9, 14)
SESSION = date(2026, 9, 18)


@pytest.fixture
def cal():
    return TradingCalendar()


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "test.db")
    database.migrate()
    return database


def frame(rows: dict[str, tuple], tz: str) -> pd.DataFrame:
    index = pd.DatetimeIndex([pd.Timestamp(d, tz=tz) for d in rows], name="Date")
    return pd.DataFrame(
        {
            "Open": [v[0] for v in rows.values()],
            "High": [v[1] for v in rows.values()],
            "Low": [v[2] for v in rows.values()],
            "Close": [v[3] for v in rows.values()],
            "Volume": [v[4] for v in rows.values()],
        },
        index=index,
    )


# --------------------------------------------------------------------------
# Instrument definitions
# --------------------------------------------------------------------------


def test_comex_and_fx_use_global_policy():
    """They trade on Indian holidays -- NSE filtering would destroy real data."""
    for instrument in (GOLD_USD, SILVER_USD, USD_INR):
        assert instrument.calendar_policy is CalendarPolicy.GLOBAL


def test_nse_etfs_use_nse_policy():
    assert GOLD_ETF.calendar_policy is CalendarPolicy.NSE


def test_all_metal_instruments_have_vendor_symbols():
    assert all(i.yahoo_symbol for i in METAL_INSTRUMENTS)


def test_seeding_registers_every_instrument(db):
    ids = seed_metal_instruments(db)

    assert set(ids) == {i.symbol for i in METAL_INSTRUMENTS}
    assert all(isinstance(v, int) for v in ids.values())


def test_seeding_is_idempotent(db):
    first = seed_metal_instruments(db)
    second = seed_metal_instruments(db)

    assert first == second


def test_calendar_policy_survives_a_round_trip(db):
    seed_metal_instruments(db)

    stored = InstrumentRepository(db).get("GOLDUSD", Exchange.COMEX)

    assert stored is not None
    assert stored.calendar_policy is CalendarPolicy.GLOBAL


def test_nse_instrument_defaults_to_nse_policy(db):
    from algorix.models import Instrument

    InstrumentRepository(db).upsert(
        Instrument(
            symbol="TCS",
            exchange=Exchange.NSE,
            instrument_type=InstrumentType.EQUITY,
        )
    )
    stored = InstrumentRepository(db).get("TCS", Exchange.NSE)

    assert stored is not None
    assert stored.calendar_policy is CalendarPolicy.NSE


# --------------------------------------------------------------------------
# Calendar policy in ingestion
# --------------------------------------------------------------------------


def test_global_instrument_keeps_its_indian_holiday_session(cal):
    """GC=F genuinely traded on 2026-09-14, an NSE holiday."""
    result = bars_from_dataframe(
        frame(
            {
                "2026-09-11": (4400.0, 4410.0, 4390.0, 4405.0, 1000),
                "2026-09-14": (4405.0, 4420.0, 4400.0, 4415.0, 1200),
            },
            tz="America/New_York",
        ),
        calendar=None,
        max_session=SESSION,
    )

    assert [b.session_date for b in result.bars] == [
        date(2026, 9, 11),
        NSE_HOLIDAY,
    ]
    assert result.rejected == []


def test_nse_instrument_still_rejects_the_holiday_bar(cal):
    """The same date, under NSE policy, is a phantom."""
    result = bars_from_dataframe(
        frame(
            {
                "2026-09-11": (125.0, 126.0, 124.0, 125.1, 1000),
                "2026-09-14": (125.1, 125.1, 125.1, 125.1, 0),
            },
            tz="Asia/Kolkata",
        ),
        calendar=cal,
        max_session=SESSION,
    )

    assert [b.session_date for b in result.bars] == [date(2026, 9, 11)]
    assert result.rejected[0].session_date == NSE_HOLIDAY


def test_global_instrument_still_rejects_incoherent_rows():
    """Dropping the session filter must not drop coherence checks."""
    result = bars_from_dataframe(
        frame({"2026-09-18": (100.0, 98.0, 99.0, 98.5, 10)}, tz="America/New_York"),
        calendar=None,
        max_session=SESSION,
    )

    assert result.bars == []
    assert "is below low" in result.rejected[0].reason


def test_global_instrument_still_rejects_in_progress_sessions():
    result = bars_from_dataframe(
        frame({"2026-09-18": (100.0, 101.0, 99.0, 100.5, 10)}, tz="America/New_York"),
        calendar=None,
        max_session=date(2026, 9, 17),
    )

    assert result.bars == []
    assert "not complete" in result.rejected[0].reason


def test_fx_zero_volume_is_accepted():
    """FX carries no volume -- always 0, and that is not an error."""
    result = bars_from_dataframe(
        frame({"2026-09-18": (95.7, 95.9, 95.6, 95.8, 0)}, tz="Europe/London"),
        calendar=None,
        max_session=SESSION,
    )

    assert len(result.bars) == 1
    assert result.bars[0].volume == 0


def test_etf_nan_row_is_rejected(cal):
    """GOLDBEES returned NaN prices with 28M volume on 2026-09-18."""
    result = bars_from_dataframe(
        frame(
            {
                "2026-09-17": (124.26, 125.0, 123.67, 124.43, 22769280),
                "2026-09-18": (float("nan"),) * 4 + (28783409,),
            },
            tz="Asia/Kolkata",
        ),
        calendar=cal,
        max_session=SESSION,
    )

    assert [b.session_date for b in result.bars] == [date(2026, 9, 17)]
    assert "missing value" in result.rejected[0].reason


# --------------------------------------------------------------------------
# Snapshot and ratios
# --------------------------------------------------------------------------


def test_gold_silver_ratio():
    snapshot = MetalSnapshot(
        session_date=SESSION, gold_usd=4424.90, silver_usd=66.556, usd_inr=95.80
    )

    assert snapshot.gold_silver_ratio == pytest.approx(66.48, abs=0.05)


def test_implied_inr_prices_use_the_session_rate():
    snapshot = MetalSnapshot(
        session_date=SESSION, gold_usd=4000.0, silver_usd=50.0, usd_inr=90.0
    )

    assert snapshot.gold_inr_implied == pytest.approx(360000.0)
    assert snapshot.silver_inr_implied == pytest.approx(4500.0)


def test_currency_move_is_separable_from_metal_move():
    """The point of D2: same gold price, weaker rupee -> higher INR price."""
    steady = MetalSnapshot(
        session_date=SESSION, gold_usd=4000.0, silver_usd=50.0, usd_inr=90.0
    )
    weaker_rupee = MetalSnapshot(
        session_date=SESSION, gold_usd=4000.0, silver_usd=50.0, usd_inr=95.0
    )

    assert weaker_rupee.gold_inr_implied > steady.gold_inr_implied
    assert weaker_rupee.gold_usd == steady.gold_usd
    assert weaker_rupee.gold_silver_ratio == steady.gold_silver_ratio


def test_snapshot_reads_stored_bars(db):
    ids = seed_metal_instruments(db)
    repository = BarRepository(db)
    for symbol, close in (
        ("GOLDUSD", 4424.90),
        ("SILVERUSD", 66.556),
        ("USDINR", 95.80),
    ):
        repository.upsert_many(
            ids[symbol],
            [
                Bar(
                    session_date=SESSION,
                    open=close,
                    high=close,
                    low=close,
                    close=close,
                    volume=0,
                )
            ],
            source="test",
        )

    snapshot = metal_snapshot(db, SESSION, instrument_ids=ids)

    assert snapshot.gold_usd == pytest.approx(4424.90)
    assert snapshot.usd_inr == pytest.approx(95.80)


def test_snapshot_refuses_a_missing_currency_leg(db):
    """A fresh metal price with no rate would misattribute the move."""
    ids = seed_metal_instruments(db)
    repository = BarRepository(db)
    for symbol in ("GOLDUSD", "SILVERUSD"):
        repository.upsert_many(
            ids[symbol],
            [Bar(session_date=SESSION, open=1, high=1, low=1, close=1, volume=0)],
            source="test",
        )

    with pytest.raises(DataUnavailableError, match="No USDINR bar"):
        metal_snapshot(db, SESSION, instrument_ids=ids)


def test_snapshot_refuses_a_missing_metal_leg(db):
    ids = seed_metal_instruments(db)

    with pytest.raises(DataUnavailableError, match="No GOLDUSD bar"):
        metal_snapshot(db, SESSION, instrument_ids=ids)


def test_snapshot_refuses_unregistered_instrument(db):
    with pytest.raises(DataUnavailableError):
        metal_snapshot(db, SESSION, instrument_ids={})


# --------------------------------------------------------------------------
# Live network check -- deselect with: -m "not network"
# --------------------------------------------------------------------------


@pytest.mark.network
def test_live_comex_trades_on_an_nse_holiday(cal):
    """Confirms the policy split is necessary, not theoretical."""
    from algorix.ingestion import YFinanceBarSource

    result = YFinanceBarSource(cal).fetch(
        GOLD_USD, date(2026, 9, 10), date(2026, 9, 18)
    )

    assert NSE_HOLIDAY in [b.session_date for b in result.bars]
