"""Tests for OHLCV ingestion and dirty-data handling.

The phantom-holiday case is modelled on a real observed yfinance response:
RELIANCE.NS returns a row for 2026-09-14 (an NSE holiday) with all four
prices equal to 1257.50 and volume 0.
"""

from datetime import date

import pandas as pd
import pytest

from algorix.calendar import TradingCalendar
from algorix.exceptions import (
    DataIntegrityError,
    DataUnavailableError,
    SourceUnreachableError,
)
from algorix.ingestion import (
    FetchResult,
    IngestReport,
    RejectedBar,
    bars_from_dataframe,
    ingest_instrument,
    ingest_many,
)
from algorix.models import Exchange, Instrument, InstrumentType
from algorix.storage import BarRepository, Database, InstrumentRepository

LAST_COMPLETED = date(2026, 9, 18)


@pytest.fixture
def cal():
    return TradingCalendar()


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "test.db")
    database.migrate()
    return database


@pytest.fixture
def reliance():
    return Instrument(
        symbol="RELIANCE",
        exchange=Exchange.NSE,
        instrument_type=InstrumentType.EQUITY,
        yahoo_symbol="RELIANCE.NS",
    )


@pytest.fixture
def reliance_id(db, reliance):
    return InstrumentRepository(db).upsert(reliance)


def frame(rows: dict[str, tuple]) -> pd.DataFrame:
    """Build a yfinance-shaped frame: {date_str: (O, H, L, C, V)}."""
    index = pd.DatetimeIndex(
        [pd.Timestamp(d, tz="Asia/Kolkata") for d in rows], name="Date"
    )
    return pd.DataFrame(
        {
            "Open": [v[0] for v in rows.values()],
            "High": [v[1] for v in rows.values()],
            "Low": [v[2] for v in rows.values()],
            "Close": [v[3] for v in rows.values()],
            "Volume": [v[4] for v in rows.values()],
            "Dividends": [0.0] * len(rows),
            "Stock Splits": [0.0] * len(rows),
        },
        index=index,
    )


class StubSource:
    """Stands in for YFinanceBarSource without touching the network."""

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = 0

    def fetch(self, instrument, start, end, max_session=None):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


# --------------------------------------------------------------------------
# Conversion -- positive
# --------------------------------------------------------------------------


def test_valid_rows_become_bars(cal):
    result = bars_from_dataframe(
        frame(
            {
                "2026-09-17": (1244.8, 1253.4, 1238.5, 1243.9, 7752895),
                "2026-09-18": (1245.0, 1247.3, 1226.4, 1226.4, 15122715),
            }
        ),
        cal,
        LAST_COMPLETED,
    )

    assert len(result.bars) == 2
    assert result.rejected == []
    assert result.bars[0].close == pytest.approx(1243.9)


def test_bars_are_sorted_oldest_first(cal):
    result = bars_from_dataframe(
        frame(
            {
                "2026-09-18": (1245.0, 1247.3, 1226.4, 1226.4, 100),
                "2026-09-16": (1243.0, 1255.0, 1240.0, 1240.0, 100),
            }
        ),
        cal,
        LAST_COMPLETED,
    )

    assert [b.session_date for b in result.bars] == [
        date(2026, 9, 16),
        date(2026, 9, 18),
    ]


def test_zero_volume_on_a_real_session_is_kept(cal):
    """Zero volume is legitimate on a genuine session -- only the date matters."""
    result = bars_from_dataframe(
        frame({"2026-09-18": (100.0, 101.0, 99.0, 100.5, 0)}), cal, LAST_COMPLETED
    )

    assert len(result.bars) == 1
    assert result.bars[0].volume == 0


# --------------------------------------------------------------------------
# The phantom-holiday bar -- the case that motivated calendar filtering
# --------------------------------------------------------------------------


def test_phantom_holiday_bar_is_rejected(cal):
    """yfinance emits 2026-09-14 (an NSE holiday) as flat OHLC, zero volume."""
    result = bars_from_dataframe(
        frame(
            {
                "2026-09-11": (1267.0, 1267.4, 1253.0, 1257.5, 8777736),
                "2026-09-14": (1257.5, 1257.5, 1257.5, 1257.5, 0),
                "2026-09-15": (1252.5, 1259.4, 1235.3, 1235.3, 13754376),
            }
        ),
        cal,
        LAST_COMPLETED,
    )

    assert [b.session_date for b in result.bars] == [
        date(2026, 9, 11),
        date(2026, 9, 15),
    ]
    assert len(result.rejected) == 1
    assert result.rejected[0].session_date == date(2026, 9, 14)
    assert "not an NSE trading session" in result.rejected[0].reason


def test_weekend_bar_is_rejected(cal):
    result = bars_from_dataframe(
        frame(
            {
                "2026-09-18": (100.0, 101.0, 99.0, 100.5, 1000),
                "2026-09-19": (100.5, 100.5, 100.5, 100.5, 0),  # Saturday
            }
        ),
        cal,
        date(2026, 9, 20),
    )

    assert len(result.bars) == 1
    assert result.rejected[0].session_date == date(2026, 9, 19)


def test_rejections_are_reported_not_silently_dropped(cal):
    result = bars_from_dataframe(
        frame({"2026-09-14": (1257.5, 1257.5, 1257.5, 1257.5, 0)}),
        cal,
        LAST_COMPLETED,
    )

    assert result.has_rejections is True
    assert "2026-09-14" in result.rejection_summary()


# --------------------------------------------------------------------------
# In-progress sessions
# --------------------------------------------------------------------------


def test_incomplete_session_is_rejected(cal):
    """A mid-session fetch returns today's partial bar -- not a close."""
    result = bars_from_dataframe(
        frame(
            {
                "2026-09-17": (1244.8, 1253.4, 1238.5, 1243.9, 7752895),
                "2026-09-18": (1245.0, 1246.0, 1244.0, 1245.5, 300),
            }
        ),
        cal,
        max_session=date(2026, 9, 17),
    )

    assert [b.session_date for b in result.bars] == [date(2026, 9, 17)]
    assert "not complete" in result.rejected[0].reason


def test_session_exactly_at_max_is_kept(cal):
    result = bars_from_dataframe(
        frame({"2026-09-18": (100.0, 101.0, 99.0, 100.5, 10)}),
        cal,
        max_session=date(2026, 9, 18),
    )

    assert len(result.bars) == 1


# --------------------------------------------------------------------------
# Malformed rows
# --------------------------------------------------------------------------


def test_nan_price_row_is_rejected(cal):
    result = bars_from_dataframe(
        frame(
            {
                "2026-09-17": (1244.8, 1253.4, 1238.5, 1243.9, 7752895),
                "2026-09-18": (float("nan"), 101.0, 99.0, 100.5, 1000),
            }
        ),
        cal,
        LAST_COMPLETED,
    )

    assert len(result.bars) == 1
    assert "missing value" in result.rejected[0].reason


def test_nan_volume_row_is_rejected(cal):
    result = bars_from_dataframe(
        frame({"2026-09-18": (100.0, 101.0, 99.0, 100.5, float("nan"))}),
        cal,
        LAST_COMPLETED,
    )

    assert result.bars == []
    assert "missing value" in result.rejected[0].reason


def test_incoherent_row_is_rejected_with_reason(cal):
    """high below low -- rejected, and the rest of the frame still ingests."""
    result = bars_from_dataframe(
        frame(
            {
                "2026-09-17": (100.0, 101.0, 99.0, 100.5, 10),
                "2026-09-18": (100.0, 98.0, 99.0, 98.5, 10),
            }
        ),
        cal,
        LAST_COMPLETED,
    )

    assert [b.session_date for b in result.bars] == [date(2026, 9, 17)]
    assert "is below low" in result.rejected[0].reason


def test_negative_price_row_is_rejected(cal):
    result = bars_from_dataframe(
        frame({"2026-09-18": (-1.0, 101.0, 99.0, 100.5, 10)}),
        cal,
        LAST_COMPLETED,
    )

    assert result.bars == []
    assert result.rejected


def test_empty_frame_raises(cal):
    with pytest.raises(DataUnavailableError, match="No price rows"):
        bars_from_dataframe(pd.DataFrame(), cal, LAST_COMPLETED)


def test_none_frame_raises(cal):
    with pytest.raises(DataUnavailableError, match="No price rows"):
        bars_from_dataframe(None, cal, LAST_COMPLETED)


def test_missing_columns_raise(cal):
    bad = pd.DataFrame(
        {"Open": [1.0], "Close": [1.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-09-18", tz="Asia/Kolkata")]),
    )

    with pytest.raises(DataIntegrityError, match="missing column"):
        bars_from_dataframe(bad, cal, LAST_COMPLETED)


# --------------------------------------------------------------------------
# Ingestion into storage
# --------------------------------------------------------------------------


def test_ingest_stores_bars_and_reports_clean(db, cal, reliance, reliance_id):
    result = bars_from_dataframe(
        frame(
            {
                "2026-09-17": (1244.8, 1253.4, 1238.5, 1243.9, 100),
                "2026-09-18": (1245.0, 1247.3, 1226.4, 1226.4, 100),
            }
        ),
        cal,
        LAST_COMPLETED,
    )

    report = ingest_instrument(
        db,
        reliance,
        reliance_id,
        date(2026, 9, 17),
        date(2026, 9, 18),
        cal,
        source=StubSource(result=result),
    )

    assert report.stored == 2
    assert report.is_clean is True
    assert len(BarRepository(db).get_range(reliance_id, date(2026, 9, 1), date(2026, 9, 30))) == 2


def test_ingest_reports_missing_sessions(db, cal, reliance, reliance_id):
    """A gap in the feed must surface, not pass as a complete ingestion."""
    partial = bars_from_dataframe(
        frame({"2026-09-18": (1245.0, 1247.3, 1226.4, 1226.4, 100)}),
        cal,
        LAST_COMPLETED,
    )

    report = ingest_instrument(
        db,
        reliance,
        reliance_id,
        date(2026, 9, 16),
        date(2026, 9, 18),
        cal,
        source=StubSource(result=partial),
    )

    assert report.stored == 1
    assert report.is_clean is False
    assert date(2026, 9, 16) in report.missing_sessions
    assert date(2026, 9, 17) in report.missing_sessions


def test_ingest_is_idempotent(db, cal, reliance, reliance_id):
    result = bars_from_dataframe(
        frame({"2026-09-18": (1245.0, 1247.3, 1226.4, 1226.4, 100)}),
        cal,
        LAST_COMPLETED,
    )
    args = (db, reliance, reliance_id, date(2026, 9, 18), date(2026, 9, 18), cal)

    ingest_instrument(*args, source=StubSource(result=result))
    ingest_instrument(*args, source=StubSource(result=result))

    stored = BarRepository(db).get_range(reliance_id, date(2026, 9, 18), date(2026, 9, 18))
    assert len(stored) == 1


def test_ingest_many_continues_past_a_failure(db, cal, reliance, reliance_id):
    """One delisted symbol must not abort a 50-stock universe refresh."""
    good = Instrument(
        symbol="TCS",
        exchange=Exchange.NSE,
        instrument_type=InstrumentType.EQUITY,
        yahoo_symbol="TCS.NS",
    )
    good_id = InstrumentRepository(db).upsert(good)

    class FlakySource:
        def fetch(self, instrument, start, end, max_session=None):
            if instrument.symbol == "RELIANCE":
                raise DataUnavailableError("delisted")
            return bars_from_dataframe(
                frame({"2026-09-18": (100.0, 101.0, 99.0, 100.5, 10)}),
                cal,
                LAST_COMPLETED,
            )

    reports = ingest_many(
        db,
        [(reliance, reliance_id), (good, good_id)],
        date(2026, 9, 18),
        date(2026, 9, 18),
        cal,
        source=FlakySource(),
    )

    by_symbol = {r.symbol: r for r in reports}
    assert by_symbol["RELIANCE"].stored == 0
    assert "DataUnavailableError" in by_symbol["RELIANCE"].rejected[0].reason
    assert by_symbol["TCS"].stored == 1


def test_ingest_many_records_unreachable_source(db, cal, reliance, reliance_id):
    reports = ingest_many(
        db,
        [(reliance, reliance_id)],
        date(2026, 9, 18),
        date(2026, 9, 18),
        cal,
        source=StubSource(error=SourceUnreachableError("network down")),
    )

    assert reports[0].stored == 0
    assert "SourceUnreachableError" in reports[0].rejected[0].reason


def test_instrument_without_yahoo_symbol_is_rejected(db, cal):
    from algorix.ingestion import YFinanceBarSource

    orphan = Instrument(
        symbol="MYSTERY",
        exchange=Exchange.NSE,
        instrument_type=InstrumentType.EQUITY,
    )

    with pytest.raises(DataUnavailableError, match="no yahoo_symbol"):
        YFinanceBarSource(cal).fetch(orphan, date(2026, 9, 17), date(2026, 9, 18))


def test_fetch_rejects_reversed_dates(cal, reliance):
    from algorix.ingestion import YFinanceBarSource

    with pytest.raises(ValueError, match="is after end"):
        YFinanceBarSource(cal).fetch(reliance, date(2026, 9, 18), date(2026, 9, 17))


# --------------------------------------------------------------------------
# Live network check -- deselect with: -m "not network"
# --------------------------------------------------------------------------


@pytest.mark.network
def test_live_fetch_filters_the_real_phantom_holiday(cal, reliance):
    """Against live yfinance, 2026-09-14 must not survive ingestion."""
    from algorix.ingestion import YFinanceBarSource

    result = YFinanceBarSource(cal).fetch(
        reliance, date(2026, 9, 7), date(2026, 9, 18)
    )

    session_dates = [b.session_date for b in result.bars]
    assert date(2026, 9, 14) not in session_dates
    assert date(2026, 9, 18) in session_dates
    assert any(r.session_date == date(2026, 9, 14) for r in result.rejected)
