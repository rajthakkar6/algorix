"""Tests for A8 earnings-surprise ingestion (INDICATORS.md A8, PEAD).

Fixture DataFrame mirrors yfinance's real `Ticker.earnings_dates` shape:
a DatetimeIndex named "Earnings Date" with US-Eastern tzoffset, and
"EPS Estimate"/"Reported EPS"/"Surprise(%)" columns -- verified live
2026-09-24 (see earnings.py's module docstring).
"""

from datetime import date

import pandas as pd
import pytest

from algorix.earnings import (
    SOURCE_YFINANCE_EARNINGS,
    EarningsFetchResult,
    ingest_earnings,
    parse_earnings_dates,
)
from algorix.exceptions import DataUnavailableError
from algorix.models import EarningsSurpriseRecord, Exchange, Instrument, InstrumentType
from algorix.storage import Database, EarningsSurpriseRepository, InstrumentRepository

RELIANCE = Instrument(
    symbol="RELIANCE", exchange=Exchange.NSE,
    instrument_type=InstrumentType.EQUITY, yahoo_symbol="RELIANCE.NS",
)


def earnings_frame(rows):
    """rows: list of (date_str, eps_estimate, reported_eps, surprise_pct),
    any of the last three may be None to simulate an unreported/future row."""
    index = pd.DatetimeIndex(
        [pd.Timestamp(d, tz="America/New_York") for d, *_ in rows],
        name="Earnings Date",
    )
    return pd.DataFrame(
        {
            "EPS Estimate": [r[1] for r in rows],
            "Reported EPS": [r[2] for r in rows],
            "Surprise(%)": [r[3] for r in rows],
        },
        index=index,
    )


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "test.db")
    database.migrate()
    return database


@pytest.fixture
def reliance_id(db):
    return InstrumentRepository(db).upsert(RELIANCE)


# ---------------------------------------------------------------------------
# parse_earnings_dates -- positive
# ---------------------------------------------------------------------------


def test_parses_a_reported_quarter():
    frame = earnings_frame([("2026-07-17", 14.97, 15.48, 3.38)])

    result = parse_earnings_dates("RELIANCE", frame)

    assert len(result.records) == 1
    record = result.records[0]
    assert record.symbol == "RELIANCE"
    assert record.report_date == date(2026, 7, 17)
    assert record.eps_estimate == pytest.approx(14.97)
    assert record.eps_actual == pytest.approx(15.48)
    assert record.surprise_pct == pytest.approx(3.38)


def test_negative_surprise_is_preserved():
    frame = earnings_frame([("2026-04-24", 15.39, 12.54, -18.52)])

    result = parse_earnings_dates("RELIANCE", frame)

    assert result.records[0].surprise_pct == pytest.approx(-18.52)


def test_multiple_quarters_all_parse():
    frame = earnings_frame(
        [
            ("2026-04-24", 15.39, 12.54, -18.52),
            ("2026-07-17", 14.97, 15.48, 3.38),
        ]
    )

    result = parse_earnings_dates("RELIANCE", frame)

    assert len(result.records) == 2
    assert {r.report_date for r in result.records} == {
        date(2026, 4, 24), date(2026, 7, 17)
    }


def test_only_the_calendar_date_is_used_not_the_tzoffset_time():
    """yfinance's US-Eastern tzoffset must never shift the reported date --
    see the module docstring's cross-check against NSE's own announcement."""
    frame = earnings_frame([("2026-07-17 09:00:00", 14.97, 15.48, 3.38)])

    result = parse_earnings_dates("RELIANCE", frame)

    assert result.records[0].report_date == date(2026, 7, 17)


# ---------------------------------------------------------------------------
# parse_earnings_dates -- negative
# ---------------------------------------------------------------------------


def test_empty_frame_is_a_legitimate_zero_result():
    """JIOFIN's real, live shape -- no earnings coverage, not an error."""
    result = parse_earnings_dates("JIOFIN", pd.DataFrame())

    assert result.records == []


def test_none_frame_is_a_legitimate_zero_result():
    result = parse_earnings_dates("JIOFIN", None)

    assert result.records == []


def test_future_unreported_quarter_is_excluded():
    """A future consensus estimate with no actual yet is not a surprise --
    matches the real live shape (the upcoming quarter's row has an
    estimate but NaN for Reported EPS and Surprise(%))."""
    frame = earnings_frame(
        [
            ("2026-07-17", 14.97, 15.48, 3.38),
            ("2026-10-16", 16.27, None, None),
        ]
    )

    result = parse_earnings_dates("RELIANCE", frame)

    assert len(result.records) == 1
    assert result.records[0].report_date == date(2026, 7, 17)


def test_row_with_only_surprise_missing_is_excluded():
    frame = earnings_frame([("2026-07-17", 14.97, 15.48, None)])

    result = parse_earnings_dates("RELIANCE", frame)

    assert result.records == []


# ---------------------------------------------------------------------------
# ingest_earnings -- positive
# ---------------------------------------------------------------------------


def test_ingest_stores_records(db, reliance_id):
    result = EarningsFetchResult(
        records=[
            EarningsSurpriseRecord(
                symbol="RELIANCE", report_date=date(2026, 7, 17),
                eps_estimate=14.97, eps_actual=15.48, surprise_pct=3.38,
            )
        ]
    )

    report = ingest_earnings(db, RELIANCE, reliance_id, result=result)

    assert report.stored == 1
    stored = EarningsSurpriseRepository(db).get_range(
        reliance_id, date(2000, 1, 1), date(2026, 12, 31)
    )
    assert len(stored) == 1
    assert stored[0].surprise_pct == pytest.approx(3.38)


def test_ingest_is_idempotent(db, reliance_id):
    result = EarningsFetchResult(
        records=[
            EarningsSurpriseRecord(
                symbol="RELIANCE", report_date=date(2026, 7, 17),
                eps_estimate=14.97, eps_actual=15.48, surprise_pct=3.38,
            )
        ]
    )

    ingest_earnings(db, RELIANCE, reliance_id, result=result)
    ingest_earnings(db, RELIANCE, reliance_id, result=result)

    stored = EarningsSurpriseRepository(db).get_range(
        reliance_id, date(2000, 1, 1), date(2026, 12, 31)
    )
    assert len(stored) == 1


def test_ingest_records_source(db, reliance_id):
    result = EarningsFetchResult(
        records=[
            EarningsSurpriseRecord(
                symbol="RELIANCE", report_date=date(2026, 7, 17),
                eps_estimate=14.97, eps_actual=15.48, surprise_pct=3.38,
            )
        ]
    )

    ingest_earnings(db, RELIANCE, reliance_id, result=result)

    with db.connect() as conn:
        source = conn.execute(
            "SELECT source FROM earnings_surprises WHERE instrument_id = ?",
            (reliance_id,),
        ).fetchone()["source"]
    assert source == SOURCE_YFINANCE_EARNINGS


# ---------------------------------------------------------------------------
# ingest_earnings -- negative
# ---------------------------------------------------------------------------


def test_ingest_with_zero_records_is_not_an_error(db, reliance_id):
    report = ingest_earnings(db, RELIANCE, reliance_id, result=EarningsFetchResult())

    assert report.stored == 0


def test_client_fetch_without_yahoo_symbol_is_rejected():
    from algorix.earnings import YFinanceEarningsClient

    no_symbol = Instrument(
        symbol="GHOST", exchange=Exchange.NSE, instrument_type=InstrumentType.EQUITY,
    )

    with pytest.raises(DataUnavailableError, match="yahoo_symbol"):
        YFinanceEarningsClient().fetch(no_symbol)


# ---------------------------------------------------------------------------
# EarningsSurpriseRepository -- point-in-time correctness
# ---------------------------------------------------------------------------


def test_get_range_excludes_reports_after_the_bound(db, reliance_id):
    """The core guarantee A8 depends on: a backtest replaying a past session
    must never see a report that had not happened yet."""
    repo = EarningsSurpriseRepository(db)
    repo.upsert_many(
        reliance_id,
        [
            EarningsSurpriseRecord(
                symbol="RELIANCE", report_date=date(2026, 4, 24),
                eps_estimate=15.39, eps_actual=12.54, surprise_pct=-18.52,
            ),
            EarningsSurpriseRecord(
                symbol="RELIANCE", report_date=date(2026, 7, 17),
                eps_estimate=14.97, eps_actual=15.48, surprise_pct=3.38,
            ),
        ],
        source="test",
    )

    before_july = repo.get_range(reliance_id, date(2000, 1, 1), date(2026, 5, 1))

    assert [r.report_date for r in before_july] == [date(2026, 4, 24)]


def test_get_range_orders_oldest_first(db, reliance_id):
    repo = EarningsSurpriseRepository(db)
    repo.upsert_many(
        reliance_id,
        [
            EarningsSurpriseRecord(
                symbol="RELIANCE", report_date=date(2026, 7, 17),
                eps_estimate=14.97, eps_actual=15.48, surprise_pct=3.38,
            ),
            EarningsSurpriseRecord(
                symbol="RELIANCE", report_date=date(2026, 4, 24),
                eps_estimate=15.39, eps_actual=12.54, surprise_pct=-18.52,
            ),
        ],
        source="test",
    )

    stored = repo.get_range(reliance_id, date(2000, 1, 1), date(2026, 12, 31))

    assert [r.report_date for r in stored] == [date(2026, 4, 24), date(2026, 7, 17)]


def test_get_range_rejects_start_after_end(db):
    with pytest.raises(ValueError, match="after"):
        EarningsSurpriseRepository(db).get_range(1, date(2026, 9, 24), date(2026, 1, 1))


def test_get_range_for_unknown_instrument_is_empty(db):
    stored = EarningsSurpriseRepository(db).get_range(
        999999, date(2000, 1, 1), date(2026, 12, 31)
    )

    assert stored == []


def test_reclassification_of_the_same_report_updates_in_place(db, reliance_id):
    """A yfinance revision to a past quarter's figures (rare, but the API
    gives no guarantee it never happens) must update, not duplicate."""
    repo = EarningsSurpriseRepository(db)
    repo.upsert_many(
        reliance_id,
        [
            EarningsSurpriseRecord(
                symbol="RELIANCE", report_date=date(2026, 7, 17),
                eps_estimate=14.97, eps_actual=15.48, surprise_pct=3.38,
            )
        ],
        source="test",
    )
    repo.upsert_many(
        reliance_id,
        [
            EarningsSurpriseRecord(
                symbol="RELIANCE", report_date=date(2026, 7, 17),
                eps_estimate=14.97, eps_actual=15.50, surprise_pct=3.54,
            )
        ],
        source="test",
    )

    stored = repo.get_range(reliance_id, date(2000, 1, 1), date(2026, 12, 31))

    assert len(stored) == 1
    assert stored[0].surprise_pct == pytest.approx(3.54)
