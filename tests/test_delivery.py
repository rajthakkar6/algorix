"""Tests for NSE bhavcopy delivery ingestion.

Fixture CSV reproduces the real file's quirks: whitespace-padded headers and
values, multiple series in one file, and `'-'` for an unpublished delivery
figure.
"""

from datetime import date

import pytest

from algorix.delivery import (
    DEFAULT_SERIES,
    BhavcopyResult,
    build_bhavcopy_url,
    ingest_delivery,
    parse_bhavcopy,
)
from algorix.exceptions import (
    DataIntegrityError,
    DataUnavailableError,
    SourceUnreachableError,
)
from algorix.models import DeliveryRecord, Exchange, Instrument, InstrumentType
from algorix.storage import Database, DeliveryRepository, InstrumentRepository

SESSION = date(2026, 9, 18)

# Mirrors the live file: leading spaces after every comma, mixed series,
# and "'-'" where delivery was not published.
BHAVCOPY_CSV = (
    "SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, "
    "LAST_PRICE, CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS, "
    "NO_OF_TRADES, DELIV_QTY, DELIV_PER\n"
    "RELIANCE, EQ, 18-Sep-2026, 1243.90, 1245.00, 1247.30, 1226.40, 1226.40, "
    "1226.40, 1235.00, 15122715, 186765.00, 250000, 11645440, 77.01\n"
    "TCS, EQ, 18-Sep-2026, 3000.00, 3010.00, 3050.00, 2990.00, 3040.00, "
    "3040.00, 3020.00, 1000000, 30200.00, 50000, 450000, 45.00\n"
    "AARNAV, BE, 18-Sep-2026, 38.62, 38.81, 39.29, 38.81, 38.81, 38.98, "
    "39.12, 135, 0.05, 10, '-', '-'\n"
    "NODELIV, EQ, 18-Sep-2026, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00, "
    "10.00, 500, 0.05, 5, '-', '-'\n"
)


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "test.db")
    database.migrate()
    return database


@pytest.fixture
def instrument_ids(db):
    repo = InstrumentRepository(db)
    return {
        symbol: repo.upsert(
            Instrument(
                symbol=symbol,
                exchange=Exchange.NSE,
                instrument_type=InstrumentType.EQUITY,
                yahoo_symbol=f"{symbol}.NS",
            )
        )
        for symbol in ("RELIANCE", "TCS", "NODELIV")
    }


# --------------------------------------------------------------------------
# URL building
# --------------------------------------------------------------------------


def test_url_uses_ddmmyyyy():
    assert build_bhavcopy_url(SESSION).endswith("sec_bhavdata_full_18092026.csv")


def test_url_zero_pads_single_digit_dates():
    assert build_bhavcopy_url(date(2026, 1, 5)).endswith("_05012026.csv")


# --------------------------------------------------------------------------
# Parsing -- positive
# --------------------------------------------------------------------------


def test_parses_delivery_records():
    result = parse_bhavcopy(BHAVCOPY_CSV, expected_date=SESSION)

    assert result.session_date == SESSION
    assert result.records["RELIANCE"].traded_quantity == 15122715
    assert result.records["RELIANCE"].delivered_quantity == 11645440


def test_delivery_pct_matches_nse_published_figure():
    """NSE reports 77.01% for RELIANCE -- our computation must agree."""
    result = parse_bhavcopy(BHAVCOPY_CSV, expected_date=SESSION)

    assert result.records["RELIANCE"].delivery_pct == pytest.approx(77.01, abs=0.01)


def test_whitespace_padding_is_stripped():
    """Live headers are " SERIES", " DATE1" -- unstripped parsing yields nothing."""
    result = parse_bhavcopy(BHAVCOPY_CSV, expected_date=SESSION)

    assert "RELIANCE" in result.records
    assert result.records["TCS"].traded_quantity == 1000000


def test_non_equity_series_is_excluded():
    """AARNAV is BE series -- a different instrument class."""
    result = parse_bhavcopy(BHAVCOPY_CSV, expected_date=SESSION)

    assert "AARNAV" not in result.records


def test_other_series_can_be_requested():
    result = parse_bhavcopy(BHAVCOPY_CSV, expected_date=SESSION, series=("BE",))

    # AARNAV is BE but has no delivery figure, so it lands in unavailable.
    assert "AARNAV" in result.unavailable
    assert "RELIANCE" not in result.records


def test_expected_date_may_be_omitted():
    result = parse_bhavcopy(BHAVCOPY_CSV)

    assert result.session_date == SESSION


def test_len_reports_record_count():
    assert len(parse_bhavcopy(BHAVCOPY_CSV, expected_date=SESSION)) == 2


# -- the `'-'` case ---------------------------------------------------------


def test_unpublished_delivery_is_unavailable_not_zero():
    """`'-'` means not published. Zero would assert nobody took delivery."""
    result = parse_bhavcopy(BHAVCOPY_CSV, expected_date=SESSION)

    assert "NODELIV" in result.unavailable
    assert "NODELIV" not in result.records


def test_unavailable_list_is_deduplicated_and_sorted():
    result = parse_bhavcopy(BHAVCOPY_CSV, expected_date=SESSION)

    assert result.unavailable == sorted(set(result.unavailable))


# --------------------------------------------------------------------------
# Parsing -- negative
# --------------------------------------------------------------------------


def test_empty_bhavcopy_is_rejected():
    with pytest.raises(DataUnavailableError, match="empty"):
        parse_bhavcopy("")


def test_html_block_page_is_not_parsed():
    with pytest.raises(SourceUnreachableError, match="HTML page instead"):
        parse_bhavcopy("<!DOCTYPE html><html><body>Denied</body></html>")


def test_missing_columns_are_rejected():
    with pytest.raises(DataIntegrityError, match="missing required column"):
        parse_bhavcopy("SYMBOL, SERIES\nRELIANCE, EQ\n")


def test_wrong_session_file_is_rejected():
    """Guards against attributing one session's delivery data to another."""
    with pytest.raises(DataIntegrityError, match="Refusing to attribute"):
        parse_bhavcopy(BHAVCOPY_CSV, expected_date=date(2026, 9, 17))


def test_unparseable_date_is_rejected():
    bad = (
        "SYMBOL, SERIES, DATE1, TTL_TRD_QNTY, DELIV_QTY\n"
        "RELIANCE, EQ, not-a-date, 100, 50\n"
    )

    with pytest.raises(DataIntegrityError, match="unparseable date"):
        parse_bhavcopy(bad)


def test_file_without_requested_series_is_rejected():
    only_be = (
        "SYMBOL, SERIES, DATE1, TTL_TRD_QNTY, DELIV_QTY\n"
        "AARNAV, BE, 18-Sep-2026, 135, 60\n"
    )

    with pytest.raises(DataUnavailableError, match="no rows for series"):
        parse_bhavcopy(only_be, series=("EQ",))


def test_delivery_exceeding_traded_is_quarantined():
    """Incoherent row goes to unavailable rather than poisoning the record."""
    bad = (
        "SYMBOL, SERIES, DATE1, TTL_TRD_QNTY, DELIV_QTY\n"
        "RELIANCE, EQ, 18-Sep-2026, 100, 150\n"
        "TCS, EQ, 18-Sep-2026, 100, 50\n"
    )

    result = parse_bhavcopy(bad)

    assert "RELIANCE" in result.unavailable
    assert "TCS" in result.records


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------


def test_ingest_stores_records(db, instrument_ids):
    result = parse_bhavcopy(BHAVCOPY_CSV, expected_date=SESSION)

    report = ingest_delivery(db, SESSION, instrument_ids, result=result)

    assert report.stored == 2
    stored = DeliveryRepository(db).get_range(
        instrument_ids["RELIANCE"], SESSION, SESSION
    )
    assert stored[0].delivery_pct == pytest.approx(77.01, abs=0.01)


def test_ingest_reports_unavailable_symbols(db, instrument_ids):
    result = parse_bhavcopy(BHAVCOPY_CSV, expected_date=SESSION)

    report = ingest_delivery(db, SESSION, instrument_ids, result=result)

    assert report.unavailable == ["NODELIV"]
    assert report.is_clean is False


def test_ingest_reports_symbols_absent_from_file(db, instrument_ids):
    repo = InstrumentRepository(db)
    instrument_ids["GHOST"] = repo.upsert(
        Instrument(
            symbol="GHOST",
            exchange=Exchange.NSE,
            instrument_type=InstrumentType.EQUITY,
        )
    )
    result = parse_bhavcopy(BHAVCOPY_CSV, expected_date=SESSION)

    report = ingest_delivery(db, SESSION, instrument_ids, result=result)

    assert report.not_in_file == ["GHOST"]


def test_unavailable_symbol_stores_no_row(db, instrument_ids):
    """Absence must stay absent -- not a zero-delivery row."""
    result = parse_bhavcopy(BHAVCOPY_CSV, expected_date=SESSION)
    ingest_delivery(db, SESSION, instrument_ids, result=result)

    assert (
        DeliveryRepository(db).get_range(
            instrument_ids["NODELIV"], SESSION, SESSION
        )
        == []
    )


def test_ingest_is_idempotent(db, instrument_ids):
    result = parse_bhavcopy(BHAVCOPY_CSV, expected_date=SESSION)

    ingest_delivery(db, SESSION, instrument_ids, result=result)
    ingest_delivery(db, SESSION, instrument_ids, result=result)

    stored = DeliveryRepository(db).get_range(
        instrument_ids["RELIANCE"], SESSION, SESSION
    )
    assert len(stored) == 1


def test_ingest_with_no_requested_symbols(db):
    result = parse_bhavcopy(BHAVCOPY_CSV, expected_date=SESSION)

    report = ingest_delivery(db, SESSION, {}, result=result)

    assert report.stored == 0
    assert report.is_clean is True


# --------------------------------------------------------------------------
# Live network check -- deselect with: -m "not network"
# --------------------------------------------------------------------------


@pytest.mark.network
def test_live_bhavcopy_fetch():
    from algorix.delivery import NseBhavcopyClient

    result = NseBhavcopyClient().fetch(SESSION)

    assert result.session_date == SESSION
    assert len(result) > 1000
    assert result.records["RELIANCE"].delivery_pct == pytest.approx(77.01, abs=0.05)
    # The real file does carry unpublished figures.
    assert result.unavailable == sorted(set(result.unavailable))


@pytest.mark.network
def test_live_bhavcopy_for_a_holiday_serves_the_previous_session():
    """NSE does not 404 on a holiday -- it serves the prior session's file.

    Verified live: requesting 14 Sep 2026 (an NSE holiday) returns HTTP 200
    carrying 11 Sep data. Without the expected_date guard this would silently
    stamp an entire market's delivery figures with the wrong session date,
    which no downstream check would catch.
    """
    from algorix.delivery import NseBhavcopyClient

    with pytest.raises(DataIntegrityError, match="Refusing to attribute"):
        NseBhavcopyClient().fetch(date(2026, 9, 14))
