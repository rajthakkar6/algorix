"""Tests for index constituency parsing, syncing and point-in-time queries."""

from datetime import date

import pytest

from algorix.exceptions import (
    DataIntegrityError,
    DataUnavailableError,
    SourceUnreachableError,
)
from algorix.storage import Database
from algorix.universe import (
    NIFTY_50,
    ConstituencyRepository,
    ConstituentRecord,
    parse_constituents_csv,
    sync_nifty_50,
)

VALID_CSV = """Company Name,Industry,Symbol,Series,ISIN Code
Reliance Industries Ltd.,Oil Gas & Consumable Fuels,RELIANCE,EQ,INE002A01018
Tata Consultancy Services Ltd.,Information Technology,TCS,EQ,INE467B01029
Infosys Ltd.,Information Technology,INFY,EQ,INE009A01021
"""


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "test.db")
    database.migrate()
    return database


@pytest.fixture
def repo(db):
    return ConstituencyRepository(db)


def records(*symbols: str) -> list[ConstituentRecord]:
    return [ConstituentRecord(symbol=s, name=f"{s} Ltd.") for s in symbols]


# --------------------------------------------------------------------------
# Parsing -- positive
# --------------------------------------------------------------------------


def test_parses_valid_csv():
    parsed = parse_constituents_csv(VALID_CSV)

    assert [r.symbol for r in parsed] == ["RELIANCE", "TCS", "INFY"]
    assert parsed[0].name == "Reliance Industries Ltd."
    assert parsed[0].isin == "INE002A01018"


def test_parsing_tolerates_trailing_blank_lines():
    parsed = parse_constituents_csv(VALID_CSV + "\n\n")

    assert len(parsed) == 3


def test_parsing_strips_whitespace_and_upcases_symbols():
    csv_text = (
        "Company Name,Industry,Symbol,Series,ISIN Code\n"
        "Infosys Ltd. ,  IT , infy , EQ , INE009A01021 \n"
    )

    parsed = parse_constituents_csv(csv_text)

    assert parsed[0].symbol == "INFY"
    assert parsed[0].name == "Infosys Ltd."


def test_record_maps_to_yfinance_symbol():
    instrument = ConstituentRecord(symbol="TCS", name="TCS Ltd.").to_instrument()

    assert instrument.symbol == "TCS"
    assert instrument.yahoo_symbol == "TCS.NS"


# -- Parsing -- negative ----------------------------------------------------


def test_empty_csv_is_rejected():
    with pytest.raises(DataUnavailableError, match="empty"):
        parse_constituents_csv("")


def test_whitespace_only_csv_is_rejected():
    with pytest.raises(DataUnavailableError, match="empty"):
        parse_constituents_csv("   \n  \n")


def test_header_without_rows_is_rejected():
    with pytest.raises(DataUnavailableError, match="no usable rows"):
        parse_constituents_csv("Company Name,Industry,Symbol,Series,ISIN Code\n")


def test_missing_symbol_column_is_rejected():
    with pytest.raises(DataIntegrityError, match="missing required column"):
        parse_constituents_csv("Company Name,Industry\nInfosys Ltd.,IT\n")


def test_html_block_page_is_not_parsed_as_data():
    """NSE serves an HTML block page with HTTP 200 when rate-limiting.

    Parsing it would yield plausible-looking garbage symbols.
    """
    html = "<!DOCTYPE html><html><body>Access Denied</body></html>"

    with pytest.raises(SourceUnreachableError, match="HTML page instead of CSV"):
        parse_constituents_csv(html)


def test_lowercase_html_block_page_is_caught():
    with pytest.raises(SourceUnreachableError, match="HTML page instead of CSV"):
        parse_constituents_csv("<html><body>blocked</body></html>")


def test_duplicate_symbols_are_rejected():
    csv_text = (
        "Company Name,Industry,Symbol,Series,ISIN Code\n"
        "Infosys Ltd.,IT,INFY,EQ,INE009A01021\n"
        "Infosys Ltd.,IT,INFY,EQ,INE009A01021\n"
    )

    with pytest.raises(DataIntegrityError, match="Duplicate symbol"):
        parse_constituents_csv(csv_text)


# --------------------------------------------------------------------------
# Sync -- positive
# --------------------------------------------------------------------------


def test_first_sync_adds_all(repo):
    result = repo.sync(NIFTY_50, records("RELIANCE", "TCS"), date(2026, 9, 18))

    assert result.added == ["RELIANCE", "TCS"]
    assert result.removed == []
    assert result.changed is True
    assert repo.current_constituents(NIFTY_50) == ["RELIANCE", "TCS"]


def test_resync_with_same_list_is_a_noop(repo):
    repo.sync(NIFTY_50, records("RELIANCE", "TCS"), date(2026, 9, 18))

    result = repo.sync(NIFTY_50, records("RELIANCE", "TCS"), date(2026, 9, 21))

    assert result.added == []
    assert result.removed == []
    assert result.unchanged == ["RELIANCE", "TCS"]
    assert result.changed is False


def test_rebalance_adds_and_removes(repo):
    repo.sync(NIFTY_50, records("RELIANCE", "TCS"), date(2026, 9, 18))

    result = repo.sync(NIFTY_50, records("RELIANCE", "INFY"), date(2026, 9, 21))

    assert result.added == ["INFY"]
    assert result.removed == ["TCS"]
    assert repo.current_constituents(NIFTY_50) == ["INFY", "RELIANCE"]


def test_sync_from_parsed_csv(db):
    result = sync_nifty_50(
        db, date(2026, 9, 18), records=parse_constituents_csv(VALID_CSV)
    )

    assert sorted(result.added) == ["INFY", "RELIANCE", "TCS"]


def test_instrument_ids_resolve(repo):
    repo.sync(NIFTY_50, records("RELIANCE", "TCS"), date(2026, 9, 18))

    ids = repo.instrument_ids(NIFTY_50)

    assert len(ids) == 2
    assert all(isinstance(i, int) for i in ids)


# -- Point-in-time queries: the survivorship-bias guard ---------------------


def test_departed_member_still_visible_on_earlier_date(repo):
    """The whole point: a backtest on an old date sees the old universe."""
    repo.sync(NIFTY_50, records("RELIANCE", "TCS"), date(2026, 9, 18))
    repo.sync(NIFTY_50, records("RELIANCE", "INFY"), date(2026, 9, 21))

    assert repo.current_constituents(NIFTY_50, on=date(2026, 9, 18)) == [
        "RELIANCE",
        "TCS",
    ]


def test_new_member_absent_before_joining(repo):
    repo.sync(NIFTY_50, records("RELIANCE", "TCS"), date(2026, 9, 18))
    repo.sync(NIFTY_50, records("RELIANCE", "INFY"), date(2026, 9, 21))

    assert "INFY" not in repo.current_constituents(NIFTY_50, on=date(2026, 9, 18))


def test_effective_to_is_exclusive(repo):
    """A membership closed on D does not include D."""
    repo.sync(NIFTY_50, records("RELIANCE", "TCS"), date(2026, 9, 18))
    repo.sync(NIFTY_50, records("RELIANCE"), date(2026, 9, 21))

    assert "TCS" in repo.current_constituents(NIFTY_50, on=date(2026, 9, 20))
    assert "TCS" not in repo.current_constituents(NIFTY_50, on=date(2026, 9, 21))


def test_query_before_any_membership_is_empty(repo):
    repo.sync(NIFTY_50, records("RELIANCE"), date(2026, 9, 18))

    assert repo.current_constituents(NIFTY_50, on=date(2026, 1, 1)) == []


def test_rejoining_member_is_tracked_across_periods(repo):
    repo.sync(NIFTY_50, records("TCS"), date(2026, 1, 5))
    repo.sync(NIFTY_50, records("RELIANCE"), date(2026, 4, 1))
    repo.sync(NIFTY_50, records("RELIANCE", "TCS"), date(2026, 9, 1))

    assert "TCS" in repo.current_constituents(NIFTY_50, on=date(2026, 2, 1))
    assert "TCS" not in repo.current_constituents(NIFTY_50, on=date(2026, 5, 1))
    assert "TCS" in repo.current_constituents(NIFTY_50, on=date(2026, 9, 15))


def test_indices_are_independent(repo):
    repo.sync(NIFTY_50, records("RELIANCE"), date(2026, 9, 18))
    repo.sync("NIFTYNEXT50", records("TCS"), date(2026, 9, 18))

    assert repo.current_constituents(NIFTY_50) == ["RELIANCE"]
    assert repo.current_constituents("NIFTYNEXT50") == ["TCS"]


# -- Sync -- negative -------------------------------------------------------


def test_empty_sync_is_refused(repo):
    """A silently-empty fetch must not wipe the universe."""
    repo.sync(NIFTY_50, records("RELIANCE", "TCS"), date(2026, 9, 18))

    with pytest.raises(DataUnavailableError, match="Refusing to sync"):
        repo.sync(NIFTY_50, [], date(2026, 9, 21))

    assert repo.current_constituents(NIFTY_50) == ["RELIANCE", "TCS"]


def test_unknown_index_has_no_constituents(repo):
    assert repo.current_constituents("NOSUCHINDEX") == []


def test_instrument_ids_of_unknown_index_is_empty(repo):
    assert repo.instrument_ids("NOSUCHINDEX") == []


# --------------------------------------------------------------------------
# Live network check -- deselect with: -m "not network"
# --------------------------------------------------------------------------


@pytest.mark.network
def test_live_nse_fetch_returns_fifty_constituents():
    from algorix.universe import NseIndexClient

    parsed = NseIndexClient().fetch_nifty_50()

    assert len(parsed) == 50
    assert all(r.symbol.isupper() for r in parsed)
