"""Tests for NSE corporate announcements ingestion (INDICATORS.md G1).

Fixture JSON mirrors the live shape verified 2026-09-24: `seq_id`, `symbol`,
`an_dt`, `desc`, `attchmntText` reliably populated; `bflag`, `csvName`,
`old_new`, `orgid` reliably null; `smIndustry` inconsistently populated.
"""

from datetime import date, datetime

import pytest

from algorix.announcements import (
    SOURCE_NSE_ANNOUNCEMENTS,
    AnnouncementFetchResult,
    ingest_announcements,
    parse_announcements,
)
from algorix.exceptions import DataIntegrityError
from algorix.models import AnnouncementRecord, Exchange, Instrument, InstrumentType
from algorix.storage import AnnouncementRepository, Database, InstrumentRepository

RELIANCE_ROW = {
    "an_dt": "23-Sep-2026 18:51:51",
    "attFileSize": "233.75 KB",
    "attchmntFile": "https://nsearchives.nseindia.com/corporate/x_23092026.pdf",
    "attchmntText": "This is further to the disclosure dated September 17, 2026.",
    "bflag": None,
    "csvName": None,
    "desc": "Updates",
    "difference": "00:00:01",
    "dt": "23092026185151",
    "exchdisstime": "23-Sep-2026 18:51:51",
    "fileSize": "233.75 KB",
    "hasXbrl": True,
    "old_new": None,
    "orgid": None,
    "seq_id": "106789883",
    "smIndustry": None,
    "sm_isin": "INE002A01018",
    "sm_name": "Reliance Industries Limited",
    "sort_date": "2026-09-23 18:51:51",
    "symbol": "RELIANCE",
}


def row(**overrides):
    merged = dict(RELIANCE_ROW)
    merged.update(overrides)
    return merged


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "test.db")
    database.migrate()
    return database


@pytest.fixture
def reliance_id(db):
    return InstrumentRepository(db).upsert(
        Instrument(
            symbol="RELIANCE",
            exchange=Exchange.NSE,
            instrument_type=InstrumentType.EQUITY,
        )
    )


# ---------------------------------------------------------------------------
# parse_announcements -- positive
# ---------------------------------------------------------------------------


def test_parses_a_well_formed_row():
    result = parse_announcements([row()])

    assert len(result.records) == 1
    record = result.records[0]
    assert record.seq_id == "106789883"
    assert record.symbol == "RELIANCE"
    assert record.category == "Updates"
    assert record.announced_at == datetime(2026, 9, 23, 18, 51, 51)
    assert record.isin == "INE002A01018"
    assert result.skipped == 0


def test_symbol_is_upper_cased():
    result = parse_announcements([row(symbol="reliance")])

    assert result.records[0].symbol == "RELIANCE"


def test_earliest_reflects_the_oldest_record():
    result = parse_announcements(
        [
            row(seq_id="1", an_dt="20-Sep-2026 10:00:00"),
            row(seq_id="2", an_dt="15-Sep-2026 09:00:00"),
            row(seq_id="3", an_dt="22-Sep-2026 11:00:00"),
        ]
    )

    assert result.earliest == date(2026, 9, 15)


def test_missing_desc_falls_back_to_unspecified():
    """NSE's own contract does not guarantee `desc` -- a blank category is
    not a reason to drop an otherwise-good announcement."""
    result = parse_announcements([row(desc="")])

    assert result.records[0].category == "Unspecified"


def test_null_optional_fields_do_not_break_parsing():
    """bflag, csvName, old_new, orgid, smIndustry are null in the live feed
    -- the parser must not depend on any of them."""
    result = parse_announcements(
        [row(bflag=None, csvName=None, old_new=None, orgid=None, smIndustry=None)]
    )

    assert len(result.records) == 1


def test_missing_attachment_url_is_none_not_an_error():
    result = parse_announcements([row(attchmntFile=None)])

    assert result.records[0].attachment_url is None


# ---------------------------------------------------------------------------
# parse_announcements -- negative
# ---------------------------------------------------------------------------


def test_non_list_payload_is_rejected():
    with pytest.raises(DataIntegrityError, match="expected a JSON list"):
        parse_announcements({"error": "not a list"})


def test_empty_list_is_a_legitimate_zero_result():
    """No announcements in a window is a real, common state -- not a fault."""
    result = parse_announcements([])

    assert result.records == []
    assert result.skipped == 0
    assert result.earliest is None


def test_row_missing_seq_id_is_skipped_not_fatal():
    result = parse_announcements([row(seq_id=""), row(seq_id="2")])

    assert len(result.records) == 1
    assert result.skipped == 1


def test_row_missing_symbol_is_skipped():
    result = parse_announcements([row(symbol="")])

    assert result.records == []
    assert result.skipped == 1


def test_row_missing_text_is_skipped():
    """A textless announcement is a parse failure, not a smaller one."""
    result = parse_announcements([row(attchmntText="")])

    assert result.records == []
    assert result.skipped == 1


def test_row_with_unparseable_timestamp_is_skipped():
    result = parse_announcements([row(an_dt="not a date")])

    assert result.records == []
    assert result.skipped == 1


def test_row_missing_timestamp_entirely_is_skipped():
    bad = row()
    del bad["an_dt"]
    result = parse_announcements([bad])

    assert result.records == []
    assert result.skipped == 1


def test_one_bad_row_does_not_abandon_the_rest():
    """Mirrors refresh.py's 'failures are collected, never fatal' rule."""
    result = parse_announcements([row(seq_id=""), row(seq_id="2"), row(seq_id="3")])

    assert len(result.records) == 2
    assert result.skipped == 1


def test_non_dict_row_is_skipped_not_fatal():
    result = parse_announcements(["not a dict", row()])

    assert len(result.records) == 1
    assert result.skipped == 1


# ---------------------------------------------------------------------------
# ingest_announcements -- positive
# ---------------------------------------------------------------------------


def test_ingest_stores_records(db, reliance_id):
    result = parse_announcements([row()])

    report = ingest_announcements(
        db, "RELIANCE", reliance_id, date(2026, 9, 1), date(2026, 9, 24), result=result
    )

    assert report.stored == 1
    assert report.skipped == 0
    stored = AnnouncementRepository(db).get_range(
        reliance_id, date(2026, 9, 1), date(2026, 9, 24)
    )
    assert len(stored) == 1
    assert stored[0].seq_id == "106789883"


def test_ingest_is_idempotent(db, reliance_id):
    """The same announcement seen twice (an overlapping refetch) upserts
    onto itself instead of duplicating -- seq_id is the natural key."""
    result = parse_announcements([row()])

    ingest_announcements(
        db, "RELIANCE", reliance_id, date(2026, 9, 1), date(2026, 9, 24), result=result
    )
    ingest_announcements(
        db, "RELIANCE", reliance_id, date(2026, 9, 1), date(2026, 9, 24), result=result
    )

    stored = AnnouncementRepository(db).get_range(
        reliance_id, date(2026, 9, 1), date(2026, 9, 24)
    )
    assert len(stored) == 1


def test_ingest_records_source(db, reliance_id):
    result = parse_announcements([row()])

    ingest_announcements(
        db, "RELIANCE", reliance_id, date(2026, 9, 1), date(2026, 9, 24), result=result
    )

    with db.connect() as conn:
        source = conn.execute(
            "SELECT source FROM corporate_announcements WHERE seq_id = ?",
            ("106789883",),
        ).fetchone()["source"]
    assert source == SOURCE_NSE_ANNOUNCEMENTS


def test_ingest_reports_earliest_returned(db, reliance_id):
    result = parse_announcements(
        [row(seq_id="1", an_dt="10-Sep-2026 09:00:00"), row(seq_id="2")]
    )

    report = ingest_announcements(
        db, "RELIANCE", reliance_id, date(2026, 9, 1), date(2026, 9, 24), result=result
    )

    assert report.earliest_returned == date(2026, 9, 10)


# ---------------------------------------------------------------------------
# ingest_announcements -- negative
# ---------------------------------------------------------------------------


def test_ingest_with_zero_announcements_is_not_an_error(db, reliance_id):
    """A quiet stock with no announcements in the window is a legitimate,
    common outcome -- ingestion must succeed with a report saying so."""
    result = parse_announcements([])

    report = ingest_announcements(
        db, "RELIANCE", reliance_id, date(2026, 9, 1), date(2026, 9, 24), result=result
    )

    assert report.stored == 0
    assert report.earliest_returned is None


def test_ingest_reports_skipped_rows(db, reliance_id):
    result = parse_announcements([row(seq_id=""), row(seq_id="2")])

    report = ingest_announcements(
        db, "RELIANCE", reliance_id, date(2026, 9, 1), date(2026, 9, 24), result=result
    )

    assert report.stored == 1
    assert report.skipped == 1


def test_client_fetch_rejects_start_after_end():
    from algorix.announcements import NseAnnouncementsClient

    with pytest.raises(ValueError, match="after"):
        NseAnnouncementsClient().fetch("RELIANCE", date(2026, 9, 24), date(2026, 9, 1))


# ---------------------------------------------------------------------------
# AnnouncementRecord model -- negative (defence in depth beneath the parser)
# ---------------------------------------------------------------------------


def test_record_rejects_empty_seq_id():
    with pytest.raises(DataIntegrityError, match="seq_id"):
        AnnouncementRecord(
            seq_id="", symbol="RELIANCE",
            announced_at=datetime(2026, 9, 23), category="Updates", text="x",
        )


def test_record_rejects_empty_symbol():
    with pytest.raises(DataIntegrityError, match="symbol"):
        AnnouncementRecord(
            seq_id="1", symbol="",
            announced_at=datetime(2026, 9, 23), category="Updates", text="x",
        )


def test_record_rejects_empty_text():
    with pytest.raises(DataIntegrityError, match="text"):
        AnnouncementRecord(
            seq_id="1", symbol="RELIANCE",
            announced_at=datetime(2026, 9, 23), category="Updates", text="",
        )


# ---------------------------------------------------------------------------
# AnnouncementRepository -- storage-layer positive/negative
# ---------------------------------------------------------------------------


def test_repository_get_range_orders_newest_first(db, reliance_id):
    repo = AnnouncementRepository(db)
    older = AnnouncementRecord(
        seq_id="1", symbol="RELIANCE", announced_at=datetime(2026, 9, 10),
        category="Updates", text="older",
    )
    newer = AnnouncementRecord(
        seq_id="2", symbol="RELIANCE", announced_at=datetime(2026, 9, 20),
        category="Updates", text="newer",
    )
    repo.upsert_many([(older, reliance_id), (newer, reliance_id)], source="test")

    stored = repo.get_range(reliance_id, date(2026, 9, 1), date(2026, 9, 30))

    assert [r.seq_id for r in stored] == ["2", "1"]


def test_repository_get_range_excludes_outside_window(db, reliance_id):
    repo = AnnouncementRepository(db)
    inside = AnnouncementRecord(
        seq_id="1", symbol="RELIANCE", announced_at=datetime(2026, 9, 15),
        category="Updates", text="inside",
    )
    outside = AnnouncementRecord(
        seq_id="2", symbol="RELIANCE", announced_at=datetime(2026, 8, 1),
        category="Updates", text="outside",
    )
    repo.upsert_many([(inside, reliance_id), (outside, reliance_id)], source="test")

    stored = repo.get_range(reliance_id, date(2026, 9, 1), date(2026, 9, 30))

    assert [r.seq_id for r in stored] == ["1"]


def test_repository_get_range_rejects_start_after_end(db):
    with pytest.raises(ValueError, match="after"):
        AnnouncementRepository(db).get_range(1, date(2026, 9, 24), date(2026, 9, 1))


def test_repository_get_range_for_unknown_instrument_is_empty(db):
    stored = AnnouncementRepository(db).get_range(
        999999, date(2026, 9, 1), date(2026, 9, 30)
    )

    assert stored == []


def test_repository_upsert_many_with_empty_iterable_is_a_noop(db):
    stored = AnnouncementRepository(db).upsert_many([], source="test")

    assert stored == 0


def test_latest_announced_at_returns_the_most_recent_timestamp(db, reliance_id):
    repo = AnnouncementRepository(db)
    older = AnnouncementRecord(
        seq_id="1", symbol="RELIANCE", announced_at=datetime(2026, 9, 10, 9, 0),
        category="Updates", text="older",
    )
    newer = AnnouncementRecord(
        seq_id="2", symbol="RELIANCE", announced_at=datetime(2026, 9, 20, 15, 30),
        category="Updates", text="newer",
    )
    repo.upsert_many([(older, reliance_id), (newer, reliance_id)], source="test")

    assert repo.latest_announced_at(reliance_id) == datetime(2026, 9, 20, 15, 30)


def test_latest_announced_at_with_no_history_is_none(db, reliance_id):
    assert AnnouncementRepository(db).latest_announced_at(reliance_id) is None
