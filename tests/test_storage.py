"""Tests for SQLite persistence."""

import sqlite3
from datetime import date

import pytest

from algorix.exceptions import StorageError
from algorix.models import (
    Bar,
    CalendarPolicy,
    DeliveryRecord,
    Exchange,
    Instrument,
    InstrumentType,
)
from algorix.storage import (
    SCHEMA_VERSION,
    BarRepository,
    Database,
    DeliveryRepository,
    InstrumentRepository,
)


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "test.db")
    database.migrate()
    return database


@pytest.fixture
def instruments(db):
    return InstrumentRepository(db)


@pytest.fixture
def bars(db):
    return BarRepository(db)


@pytest.fixture
def reliance(instruments):
    return instruments.upsert(
        Instrument(
            symbol="RELIANCE",
            exchange=Exchange.NSE,
            instrument_type=InstrumentType.EQUITY,
            yahoo_symbol="RELIANCE.NS",
        )
    )


def make_bar(day: date, close: float = 100.0) -> Bar:
    return Bar(
        session_date=day,
        open=close - 1,
        high=close + 2,
        low=close - 2,
        close=close,
        volume=1000,
    )


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------


def test_migrate_creates_schema(tmp_path):
    database = Database(tmp_path / "new.db")

    assert database.migrate() == SCHEMA_VERSION
    assert database.schema_version() == SCHEMA_VERSION


def test_migrate_is_idempotent(tmp_path):
    database = Database(tmp_path / "new.db")
    database.migrate()
    database.migrate()

    assert database.schema_version() == SCHEMA_VERSION


def test_migrate_creates_parent_directories(tmp_path):
    database = Database(tmp_path / "deep" / "nested" / "algorix.db")
    database.migrate()

    assert (tmp_path / "deep" / "nested" / "algorix.db").exists()


def test_schema_version_of_missing_database_is_zero(tmp_path):
    assert Database(tmp_path / "absent.db").schema_version() == 0


def test_existing_older_database_upgrades_in_place(tmp_path):
    """A v1 database must gain v2 tables without losing its data.

    Guards the migration path itself: users have an existing database, and a
    migrate() that only ever created a fresh schema would either fail or
    silently skip new tables.
    """
    from algorix.storage import _MIGRATIONS, _utc_now_iso

    path = tmp_path / "legacy.db"
    database = Database(path)

    # Build a database at version 1 only, writing the row with raw v1-shaped
    # SQL -- the repository writes columns added by later migrations.
    with database.connect() as conn:
        conn.executescript(_MIGRATIONS[0][1])
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (1, ?)",
            (_utc_now_iso(),),
        )
        conn.execute(
            """
            INSERT INTO instruments
                (symbol, exchange, instrument_type, name, yahoo_symbol,
                 is_active, created_at)
            VALUES ('LEGACY', 'NSE', 'EQUITY', 'Legacy Co', NULL, 1, ?)
            """,
            (_utc_now_iso(),),
        )
    assert database.schema_version() == 1

    assert database.migrate() == SCHEMA_VERSION

    # Later tables and columns now exist, and the pre-existing row survived
    # -- with the new column taking its documented default.
    with database.connect() as conn:
        conn.execute("SELECT COUNT(*) FROM index_constituents").fetchone()
    stored = InstrumentRepository(database).get("LEGACY", Exchange.NSE)
    assert stored is not None
    assert stored.name == "Legacy Co"
    assert stored.calendar_policy is CalendarPolicy.NSE


def test_migration_records_every_applied_version(tmp_path):
    """Version-agnostic: every declared migration must be recorded."""
    from algorix.storage import _MIGRATIONS

    database = Database(tmp_path / "versions.db")
    database.migrate()

    with database.connect() as conn:
        rows = conn.execute(
            "SELECT version FROM schema_version ORDER BY version"
        ).fetchall()

    assert [r["version"] for r in rows] == [v for v, _ in _MIGRATIONS]


def test_newer_schema_is_refused(tmp_path):
    """An older build must not write to a database a newer build created."""
    database = Database(tmp_path / "future.db")
    database.migrate()
    with database.connect() as conn:
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (SCHEMA_VERSION + 5, "2026-01-01T00:00:00+00:00"),
        )

    with pytest.raises(StorageError, match="newer than this code supports"):
        database.migrate()


def test_foreign_keys_are_enforced(db):
    """SQLite defaults foreign keys OFF -- verify the PRAGMA actually applies."""
    with pytest.raises(sqlite3.IntegrityError):
        with db.connect() as conn:
            conn.execute(
                """
                INSERT INTO ohlcv_bars
                    (instrument_id, session_date, open, high, low, close,
                     volume, source, ingested_at)
                VALUES (9999, '2026-09-18', 1, 1, 1, 1, 0, 'test', 'now')
                """
            )


def test_schema_rejects_incoherent_bar_directly(db, reliance):
    """Integrity does not depend on the Python layer being used."""
    with pytest.raises(sqlite3.IntegrityError):
        with db.connect() as conn:
            conn.execute(
                """
                INSERT INTO ohlcv_bars
                    (instrument_id, session_date, open, high, low, close,
                     volume, source, ingested_at)
                VALUES (?, '2026-09-18', 100, 98, 99, 98.5, 10, 'test', 'now')
                """,
                (reliance,),
            )


# --------------------------------------------------------------------------
# Instruments
# --------------------------------------------------------------------------


def test_upsert_returns_id(instruments):
    instrument_id = instruments.upsert(
        Instrument(
            symbol="TCS",
            exchange=Exchange.NSE,
            instrument_type=InstrumentType.EQUITY,
        )
    )

    assert isinstance(instrument_id, int)


def test_upsert_is_idempotent(instruments):
    spec = Instrument(
        symbol="INFY", exchange=Exchange.NSE, instrument_type=InstrumentType.EQUITY
    )

    first = instruments.upsert(spec)
    second = instruments.upsert(spec)

    assert first == second
    assert len(instruments.list_active()) == 1


def test_upsert_updates_mutable_fields(instruments):
    instruments.upsert(
        Instrument(
            symbol="INFY",
            exchange=Exchange.NSE,
            instrument_type=InstrumentType.EQUITY,
            name="Infosys",
        )
    )
    instruments.upsert(
        Instrument(
            symbol="INFY",
            exchange=Exchange.NSE,
            instrument_type=InstrumentType.EQUITY,
            name="Infosys Limited",
        )
    )

    stored = instruments.get("INFY", Exchange.NSE)
    assert stored is not None
    assert stored.name == "Infosys Limited"


def test_upsert_does_not_erase_existing_name_with_none(instruments):
    """A partial refresh must not blank fields it does not carry."""
    instruments.upsert(
        Instrument(
            symbol="INFY",
            exchange=Exchange.NSE,
            instrument_type=InstrumentType.EQUITY,
            name="Infosys",
            yahoo_symbol="INFY.NS",
        )
    )
    instruments.upsert(
        Instrument(
            symbol="INFY",
            exchange=Exchange.NSE,
            instrument_type=InstrumentType.EQUITY,
        )
    )

    stored = instruments.get("INFY", Exchange.NSE)
    assert stored is not None
    assert stored.name == "Infosys"
    assert stored.yahoo_symbol == "INFY.NS"


def test_same_symbol_on_two_exchanges_is_distinct(instruments):
    nse = instruments.upsert(
        Instrument(
            symbol="RELIANCE",
            exchange=Exchange.NSE,
            instrument_type=InstrumentType.EQUITY,
        )
    )
    bse = instruments.upsert(
        Instrument(
            symbol="RELIANCE",
            exchange=Exchange.BSE,
            instrument_type=InstrumentType.EQUITY,
        )
    )

    assert nse != bse


def test_get_unknown_instrument_returns_none(instruments):
    assert instruments.get("NOSUCH", Exchange.NSE) is None


def test_list_active_filters_by_type(instruments):
    instruments.upsert(
        Instrument(
            symbol="TCS", exchange=Exchange.NSE, instrument_type=InstrumentType.EQUITY
        )
    )
    instruments.upsert(
        Instrument(
            symbol="GOLD",
            exchange=Exchange.MCX,
            instrument_type=InstrumentType.COMMODITY,
        )
    )

    equities = instruments.list_active(InstrumentType.EQUITY)
    commodities = instruments.list_active(InstrumentType.COMMODITY)

    assert [i.symbol for i in equities] == ["TCS"]
    assert [i.symbol for i in commodities] == ["GOLD"]


def test_deactivate_removes_from_active_list(instruments):
    instruments.upsert(
        Instrument(
            symbol="TCS", exchange=Exchange.NSE, instrument_type=InstrumentType.EQUITY
        )
    )

    assert instruments.deactivate("TCS", Exchange.NSE) is True
    assert instruments.list_active() == []
    # Deactivated, not deleted -- history stays addressable.
    assert instruments.get("TCS", Exchange.NSE) is not None


def test_deactivate_unknown_instrument_reports_no_change(instruments):
    assert instruments.deactivate("NOSUCH", Exchange.NSE) is False


# --------------------------------------------------------------------------
# Bars
# --------------------------------------------------------------------------


def test_upsert_and_read_back_bars(bars, reliance):
    written = bars.upsert_many(
        reliance,
        [make_bar(date(2026, 9, 17), 100.0), make_bar(date(2026, 9, 18), 102.0)],
        source="test",
    )

    stored = bars.get_range(reliance, date(2026, 9, 17), date(2026, 9, 18))

    assert written == 2
    assert [b.close for b in stored] == [100.0, 102.0]


def test_bars_are_returned_oldest_first(bars, reliance):
    bars.upsert_many(
        reliance,
        [make_bar(date(2026, 9, 18)), make_bar(date(2026, 9, 16))],
        source="test",
    )

    stored = bars.get_range(reliance, date(2026, 9, 1), date(2026, 9, 30))

    assert [b.session_date for b in stored] == [date(2026, 9, 16), date(2026, 9, 18)]


def test_reingesting_a_session_updates_rather_than_duplicates(bars, reliance):
    """A daily cron can run twice -- that must not double-write history."""
    bars.upsert_many(reliance, [make_bar(date(2026, 9, 18), 100.0)], source="first")
    bars.upsert_many(reliance, [make_bar(date(2026, 9, 18), 111.0)], source="second")

    stored = bars.get_range(reliance, date(2026, 9, 18), date(2026, 9, 18))

    assert len(stored) == 1
    assert stored[0].close == 111.0


def test_upsert_empty_iterable_is_a_noop(bars, reliance):
    assert bars.upsert_many(reliance, [], source="test") == 0


def test_get_range_is_inclusive_of_both_ends(bars, reliance):
    bars.upsert_many(
        reliance,
        [make_bar(date(2026, 9, d)) for d in (16, 17, 18)],
        source="test",
    )

    stored = bars.get_range(reliance, date(2026, 9, 16), date(2026, 9, 18))

    assert len(stored) == 3


def test_get_range_excludes_outside_dates(bars, reliance):
    bars.upsert_many(
        reliance,
        [make_bar(date(2026, 9, d)) for d in (15, 16, 17, 18)],
        source="test",
    )

    stored = bars.get_range(reliance, date(2026, 9, 16), date(2026, 9, 17))

    assert [b.session_date for b in stored] == [date(2026, 9, 16), date(2026, 9, 17)]


def test_latest_session_returns_most_recent(bars, reliance):
    bars.upsert_many(
        reliance,
        [make_bar(date(2026, 9, 16)), make_bar(date(2026, 9, 18))],
        source="test",
    )

    assert bars.latest_session(reliance) == date(2026, 9, 18)


def test_latest_session_is_none_with_no_history(bars, reliance):
    assert bars.latest_session(reliance) is None


def test_bars_are_scoped_per_instrument(bars, instruments, reliance):
    tcs = instruments.upsert(
        Instrument(
            symbol="TCS", exchange=Exchange.NSE, instrument_type=InstrumentType.EQUITY
        )
    )
    bars.upsert_many(reliance, [make_bar(date(2026, 9, 18), 100.0)], source="test")

    assert bars.get_range(tcs, date(2026, 9, 1), date(2026, 9, 30)) == []


# -- gap detection ----------------------------------------------------------


def test_missing_sessions_reports_gaps(bars, reliance):
    bars.upsert_many(
        reliance,
        [make_bar(date(2026, 9, 16)), make_bar(date(2026, 9, 18))],
        source="test",
    )

    missing = bars.missing_sessions(
        reliance, [date(2026, 9, 16), date(2026, 9, 17), date(2026, 9, 18)]
    )

    assert missing == [date(2026, 9, 17)]


def test_missing_sessions_empty_when_complete(bars, reliance):
    sessions = [date(2026, 9, 16), date(2026, 9, 17)]
    bars.upsert_many(reliance, [make_bar(d) for d in sessions], source="test")

    assert bars.missing_sessions(reliance, sessions) == []


def test_missing_sessions_reports_all_when_no_history(bars, reliance):
    sessions = [date(2026, 9, 16), date(2026, 9, 17)]

    assert bars.missing_sessions(reliance, sessions) == sessions


def test_missing_sessions_with_empty_expectation(bars, reliance):
    assert bars.missing_sessions(reliance, []) == []


def test_missing_sessions_deduplicates_and_sorts(bars, reliance):
    missing = bars.missing_sessions(
        reliance, [date(2026, 9, 18), date(2026, 9, 16), date(2026, 9, 18)]
    )

    assert missing == [date(2026, 9, 16), date(2026, 9, 18)]


# -- negative ---------------------------------------------------------------


def test_get_range_rejects_reversed_dates(bars, reliance):
    with pytest.raises(ValueError, match="is after end"):
        bars.get_range(reliance, date(2026, 9, 18), date(2026, 9, 16))


def test_bars_for_unknown_instrument_are_rejected(bars):
    """A typo'd instrument id must fail loudly, not create orphan rows."""
    with pytest.raises(StorageError, match="Rejected bars"):
        bars.upsert_many(9999, [make_bar(date(2026, 9, 18))], source="test")


def test_deleting_instrument_cascades_to_bars(db, instruments, bars, reliance):
    bars.upsert_many(reliance, [make_bar(date(2026, 9, 18))], source="test")

    with db.connect() as conn:
        conn.execute("DELETE FROM instruments WHERE id = ?", (reliance,))

    assert bars.get_range(reliance, date(2026, 9, 1), date(2026, 9, 30)) == []


# --------------------------------------------------------------------------
# Delivery data
# --------------------------------------------------------------------------


def test_delivery_round_trip(db, reliance):
    repo = DeliveryRepository(db)
    repo.upsert_many(
        reliance,
        [
            DeliveryRecord(
                session_date=date(2026, 9, 18),
                traded_quantity=1000,
                delivered_quantity=400,
            )
        ],
        source="test",
    )

    stored = repo.get_range(reliance, date(2026, 9, 18), date(2026, 9, 18))

    assert len(stored) == 1
    assert stored[0].delivery_pct == pytest.approx(40.0)


def test_delivery_reingest_updates(db, reliance):
    repo = DeliveryRepository(db)
    day = date(2026, 9, 18)
    repo.upsert_many(
        reliance,
        [DeliveryRecord(session_date=day, traded_quantity=1000, delivered_quantity=400)],
        source="first",
    )
    repo.upsert_many(
        reliance,
        [DeliveryRecord(session_date=day, traded_quantity=1000, delivered_quantity=600)],
        source="second",
    )

    stored = repo.get_range(reliance, day, day)

    assert len(stored) == 1
    assert stored[0].delivered_quantity == 600


def test_delivery_absent_is_empty_not_zero(db, reliance):
    """No delivery row means "not published", which is not "zero delivery"."""
    repo = DeliveryRepository(db)

    assert repo.get_range(reliance, date(2026, 9, 18), date(2026, 9, 18)) == []


def test_delivery_for_unknown_instrument_is_rejected(db):
    repo = DeliveryRepository(db)

    with pytest.raises(StorageError, match="Rejected delivery data"):
        repo.upsert_many(
            9999,
            [
                DeliveryRecord(
                    session_date=date(2026, 9, 18),
                    traded_quantity=10,
                    delivered_quantity=5,
                )
            ],
            source="test",
        )
