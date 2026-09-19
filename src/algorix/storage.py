"""SQLite persistence.

SQLite is in the Python standard library -- no server, no setup, one file on
disk. That is the right shape for a single-user tool, and the schema is kept
portable enough to move to Postgres if concurrency ever demands it.

Two behaviours are deliberate:

1. **Integrity is enforced in the schema, not only in Python.** CHECK
   constraints make an incoherent bar (high below low, non-positive price)
   physically unstorable. Validation code can be bypassed; the schema cannot.

2. **Writes are idempotent.** A daily job can run twice -- after a failure, or
   because the user triggered it manually. Re-ingesting a session updates the
   row rather than duplicating or erroring.

Dates are stored as ISO `YYYY-MM-DD` text, which sorts chronologically as a
string, and are converted explicitly at the boundary (Python 3.12 deprecated
sqlite3's implicit date adapters).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path

from algorix.exceptions import StorageError
from algorix.models import (
    Bar,
    CalendarPolicy,
    DeliveryRecord,
    Exchange,
    Instrument,
    InstrumentType,
)

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER NOT NULL,
    applied_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS instruments (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol           TEXT    NOT NULL,
    exchange         TEXT    NOT NULL,
    instrument_type  TEXT    NOT NULL,
    name             TEXT,
    yahoo_symbol     TEXT,
    is_active        INTEGER NOT NULL DEFAULT 1,
    created_at       TEXT    NOT NULL,
    UNIQUE (symbol, exchange)
);

-- CHECK constraints make incoherent bars unstorable. Volume may be zero (an
-- illiquid name can trade nothing); prices may not.
CREATE TABLE IF NOT EXISTS ohlcv_bars (
    instrument_id  INTEGER NOT NULL REFERENCES instruments(id) ON DELETE CASCADE,
    session_date   TEXT    NOT NULL,
    open           REAL    NOT NULL CHECK (open  > 0),
    high           REAL    NOT NULL CHECK (high  > 0),
    low            REAL    NOT NULL CHECK (low   > 0),
    close          REAL    NOT NULL CHECK (close > 0),
    volume         INTEGER NOT NULL CHECK (volume >= 0),
    source         TEXT    NOT NULL,
    ingested_at    TEXT    NOT NULL,
    PRIMARY KEY (instrument_id, session_date),
    CHECK (high >= low),
    CHECK (high >= open AND high >= close),
    CHECK (low  <= open AND low  <= close)
);

CREATE INDEX IF NOT EXISTS idx_bars_session_date
    ON ohlcv_bars (session_date);

CREATE TABLE IF NOT EXISTS delivery_data (
    instrument_id       INTEGER NOT NULL REFERENCES instruments(id) ON DELETE CASCADE,
    session_date        TEXT    NOT NULL,
    traded_quantity     INTEGER NOT NULL CHECK (traded_quantity    >= 0),
    delivered_quantity  INTEGER NOT NULL CHECK (delivered_quantity >= 0),
    source              TEXT    NOT NULL,
    ingested_at         TEXT    NOT NULL,
    PRIMARY KEY (instrument_id, session_date),
    CHECK (delivered_quantity <= traded_quantity)
);
"""

# Index membership is tracked point-in-time rather than as a flat "current 50".
# Nifty 50 rebalances semi-annually; storing only today's members would bake
# survivorship bias into every future backtest -- the index would appear to
# have always contained whatever is winning now. `effective_to` is exclusive:
# a row with effective_to = D means the instrument was NOT a member on D.
_SCHEMA_V2 = """
CREATE TABLE IF NOT EXISTS index_constituents (
    index_symbol   TEXT    NOT NULL,
    instrument_id  INTEGER NOT NULL REFERENCES instruments(id) ON DELETE CASCADE,
    effective_from TEXT    NOT NULL,
    effective_to   TEXT,
    recorded_at    TEXT    NOT NULL,
    PRIMARY KEY (index_symbol, instrument_id, effective_from),
    CHECK (effective_to IS NULL OR effective_to > effective_from)
);

CREATE INDEX IF NOT EXISTS idx_constituents_lookup
    ON index_constituents (index_symbol, effective_from, effective_to);
"""

# Instruments do not all follow the NSE calendar: COMEX futures and FX trade
# on days NSE is shut. Storing the policy per instrument keeps the phantom-bar
# filter strict where it applies and silent where it would destroy real data.
_SCHEMA_V3 = """
ALTER TABLE instruments
    ADD COLUMN calendar_policy TEXT NOT NULL DEFAULT 'NSE';
"""

# Institutional flows are market-level, not per-instrument (INDICATORS.md B4).
# NSE's endpoint serves only the latest session, so this history accrues
# forward from first run rather than being backfillable.
_SCHEMA_V4 = """
CREATE TABLE IF NOT EXISTS market_flows (
    session_date  TEXT NOT NULL,
    category      TEXT NOT NULL,
    buy_value     REAL NOT NULL,
    sell_value    REAL NOT NULL,
    net_value     REAL NOT NULL,
    source        TEXT NOT NULL,
    ingested_at   TEXT NOT NULL,
    PRIMARY KEY (session_date, category)
);
"""

#: Ordered migrations. Each runs once, in version order, against databases
#: older than it. Never edit a migration that has shipped -- add a new one.
_MIGRATIONS: list[tuple[int, str]] = [
    (1, _SCHEMA_V1),
    (2, _SCHEMA_V2),
    (3, _SCHEMA_V3),
    (4, _SCHEMA_V4),
]

SCHEMA_VERSION = _MIGRATIONS[-1][0]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    """Connection management and schema migration."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a configured connection, committing on clean exit.

        Foreign keys are enabled per-connection because SQLite defaults them
        OFF -- without this the REFERENCES clauses above are decorative.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            # WAL lets a read (e.g. an on-demand lookup) proceed while the
            # nightly ingestion is writing.
            conn.execute("PRAGMA journal_mode = WAL")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def migrate(self) -> int:
        """Create or upgrade the schema. Returns the resulting version.

        Applies only migrations newer than the stored version, so an existing
        database upgrades in place rather than needing a rebuild.
        """
        current = self.schema_version()

        if current > SCHEMA_VERSION:
            raise StorageError(
                f"Database schema version {current} is newer than this code "
                f"supports ({SCHEMA_VERSION}). Refusing to continue -- an "
                "older build must not write to a newer database."
            )
        if current == SCHEMA_VERSION:
            return SCHEMA_VERSION

        with self.connect() as conn:
            for version, script in _MIGRATIONS:
                if version <= current:
                    continue
                conn.executescript(script)
                conn.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                    (version, _utc_now_iso()),
                )
        return SCHEMA_VERSION

    def schema_version(self) -> int:
        with self.connect() as conn:
            try:
                row = conn.execute(
                    "SELECT MAX(version) AS version FROM schema_version"
                ).fetchone()
            except sqlite3.OperationalError:
                return 0
            return row["version"] if row and row["version"] is not None else 0


class InstrumentRepository:
    """Read/write access to the instrument universe."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def upsert(self, instrument: Instrument) -> int:
        """Insert or update by (symbol, exchange). Returns the instrument id."""
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO instruments
                    (symbol, exchange, instrument_type, name, yahoo_symbol,
                     is_active, calendar_policy, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (symbol, exchange) DO UPDATE SET
                    instrument_type = excluded.instrument_type,
                    name            = COALESCE(excluded.name, instruments.name),
                    yahoo_symbol    = COALESCE(excluded.yahoo_symbol,
                                               instruments.yahoo_symbol),
                    is_active       = excluded.is_active,
                    calendar_policy = excluded.calendar_policy
                """,
                (
                    instrument.symbol,
                    str(instrument.exchange),
                    str(instrument.instrument_type),
                    instrument.name,
                    instrument.yahoo_symbol,
                    int(instrument.is_active),
                    str(instrument.calendar_policy),
                    _utc_now_iso(),
                ),
            )
            row = conn.execute(
                "SELECT id FROM instruments WHERE symbol = ? AND exchange = ?",
                (instrument.symbol, str(instrument.exchange)),
            ).fetchone()
            return int(row["id"])

    def get(self, symbol: str, exchange: Exchange) -> Instrument | None:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM instruments WHERE symbol = ? AND exchange = ?",
                (symbol, str(exchange)),
            ).fetchone()
        return _row_to_instrument(row) if row else None

    def list_active(
        self, instrument_type: InstrumentType | None = None
    ) -> list[Instrument]:
        query = "SELECT * FROM instruments WHERE is_active = 1"
        params: list[object] = []
        if instrument_type is not None:
            query += " AND instrument_type = ?"
            params.append(str(instrument_type))
        query += " ORDER BY symbol"

        with self.db.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_row_to_instrument(r) for r in rows]

    def deactivate(self, symbol: str, exchange: Exchange) -> bool:
        """Mark an instrument inactive. Returns True if a row changed.

        Instruments are deactivated, never deleted: their historical bars stay
        valid and a delisted name must not vanish from the journal.
        """
        with self.db.connect() as conn:
            cursor = conn.execute(
                "UPDATE instruments SET is_active = 0 "
                "WHERE symbol = ? AND exchange = ? AND is_active = 1",
                (symbol, str(exchange)),
            )
            return cursor.rowcount > 0


class BarRepository:
    """Read/write access to OHLCV history."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def upsert_many(
        self, instrument_id: int, bars: Iterable[Bar], source: str
    ) -> int:
        """Store bars idempotently. Returns the number written."""
        rows = [
            (
                instrument_id,
                bar.session_date.isoformat(),
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.volume,
                source,
                _utc_now_iso(),
            )
            for bar in bars
        ]
        if not rows:
            return 0

        with self.db.connect() as conn:
            try:
                conn.executemany(
                    """
                    INSERT INTO ohlcv_bars
                        (instrument_id, session_date, open, high, low, close,
                         volume, source, ingested_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (instrument_id, session_date) DO UPDATE SET
                        open        = excluded.open,
                        high        = excluded.high,
                        low         = excluded.low,
                        close       = excluded.close,
                        volume      = excluded.volume,
                        source      = excluded.source,
                        ingested_at = excluded.ingested_at
                    """,
                    rows,
                )
            except sqlite3.IntegrityError as exc:
                raise StorageError(
                    f"Rejected bars for instrument {instrument_id}: {exc}"
                ) from exc
        return len(rows)

    def get_range(
        self, instrument_id: int, start: date, end: date
    ) -> list[Bar]:
        """Bars from `start` to `end` inclusive, oldest first."""
        if start > end:
            raise ValueError(f"start {start} is after end {end}")

        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT session_date, open, high, low, close, volume
                FROM ohlcv_bars
                WHERE instrument_id = ? AND session_date BETWEEN ? AND ?
                ORDER BY session_date
                """,
                (instrument_id, start.isoformat(), end.isoformat()),
            ).fetchall()

        return [
            Bar(
                session_date=date.fromisoformat(r["session_date"]),
                open=r["open"],
                high=r["high"],
                low=r["low"],
                close=r["close"],
                volume=r["volume"],
            )
            for r in rows
        ]

    def latest_session(self, instrument_id: int) -> date | None:
        """Most recent stored session, or None if there is no history."""
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT MAX(session_date) AS latest FROM ohlcv_bars "
                "WHERE instrument_id = ?",
                (instrument_id,),
            ).fetchone()
        if not row or row["latest"] is None:
            return None
        return date.fromisoformat(row["latest"])

    def missing_sessions(
        self, instrument_id: int, expected: Iterable[date]
    ) -> list[date]:
        """Which of `expected` have no stored bar.

        Gap detection is a first-class operation, not a debugging aid: a
        scoring run over silently incomplete history produces numbers that
        look valid. Callers are expected to surface a non-empty result rather
        than proceed.
        """
        expected_list = sorted(set(expected))
        if not expected_list:
            return []

        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT session_date FROM ohlcv_bars
                WHERE instrument_id = ? AND session_date BETWEEN ? AND ?
                """,
                (
                    instrument_id,
                    expected_list[0].isoformat(),
                    expected_list[-1].isoformat(),
                ),
            ).fetchall()

        stored = {date.fromisoformat(r["session_date"]) for r in rows}
        return [d for d in expected_list if d not in stored]


class DeliveryRepository:
    """Read/write access to NSE delivery data (INDICATORS.md A6)."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def upsert_many(
        self, instrument_id: int, records: Iterable[DeliveryRecord], source: str
    ) -> int:
        rows = [
            (
                instrument_id,
                rec.session_date.isoformat(),
                rec.traded_quantity,
                rec.delivered_quantity,
                source,
                _utc_now_iso(),
            )
            for rec in records
        ]
        if not rows:
            return 0

        with self.db.connect() as conn:
            try:
                conn.executemany(
                    """
                    INSERT INTO delivery_data
                        (instrument_id, session_date, traded_quantity,
                         delivered_quantity, source, ingested_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT (instrument_id, session_date) DO UPDATE SET
                        traded_quantity    = excluded.traded_quantity,
                        delivered_quantity = excluded.delivered_quantity,
                        source             = excluded.source,
                        ingested_at        = excluded.ingested_at
                    """,
                    rows,
                )
            except sqlite3.IntegrityError as exc:
                raise StorageError(
                    f"Rejected delivery data for instrument {instrument_id}: {exc}"
                ) from exc
        return len(rows)

    def sessions_present(self, start: date, end: date) -> set[date]:
        """Sessions with any delivery data stored.

        Delivery is fetched market-wide (one bhavcopy per session), so
        coverage is uniform across instruments -- table-level is the right
        granularity for deciding which sessions still need fetching.
        """
        if start > end:
            raise ValueError(f"start {start} is after end {end}")

        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT session_date FROM delivery_data "
                "WHERE session_date BETWEEN ? AND ?",
                (start.isoformat(), end.isoformat()),
            ).fetchall()
        return {date.fromisoformat(r["session_date"]) for r in rows}

    def get_range(
        self, instrument_id: int, start: date, end: date
    ) -> list[DeliveryRecord]:
        if start > end:
            raise ValueError(f"start {start} is after end {end}")

        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT session_date, traded_quantity, delivered_quantity
                FROM delivery_data
                WHERE instrument_id = ? AND session_date BETWEEN ? AND ?
                ORDER BY session_date
                """,
                (instrument_id, start.isoformat(), end.isoformat()),
            ).fetchall()

        return [
            DeliveryRecord(
                session_date=date.fromisoformat(r["session_date"]),
                traded_quantity=r["traded_quantity"],
                delivered_quantity=r["delivered_quantity"],
            )
            for r in rows
        ]


def _row_to_instrument(row: sqlite3.Row) -> Instrument:
    return Instrument(
        id=row["id"],
        symbol=row["symbol"],
        exchange=Exchange(row["exchange"]),
        instrument_type=InstrumentType(row["instrument_type"]),
        name=row["name"],
        yahoo_symbol=row["yahoo_symbol"],
        is_active=bool(row["is_active"]),
        calendar_policy=CalendarPolicy(row["calendar_policy"]),
    )
