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

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path

from algorix.exceptions import StorageError
from algorix.models import (
    AnnouncementRecord,
    Bar,
    CalendarPolicy,
    ChartDrawing,
    DeliveryRecord,
    EarningsSurpriseRecord,
    EventType,
    Exchange,
    ExtractedEvent,
    Instrument,
    InstrumentType,
    Materiality,
    Polarity,
    QaQuery,
    WatchlistEntry,
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

# NSE's own index constituent feed carries a sector classification
# ("Industry") that was being parsed into ConstituentRecord and then
# discarded -- nothing persisted it. It is free (same request already made
# for the universe sync) and is the input C3 (sector concentration) and any
# future sector-relative scoring need. Nullable: metals, indices and FX have
# no sector, and existing rows predate this column.
_SCHEMA_V5 = """
ALTER TABLE instruments
    ADD COLUMN industry TEXT;
"""

# NSE's corporate announcements endpoint hands out its own globally unique
# id per announcement (`seq_id`) -- unlike every other table here, that is
# the natural primary key on its own, not a composite of instrument and
# date. The same announcement re-fetched on an overlapping date-range pull
# upserts onto itself instead of duplicating.
_SCHEMA_V6 = """
CREATE TABLE IF NOT EXISTS corporate_announcements (
    seq_id             TEXT    PRIMARY KEY,
    instrument_id      INTEGER NOT NULL REFERENCES instruments(id) ON DELETE CASCADE,
    announced_at       TEXT    NOT NULL,
    category           TEXT    NOT NULL,
    announcement_text  TEXT    NOT NULL,
    attachment_url     TEXT,
    isin               TEXT,
    source             TEXT    NOT NULL,
    ingested_at        TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_announcements_instrument
    ON corporate_announcements (instrument_id, announced_at);
"""

# G4 (INDICATORS.md Bucket G) -- LLM structured extraction over G1
# announcements. Two tables:
#
# `extracted_events` is one row per announcement (PK = the announcement's own
# seq_id, so it upserts the same way the rest of this schema does). Never
# joined into scoring -- G0 forbids sentiment as a directional score input.
#
# `extraction_batches` tracks Anthropic Batch API submissions. Batches are
# asynchronous (results can take up to 24h), so submission and collection are
# two separate operations that may run on different days -- this table is
# what lets a later run find a batch it submitted earlier and ask whether it
# is done yet.
_SCHEMA_V7 = """
CREATE TABLE IF NOT EXISTS extracted_events (
    announcement_seq_id  TEXT    PRIMARY KEY
                                  REFERENCES corporate_announcements(seq_id)
                                  ON DELETE CASCADE,
    event_type           TEXT    NOT NULL,
    entities              TEXT   NOT NULL,
    polarity               TEXT  NOT NULL,
    materiality             TEXT NOT NULL,
    source_credibility      TEXT NOT NULL,
    risk_flag             INTEGER NOT NULL,
    risk_reason            TEXT,
    model_id                TEXT NOT NULL,
    prompt_version       INTEGER NOT NULL,
    extracted_at             TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS extraction_batches (
    batch_id         TEXT    PRIMARY KEY,
    submitted_at     TEXT    NOT NULL,
    item_count       INTEGER NOT NULL,
    model_id         TEXT    NOT NULL,
    prompt_version   INTEGER NOT NULL,
    status           TEXT    NOT NULL,
    collected_at     TEXT
);
"""

# A8 (INDICATORS.md Bucket F -> promoted): Post-Earnings Announcement Drift.
# One row per completed quarterly report -- yfinance's own estimate/actual,
# not NSE's (NSE publishes the raw result, not a pre-earnings consensus to
# compare it against). Keyed on (instrument_id, report_date): a company
# reports at most once per date, and re-ingesting is a small, cheap full
# re-pull (yfinance returns ~25 rows total, not an incremental feed), so
# upserting the same date twice must update in place, not duplicate.
_SCHEMA_V8 = """
CREATE TABLE IF NOT EXISTS earnings_surprises (
    instrument_id   INTEGER NOT NULL REFERENCES instruments(id) ON DELETE CASCADE,
    report_date     TEXT    NOT NULL,
    eps_estimate    REAL    NOT NULL,
    eps_actual      REAL    NOT NULL,
    surprise_pct    REAL    NOT NULL,
    source          TEXT    NOT NULL,
    ingested_at     TEXT    NOT NULL,
    PRIMARY KEY (instrument_id, report_date)
);
"""

# User-placed chart annotations (trendlines, and manually-placed
# breakout/dip markers) on the web UI's stock detail page -- see
# ChartDrawing in models.py. `points` is deliberately opaque JSON: its
# count and meaning depend on `tool_type`, which this table does not
# constrain to an enum (no CHECK(tool_type IN (...))) -- that would force a
# migration every time a new drawing tool ships, exactly what this design
# exists to avoid. The known-type allow-list lives in web/server.py
# instead, where adding one is a single dict entry. System-auto-detected
# breakout/dip highlights are NOT rows here -- they are computed fresh on
# every page load and never stored.
_SCHEMA_V9 = """
CREATE TABLE IF NOT EXISTS chart_drawings (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument_id  INTEGER NOT NULL REFERENCES instruments(id) ON DELETE CASCADE,
    tool_type      TEXT    NOT NULL CHECK (tool_type <> ''),
    points         TEXT    NOT NULL CHECK (json_valid(points)),
    created_at     TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chart_drawings_instrument
    ON chart_drawings (instrument_id);
"""

# Personal watchlist -- membership is boolean per instrument, so
# instrument_id is the primary key itself rather than a synthetic
# autoincrement id: there is no second identity to give a watchlist entry.
# UI-only, like chart_drawings -- never read by scoring.py.
_SCHEMA_V10 = """
CREATE TABLE IF NOT EXISTS watchlist (
    instrument_id  INTEGER PRIMARY KEY REFERENCES instruments(id) ON DELETE CASCADE,
    added_at       TEXT    NOT NULL
);
"""

# AI Q&A audit trail (CLAUDE.md invariant 7: every LLM verdict logged with
# prompt version and model ID). Stores the full assembled context, not a
# hash -- this is a personal, low-volume tool, and full context is what
# actually lets a bad answer be debugged later; a hash tells you nothing.
# UI-only, like chart_drawings/watchlist -- never read by scoring.py.
_SCHEMA_V11 = """
CREATE TABLE IF NOT EXISTS qa_queries (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument_id  INTEGER NOT NULL REFERENCES instruments(id) ON DELETE CASCADE,
    question       TEXT    NOT NULL CHECK (question <> ''),
    context        TEXT    NOT NULL CHECK (json_valid(context)),
    answer         TEXT    NOT NULL CHECK (answer <> ''),
    model_id       TEXT    NOT NULL,
    prompt_version INTEGER NOT NULL,
    asked_at       TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_qa_queries_instrument
    ON qa_queries (instrument_id, asked_at);
"""

#: Ordered migrations. Each runs once, in version order, against databases
#: older than it. Never edit a migration that has shipped -- add a new one.
_MIGRATIONS: list[tuple[int, str]] = [
    (1, _SCHEMA_V1),
    (2, _SCHEMA_V2),
    (3, _SCHEMA_V3),
    (4, _SCHEMA_V4),
    (5, _SCHEMA_V5),
    (6, _SCHEMA_V6),
    (7, _SCHEMA_V7),
    (8, _SCHEMA_V8),
    (9, _SCHEMA_V9),
    (10, _SCHEMA_V10),
    (11, _SCHEMA_V11),
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
                     is_active, calendar_policy, industry, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (symbol, exchange) DO UPDATE SET
                    instrument_type = excluded.instrument_type,
                    name            = COALESCE(excluded.name, instruments.name),
                    yahoo_symbol    = COALESCE(excluded.yahoo_symbol,
                                               instruments.yahoo_symbol),
                    is_active       = excluded.is_active,
                    calendar_policy = excluded.calendar_policy,
                    -- NSE occasionally serves the constituent list without
                    -- Industry populated; do not let a transient miss erase
                    -- a value already on record.
                    industry        = COALESCE(excluded.industry,
                                               instruments.industry)
                """,
                (
                    instrument.symbol,
                    str(instrument.exchange),
                    str(instrument.instrument_type),
                    instrument.name,
                    instrument.yahoo_symbol,
                    int(instrument.is_active),
                    str(instrument.calendar_policy),
                    instrument.industry,
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

    def get_by_id(self, instrument_id: int) -> Instrument | None:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM instruments WHERE id = ?", (instrument_id,)
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


class AnnouncementRepository:
    """Read/write access to NSE corporate announcements (INDICATORS.md G1)."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def latest_announced_at(self, instrument_id: int) -> datetime | None:
        """Most recent stored announcement's timestamp, or None if there is
        no history yet. Mirrors BarRepository.latest_session -- the same
        incremental-refresh watermark, one layer over."""
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT MAX(announced_at) AS latest FROM corporate_announcements "
                "WHERE instrument_id = ?",
                (instrument_id,),
            ).fetchone()
        if not row or row["latest"] is None:
            return None
        return datetime.fromisoformat(row["latest"])

    def upsert_many(
        self,
        pairs: Iterable[tuple[AnnouncementRecord, int]],
        source: str,
    ) -> int:
        """Store announcements paired with their already-resolved instrument id.

        Takes (record, instrument_id) pairs rather than one instrument_id
        for the whole batch: unlike delivery (one bhavcopy file = one
        session, symbol-keyed within it), announcements are typically
        fetched per-symbol already, but nothing here should assume that --
        a market-wide pull naturally mixes symbols in one response.

        Keyed on `seq_id` (NSE's own id), so the same announcement seen
        again on an overlapping refetch updates in place rather than
        duplicating.
        """
        rows = [
            (
                rec.seq_id,
                instrument_id,
                rec.announced_at.isoformat(),
                rec.category,
                rec.text,
                rec.attachment_url,
                rec.isin,
                source,
                _utc_now_iso(),
            )
            for rec, instrument_id in pairs
        ]
        if not rows:
            return 0

        with self.db.connect() as conn:
            try:
                conn.executemany(
                    """
                    INSERT INTO corporate_announcements
                        (seq_id, instrument_id, announced_at, category,
                         announcement_text, attachment_url, isin, source,
                         ingested_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (seq_id) DO UPDATE SET
                        instrument_id     = excluded.instrument_id,
                        announced_at      = excluded.announced_at,
                        category          = excluded.category,
                        announcement_text = excluded.announcement_text,
                        attachment_url    = excluded.attachment_url,
                        isin              = excluded.isin,
                        source            = excluded.source,
                        ingested_at       = excluded.ingested_at
                    """,
                    rows,
                )
            except sqlite3.IntegrityError as exc:
                raise StorageError(
                    f"Rejected announcement data: {exc}"
                ) from exc
        return len(rows)

    def get_range(
        self, instrument_id: int, start: date, end: date
    ) -> list[AnnouncementRecord]:
        """Announcements for one instrument, newest first.

        `start`/`end` bound the announcement *date* (not just session date
        -- announcements carry a time of day and can land after market
        close), so both ends are inclusive calendar days.
        """
        if start > end:
            raise ValueError(f"start {start} is after end {end}")

        with self.db.connect() as conn:
            symbol_row = conn.execute(
                "SELECT symbol FROM instruments WHERE id = ?", (instrument_id,)
            ).fetchone()
            if symbol_row is None:
                return []
            symbol = symbol_row["symbol"]

            rows = conn.execute(
                """
                SELECT seq_id, announced_at, category, announcement_text,
                       attachment_url, isin
                FROM corporate_announcements
                WHERE instrument_id = ?
                  AND date(announced_at) BETWEEN ? AND ?
                ORDER BY announced_at DESC
                """,
                (instrument_id, start.isoformat(), end.isoformat()),
            ).fetchall()

        return [
            AnnouncementRecord(
                seq_id=r["seq_id"],
                symbol=symbol,
                announced_at=datetime.fromisoformat(r["announced_at"]),
                category=r["category"],
                text=r["announcement_text"],
                attachment_url=r["attachment_url"],
                isin=r["isin"],
            )
            for r in rows
        ]

    def get_by_seq_ids(self, seq_ids: Iterable[str]) -> list[AnnouncementRecord]:
        """Look up announcements by their own id, not by instrument+date.

        The natural companion to `EventExtractionRepository.unextracted_seq_ids`,
        which returns bare ids -- this is how the extraction pipeline gets
        the actual text back for them.
        """
        seq_ids = list(seq_ids)
        if not seq_ids:
            return []

        with self.db.connect() as conn:
            placeholders = ",".join("?" * len(seq_ids))
            rows = conn.execute(
                f"""
                SELECT a.seq_id, i.symbol, a.announced_at, a.category,
                       a.announcement_text, a.attachment_url, a.isin
                FROM corporate_announcements a
                JOIN instruments i ON i.id = a.instrument_id
                WHERE a.seq_id IN ({placeholders})
                """,
                seq_ids,
            ).fetchall()

        return [
            AnnouncementRecord(
                seq_id=r["seq_id"],
                symbol=r["symbol"],
                announced_at=datetime.fromisoformat(r["announced_at"]),
                category=r["category"],
                text=r["announcement_text"],
                attachment_url=r["attachment_url"],
                isin=r["isin"],
            )
            for r in rows
        ]


class EventExtractionRepository:
    """Read/write access to G4 structured extractions.

    Never read by `scoring.py` -- see ExtractedEvent's docstring. This
    exists for the journal, the digest's event/risk-flag lines, and future
    contrarian-extreme detection, all of which are separate from the score.
    """

    def __init__(self, db: Database) -> None:
        self.db = db

    def upsert_many(self, events: Iterable[ExtractedEvent]) -> int:
        """Store extractions, keyed on the source announcement's seq_id.

        One extraction per announcement by construction (the prompt asks
        for exactly one judgement per item), so a second extraction for the
        same announcement -- a rerun after a prompt change, say -- replaces
        the first rather than producing two conflicting rows for one event.
        """
        rows = [
            (
                e.announcement_seq_id,
                str(e.event_type),
                json.dumps(e.entities),
                str(e.polarity),
                str(e.materiality),
                e.source_credibility,
                int(e.risk_flag),
                e.risk_reason,
                e.model_id,
                e.prompt_version,
                e.extracted_at.isoformat(),
            )
            for e in events
        ]
        if not rows:
            return 0

        with self.db.connect() as conn:
            try:
                conn.executemany(
                    """
                    INSERT INTO extracted_events
                        (announcement_seq_id, event_type, entities, polarity,
                         materiality, source_credibility, risk_flag,
                         risk_reason, model_id, prompt_version, extracted_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (announcement_seq_id) DO UPDATE SET
                        event_type          = excluded.event_type,
                        entities            = excluded.entities,
                        polarity            = excluded.polarity,
                        materiality         = excluded.materiality,
                        source_credibility  = excluded.source_credibility,
                        risk_flag           = excluded.risk_flag,
                        risk_reason         = excluded.risk_reason,
                        model_id            = excluded.model_id,
                        prompt_version      = excluded.prompt_version,
                        extracted_at        = excluded.extracted_at
                    """,
                    rows,
                )
            except sqlite3.IntegrityError as exc:
                raise StorageError(f"Rejected extracted event data: {exc}") from exc
        return len(rows)

    def unextracted_seq_ids(self, limit: int | None = None) -> list[str]:
        """Announcements with no extraction yet, oldest first.

        Oldest first so a submission cap (Anthropic allows up to 100,000
        items per batch, but a nightly job should not need anywhere near
        that) never permanently starves an old announcement behind a
        constant stream of new ones.
        """
        query = """
            SELECT a.seq_id
            FROM corporate_announcements a
            LEFT JOIN extracted_events e ON e.announcement_seq_id = a.seq_id
            WHERE e.announcement_seq_id IS NULL
            ORDER BY a.announced_at ASC
        """
        params: tuple = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (limit,)

        with self.db.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [r["seq_id"] for r in rows]

    def get(self, announcement_seq_id: str) -> ExtractedEvent | None:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM extracted_events WHERE announcement_seq_id = ?",
                (announcement_seq_id,),
            ).fetchone()
        return _row_to_extracted_event(row) if row else None


def _row_to_extracted_event(row: sqlite3.Row) -> ExtractedEvent:
    return ExtractedEvent(
        announcement_seq_id=row["announcement_seq_id"],
        event_type=EventType(row["event_type"]),
        entities=json.loads(row["entities"]),
        polarity=Polarity(row["polarity"]),
        materiality=Materiality(row["materiality"]),
        source_credibility=row["source_credibility"],
        risk_flag=bool(row["risk_flag"]),
        risk_reason=row["risk_reason"],
        model_id=row["model_id"],
        prompt_version=row["prompt_version"],
        extracted_at=datetime.fromisoformat(row["extracted_at"]),
    )


@dataclass(frozen=True)
class ExtractionBatch:
    """One submitted Anthropic Batch API job, tracked for later collection."""

    batch_id: str
    submitted_at: datetime
    item_count: int
    model_id: str
    prompt_version: int
    status: str
    collected_at: datetime | None = None


class ExtractionBatchRepository:
    """Tracks submitted extraction batches so a later run can find and poll
    them -- the Batch API is asynchronous (results can take up to 24h), so
    submission and collection are necessarily separate operations."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def record_submission(self, batch: ExtractionBatch) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO extraction_batches
                    (batch_id, submitted_at, item_count, model_id,
                     prompt_version, status, collected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    batch.batch_id,
                    batch.submitted_at.isoformat(),
                    batch.item_count,
                    batch.model_id,
                    batch.prompt_version,
                    batch.status,
                    batch.collected_at.isoformat() if batch.collected_at else None,
                ),
            )

    def mark_collected(self, batch_id: str, collected_at: datetime) -> None:
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE extraction_batches SET status = 'collected', "
                "collected_at = ? WHERE batch_id = ?",
                (collected_at.isoformat(), batch_id),
            )

    def pending(self) -> list[ExtractionBatch]:
        """Submitted batches not yet marked collected, oldest first."""
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM extraction_batches WHERE status = 'submitted' "
                "ORDER BY submitted_at ASC"
            ).fetchall()
        return [_row_to_batch(r) for r in rows]


def _row_to_batch(row: sqlite3.Row) -> ExtractionBatch:
    return ExtractionBatch(
        batch_id=row["batch_id"],
        submitted_at=datetime.fromisoformat(row["submitted_at"]),
        item_count=row["item_count"],
        model_id=row["model_id"],
        prompt_version=row["prompt_version"],
        status=row["status"],
        collected_at=(
            datetime.fromisoformat(row["collected_at"])
            if row["collected_at"]
            else None
        ),
    )


class EarningsSurpriseRepository:
    """Read/write access to A8 earnings-surprise history."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def upsert_many(
        self,
        instrument_id: int,
        records: Iterable[EarningsSurpriseRecord],
        source: str,
    ) -> int:
        rows = [
            (
                instrument_id,
                rec.report_date.isoformat(),
                rec.eps_estimate,
                rec.eps_actual,
                rec.surprise_pct,
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
                    INSERT INTO earnings_surprises
                        (instrument_id, report_date, eps_estimate, eps_actual,
                         surprise_pct, source, ingested_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (instrument_id, report_date) DO UPDATE SET
                        eps_estimate = excluded.eps_estimate,
                        eps_actual   = excluded.eps_actual,
                        surprise_pct = excluded.surprise_pct,
                        source       = excluded.source,
                        ingested_at  = excluded.ingested_at
                    """,
                    rows,
                )
            except sqlite3.IntegrityError as exc:
                raise StorageError(
                    f"Rejected earnings surprise data for instrument "
                    f"{instrument_id}: {exc}"
                ) from exc
        return len(rows)

    def get_range(
        self, instrument_id: int, start: date, end: date
    ) -> list[EarningsSurpriseRecord]:
        """Reports between `start` and `end` inclusive, oldest first.

        Mirrors `DeliveryRepository.get_range`/`AnnouncementRepository.get_range`
        exactly: the *caller* is what enforces point-in-time correctness by
        passing `end=as_of`, not this method or the indicator function that
        consumes its result -- the same convention every other time-bounded
        signal in this codebase follows, so a backtest replaying a past
        session cannot see a report that had not happened yet.
        """
        if start > end:
            raise ValueError(f"start {start} is after end {end}")

        with self.db.connect() as conn:
            symbol_row = conn.execute(
                "SELECT symbol FROM instruments WHERE id = ?", (instrument_id,)
            ).fetchone()
            if symbol_row is None:
                return []
            symbol = symbol_row["symbol"]

            rows = conn.execute(
                """
                SELECT report_date, eps_estimate, eps_actual, surprise_pct
                FROM earnings_surprises
                WHERE instrument_id = ? AND report_date BETWEEN ? AND ?
                ORDER BY report_date ASC
                """,
                (instrument_id, start.isoformat(), end.isoformat()),
            ).fetchall()

        return [
            EarningsSurpriseRecord(
                symbol=symbol,
                report_date=date.fromisoformat(r["report_date"]),
                eps_estimate=r["eps_estimate"],
                eps_actual=r["eps_actual"],
                surprise_pct=r["surprise_pct"],
            )
            for r in rows
        ]


class ChartDrawingRepository:
    """Read/write access to user-placed chart annotations (see ChartDrawing
    in models.py). UI-only -- never read by scoring.py, scan.py, or the
    journal.
    """

    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, drawing: ChartDrawing) -> ChartDrawing:
        """Insert one drawing. Returns it with `id`/`created_at` filled in.

        Unlike InstrumentRepository.upsert, a drawing has no natural key to
        re-select by afterward -- it's identified only by its own
        autoincrement id -- so cursor.lastrowid is used directly rather than
        a follow-up SELECT.
        """
        created_at = _utc_now_iso()
        with self.db.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO chart_drawings
                    (instrument_id, tool_type, points, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    drawing.instrument_id,
                    drawing.tool_type,
                    json.dumps(drawing.points),
                    created_at,
                ),
            )
            new_id = int(cursor.lastrowid)
        return replace(
            drawing, id=new_id, created_at=datetime.fromisoformat(created_at)
        )

    def list_for(self, instrument_id: int) -> list[ChartDrawing]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chart_drawings WHERE instrument_id = ? "
                "ORDER BY created_at",
                (instrument_id,),
            ).fetchall()
        return [_row_to_chart_drawing(r) for r in rows]

    def delete(self, drawing_id: int, instrument_id: int) -> bool:
        """Delete one drawing, scoped to its instrument. Returns True if a
        row changed -- the rowcount idiom from
        InstrumentRepository.deactivate(), the only delete-by-id precedent
        in this file before this one.

        `instrument_id` is required, not optional: a delete request for a
        real drawing id under the wrong symbol must not delete it -- this
        makes cross-instrument deletion structurally impossible, not just
        checked for.
        """
        with self.db.connect() as conn:
            cursor = conn.execute(
                "DELETE FROM chart_drawings WHERE id = ? AND instrument_id = ?",
                (drawing_id, instrument_id),
            )
            return cursor.rowcount > 0


def _row_to_chart_drawing(row: sqlite3.Row) -> ChartDrawing:
    return ChartDrawing(
        id=row["id"],
        instrument_id=row["instrument_id"],
        tool_type=row["tool_type"],
        points=json.loads(row["points"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


class WatchlistRepository:
    """Read/write access to the personal watchlist. UI-only -- never read
    by scoring.py, scan.py, or the journal, the same boundary
    ChartDrawingRepository draws.
    """

    def __init__(self, db: Database) -> None:
        self.db = db

    def add(self, instrument_id: int) -> WatchlistEntry:
        """Add an instrument to the watchlist. Idempotent -- adding an
        already-watchlisted instrument is a no-op, not an error, matching
        the upsert convention every other write in this schema follows.
        """
        added_at = _utc_now_iso()
        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO watchlist (instrument_id, added_at) VALUES (?, ?) "
                "ON CONFLICT (instrument_id) DO NOTHING",
                (instrument_id, added_at),
            )
            row = conn.execute(
                "SELECT added_at FROM watchlist WHERE instrument_id = ?",
                (instrument_id,),
            ).fetchone()
        return WatchlistEntry(instrument_id, datetime.fromisoformat(row["added_at"]))

    def remove(self, instrument_id: int) -> bool:
        with self.db.connect() as conn:
            cursor = conn.execute(
                "DELETE FROM watchlist WHERE instrument_id = ?", (instrument_id,)
            )
            return cursor.rowcount > 0

    def contains(self, instrument_id: int) -> bool:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM watchlist WHERE instrument_id = ?", (instrument_id,)
            ).fetchone()
        return row is not None

    def list_all(self) -> list[WatchlistEntry]:
        """Oldest-added first."""
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT instrument_id, added_at FROM watchlist ORDER BY added_at"
            ).fetchall()
        return [
            WatchlistEntry(r["instrument_id"], datetime.fromisoformat(r["added_at"]))
            for r in rows
        ]


class QaQueryRepository:
    """Read/write access to the AI Q&A audit trail (see QaQuery in
    models.py). UI-only -- never read by scoring.py, scan.py, or the
    journal.
    """

    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, query: QaQuery) -> QaQuery:
        """Insert one Q&A audit row. Returns it with `id`/`asked_at` filled
        in -- same cursor.lastrowid idiom as ChartDrawingRepository.create,
        no natural key to re-select by."""
        asked_at = _utc_now_iso()
        with self.db.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO qa_queries
                    (instrument_id, question, context, answer, model_id,
                     prompt_version, asked_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    query.instrument_id,
                    query.question,
                    json.dumps(query.context),
                    query.answer,
                    query.model_id,
                    query.prompt_version,
                    asked_at,
                ),
            )
            new_id = int(cursor.lastrowid)
        return replace(query, id=new_id, asked_at=datetime.fromisoformat(asked_at))

    def history_for(self, instrument_id: int, limit: int = 20) -> list[QaQuery]:
        """Most recent first."""
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM qa_queries WHERE instrument_id = ? "
                "ORDER BY asked_at DESC LIMIT ?",
                (instrument_id, limit),
            ).fetchall()
        return [_row_to_qa_query(r) for r in rows]


def _row_to_qa_query(row: sqlite3.Row) -> QaQuery:
    return QaQuery(
        id=row["id"],
        instrument_id=row["instrument_id"],
        question=row["question"],
        context=json.loads(row["context"]),
        answer=row["answer"],
        model_id=row["model_id"],
        prompt_version=row["prompt_version"],
        asked_at=datetime.fromisoformat(row["asked_at"]),
    )


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
        industry=row["industry"],
    )
