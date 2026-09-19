"""Trade journal: every scan, preserved for later evaluation.

This is not a reporting convenience. It is the only mechanism by which the
scoring model can ever be judged. Scores are opinions until matched against
what the market subsequently did; without a durable record of what was
claimed and when, there is nothing to check and no basis for changing the
weights (INDICATORS.md open decision 3).

Three things are recorded so a stored score stays interpretable months later:

- **The full contribution breakdown**, not just the number. "Why did this
  score 82?" must be answerable after the fact.
- **The method and cohort size** (principle 0.2a), so scores derived
  differently are never silently compared.
- **The regime verdict** in force at the time, since a score's meaning
  depends on whether momentum was working that week.

CLAUDE.md invariant 7: log what the journal needs to stay interpretable, or
months of data become unusable because you cannot tell what changed
underneath it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone

from algorix.regime import MarketRegime
from algorix.scoring import ScoredUniverse, StockScore
from algorix.storage import Database

#: Bumped when the scoring logic changes in a way that makes new scores
#: incomparable to old ones. Stored per row so a mixed-version history can
#: still be analysed correctly.
SCORING_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scan_runs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_date     TEXT NOT NULL,
    run_at           TEXT NOT NULL,
    scoring_version  INTEGER NOT NULL,
    universe_label   TEXT NOT NULL,
    regime_verdict   TEXT,
    regime_summary   TEXT,
    regime_warnings  TEXT,
    UNIQUE (session_date, scoring_version, universe_label)
);

CREATE TABLE IF NOT EXISTS scan_scores (
    run_id         INTEGER NOT NULL REFERENCES scan_runs(id) ON DELETE CASCADE,
    symbol         TEXT    NOT NULL,
    score          REAL,
    unavailable    TEXT,
    eligible       INTEGER NOT NULL,
    method         TEXT    NOT NULL,
    cohort_size    INTEGER NOT NULL,
    contributions  TEXT    NOT NULL,
    missing        TEXT    NOT NULL,
    atr            REAL,
    stop_price     REAL,
    PRIMARY KEY (run_id, symbol)
);

CREATE INDEX IF NOT EXISTS idx_scan_scores_symbol ON scan_scores (symbol);
"""


@dataclass(frozen=True)
class JournalEntry:
    """A stored score, read back."""

    session_date: date
    symbol: str
    score: float | None
    eligible: bool
    method: str
    cohort_size: int
    contributions: dict[str, float]
    regime_verdict: str | None


class Journal:
    """Append-only record of scan results."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def ensure_schema(self) -> None:
        with self.db.connect() as conn:
            conn.executescript(_SCHEMA)

    def record(
        self,
        universe: ScoredUniverse,
        regime: MarketRegime | None = None,
        universe_label: str = "NIFTY50",
    ) -> int:
        """Store a scan. Re-recording the same session replaces it.

        Replacement rather than duplication keeps a re-run from inflating the
        record with near-identical rows -- but the scoring version is part of
        the key, so a logic change creates a new row rather than overwriting
        history computed by different code.
        """
        self.ensure_schema()

        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO scan_runs
                    (session_date, run_at, scoring_version, universe_label,
                     regime_verdict, regime_summary, regime_warnings)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (session_date, scoring_version, universe_label)
                DO UPDATE SET
                    run_at          = excluded.run_at,
                    regime_verdict  = excluded.regime_verdict,
                    regime_summary  = excluded.regime_summary,
                    regime_warnings = excluded.regime_warnings
                """,
                (
                    universe.as_of.isoformat(),
                    datetime.now(timezone.utc).isoformat(),
                    SCORING_VERSION,
                    universe_label,
                    str(regime.verdict) if regime else None,
                    regime.summary() if regime else None,
                    json.dumps(regime.warnings) if regime else None,
                ),
            )
            row = conn.execute(
                """
                SELECT id FROM scan_runs
                WHERE session_date = ? AND scoring_version = ?
                  AND universe_label = ?
                """,
                (universe.as_of.isoformat(), SCORING_VERSION, universe_label),
            ).fetchone()
            run_id = int(row["id"])

            conn.execute("DELETE FROM scan_scores WHERE run_id = ?", (run_id,))
            conn.executemany(
                """
                INSERT INTO scan_scores
                    (run_id, symbol, score, unavailable, eligible, method,
                     cohort_size, contributions, missing, atr, stop_price)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [_score_row(run_id, s) for s in universe.scores.values()],
            )

        return run_id

    def scores_for(self, session_date: date) -> list[JournalEntry]:
        self.ensure_schema()
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT s.*, r.session_date, r.regime_verdict
                FROM scan_scores s
                JOIN scan_runs r ON r.id = s.run_id
                WHERE r.session_date = ? AND r.scoring_version = ?
                ORDER BY s.score DESC NULLS LAST
                """,
                (session_date.isoformat(), SCORING_VERSION),
            ).fetchall()
        return [_to_entry(r) for r in rows]

    def history_for(self, symbol: str, limit: int = 90) -> list[JournalEntry]:
        """One symbol's score over time -- the basis for evaluating it."""
        self.ensure_schema()
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT s.*, r.session_date, r.regime_verdict
                FROM scan_scores s
                JOIN scan_runs r ON r.id = s.run_id
                WHERE s.symbol = ? AND r.scoring_version = ?
                ORDER BY r.session_date DESC
                LIMIT ?
                """,
                (symbol, SCORING_VERSION, limit),
            ).fetchall()
        return [_to_entry(r) for r in rows]

    def recorded_sessions(self) -> list[date]:
        self.ensure_schema()
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT session_date FROM scan_runs ORDER BY session_date"
            ).fetchall()
        return [date.fromisoformat(r["session_date"]) for r in rows]


def _score_row(run_id: int, score: StockScore) -> tuple:
    return (
        run_id,
        score.symbol,
        score.score.value if score.score.available else None,
        None if score.score.available else score.score.reason,
        int(score.eligible),
        score.method,
        score.cohort_size,
        json.dumps({c.code: round(c.percentile, 2) for c in score.contributions}),
        json.dumps(score.missing),
        score.risk.atr.value if score.risk and score.risk.atr.available else None,
        (
            score.risk.stop_price.value
            if score.risk and score.risk.stop_price.available
            else None
        ),
    )


def _to_entry(row) -> JournalEntry:
    return JournalEntry(
        session_date=date.fromisoformat(row["session_date"]),
        symbol=row["symbol"],
        score=row["score"],
        eligible=bool(row["eligible"]),
        method=row["method"],
        cohort_size=row["cohort_size"],
        contributions=json.loads(row["contributions"]),
        regime_verdict=row["regime_verdict"],
    )
