"""Tests for the trade journal."""

from datetime import date, timedelta

import pytest

from algorix.cross_sectional import compute_universe_momentum
from algorix.journal import SCORING_VERSION, Journal
from algorix.models import Bar, DeliveryRecord
from algorix.regime import FlowRecord, RegimeVerdict, assess_regime
from algorix.scoring import score_universe
from algorix.series import PriceSeries
from algorix.storage import Database

AS_OF = date(2026, 9, 18)


def series_from(closes, instrument_id=1, volume=100_000):
    start = date(2020, 1, 1)
    bars = tuple(
        Bar(session_date=start + timedelta(days=i), open=c, high=c + 1,
            low=c - 1, close=c, volume=volume)
        for i, c in enumerate(closes)
    )
    return PriceSeries(instrument_id=instrument_id, as_of=bars[-1].session_date, bars=bars)


def build_universe(as_of=AS_OF, length=400):
    series = {
        f"S{i}": series_from(
            [100.0 * ((1 + i / 10000.0) ** j) for j in range(length)],
            instrument_id=i + 1,
        )
        for i in range(12)
    }
    delivery = {
        sym: [
            DeliveryRecord(session_date=date(2026, 8, 1) + timedelta(days=d),
                           traded_quantity=1000, delivered_quantity=500)
            for d in range(25)
        ]
        for sym in series
    }
    momentum = compute_universe_momentum(series, as_of)
    return score_universe(series, momentum, as_of, delivery_by_symbol=delivery)


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "test.db")
    database.migrate()
    return database


@pytest.fixture
def journal(db):
    return Journal(db)


def test_records_a_scan(journal):
    run_id = journal.record(build_universe())

    assert isinstance(run_id, int)
    assert journal.recorded_sessions() == [AS_OF]


def test_stored_scores_are_readable(journal):
    journal.record(build_universe())

    entries = journal.scores_for(AS_OF)

    assert len(entries) == 12
    assert entries[0].score is not None


def test_stores_the_contribution_breakdown(journal):
    """'Why did this score 82?' must be answerable months later."""
    journal.record(build_universe())

    entries = journal.scores_for(AS_OF)
    top = entries[0]

    assert set(top.contributions) >= {"A1", "A2", "A3", "A4", "A5"}
    assert all(isinstance(v, float) for v in top.contributions.values())


def test_stores_method_and_cohort(journal):
    journal.record(build_universe())

    top = journal.scores_for(AS_OF)[0]

    assert top.method.startswith("cross-sectional")
    assert top.cohort_size == 12


def test_stores_the_regime_verdict(journal):
    universe = build_universe()
    regime = assess_regime(
        AS_OF,
        {s: series_from([100.0] * 79 + [200.0], i) for i, s in enumerate(universe.scores)},
        flows={"FII/FPI": FlowRecord(AS_OF, "FII/FPI", 1.0, 0.0, 500.0)},
    )

    journal.record(universe, regime=regime)

    assert journal.scores_for(AS_OF)[0].regime_verdict is not None


def test_rerecording_replaces_rather_than_duplicates(journal):
    journal.record(build_universe())
    journal.record(build_universe())

    assert len(journal.scores_for(AS_OF)) == 12
    assert journal.recorded_sessions() == [AS_OF]


def test_history_for_a_symbol_spans_sessions(journal):
    journal.record(build_universe(as_of=date(2026, 9, 17)))
    journal.record(build_universe(as_of=AS_OF))

    history = journal.history_for("S11")

    assert len(history) == 2
    assert history[0].session_date == AS_OF  # newest first


def test_history_is_limited(journal):
    for day in range(1, 6):
        journal.record(build_universe(as_of=date(2026, 9, 10 + day)))

    assert len(journal.history_for("S11", limit=3)) == 3


def test_unscored_entries_are_stored_with_their_reason(journal):
    universe = build_universe()
    journal.record(universe)

    entries = {e.symbol: e for e in journal.scores_for(AS_OF)}
    assert len(entries) == 12


def test_empty_session_returns_nothing(journal):
    assert journal.scores_for(date(2020, 1, 1)) == []


def test_history_of_unknown_symbol_is_empty(journal):
    journal.record(build_universe())

    assert journal.history_for("NOSUCH") == []


def test_scoring_version_is_recorded(db, journal):
    journal.record(build_universe())

    with db.connect() as conn:
        row = conn.execute("SELECT scoring_version FROM scan_runs").fetchone()

    assert row["scoring_version"] == SCORING_VERSION
