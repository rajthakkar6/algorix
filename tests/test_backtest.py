"""Tests for historical replay and score evaluation."""

from datetime import date, timedelta

import pytest

from algorix.backtest import (
    MIN_SESSION_COHORT,
    ScoredOutcome,
    evaluate,
    replay_session,
    run_backtest,
    spearman,
)
from algorix.calendar import TradingCalendar
from algorix.models import Bar, DeliveryRecord, Exchange
from algorix.series import PriceSeries
from algorix.storage import BarRepository, Database, InstrumentRepository
from algorix.universe import NIFTY_50, ConstituencyRepository, ConstituentRecord

END = date(2026, 9, 18)


@pytest.fixture
def cal():
    return TradingCalendar()


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "test.db")
    database.migrate()
    return database


def series_from(closes, sessions, instrument_id=1):
    bars = tuple(
        Bar(session_date=d, open=c, high=c + 1, low=c - 1, close=c, volume=200_000)
        for d, c in zip(sessions, closes)
    )
    return PriceSeries(instrument_id=instrument_id, as_of=sessions[-1], bars=bars)


# --- spearman -------------------------------------------------------------


def test_perfect_positive_correlation():
    assert spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == pytest.approx(1.0)


def test_perfect_negative_correlation():
    assert spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) == pytest.approx(-1.0)


def test_monotonic_but_nonlinear_is_still_perfect():
    """Rank correlation, not linear -- that is the point."""
    assert spearman([1, 2, 3, 4, 5], [1, 4, 9, 16, 25]) == pytest.approx(1.0)


def test_uncorrelated_is_near_zero():
    result = spearman([1, 2, 3, 4, 5, 6], [3, 1, 4, 1, 5, 9])

    assert abs(result) < 0.8


def test_all_tied_is_undefined_not_zero():
    """Zero correlation and 'cannot be computed' are different claims."""
    assert spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None


def test_too_few_points_is_undefined():
    assert spearman([1, 2], [3, 4]) is None


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        spearman([1, 2, 3], [1, 2])


# --- evaluation -----------------------------------------------------------


def outcomes_where(score_to_return, session=END, horizon=5):
    return [
        ScoredOutcome(session, f"S{i}", score, {horizon: ret})
        for i, (score, ret) in enumerate(score_to_return)
    ]


def test_perfect_signal_has_high_ic():
    pairs = [(float(i), float(i)) for i in range(20)]

    metrics = evaluate(outcomes_where(pairs), 5)

    assert metrics.mean_ic == pytest.approx(1.0)


def test_inverted_signal_has_negative_ic():
    pairs = [(float(i), float(-i)) for i in range(20)]

    metrics = evaluate(outcomes_where(pairs), 5)

    assert metrics.mean_ic == pytest.approx(-1.0)


def test_quintile_spread_is_positive_for_a_good_signal():
    pairs = [(float(i), float(i)) for i in range(20)]

    metrics = evaluate(outcomes_where(pairs), 5)

    assert metrics.quintile_spread > 0
    assert metrics.top_quintile_return > metrics.bottom_quintile_return


def test_quintile_spread_is_negative_for_an_inverted_signal():
    pairs = [(float(i), float(-i)) for i in range(20)]

    metrics = evaluate(outcomes_where(pairs), 5)

    assert metrics.quintile_spread < 0


def test_hit_rate_is_one_for_a_perfect_signal():
    pairs = [(float(i), float(i)) for i in range(20)]

    metrics = evaluate(outcomes_where(pairs), 5)

    assert metrics.hit_rate == pytest.approx(1.0)


def test_thin_sessions_are_excluded():
    """Metrics from a handful of names are noise."""
    pairs = [(float(i), float(i)) for i in range(MIN_SESSION_COHORT - 1)]

    metrics = evaluate(outcomes_where(pairs), 5)

    assert metrics.sessions == 0
    assert metrics.observations == 0


def test_missing_horizon_yields_no_metrics():
    pairs = [(float(i), float(i)) for i in range(20)]

    metrics = evaluate(outcomes_where(pairs, horizon=5), 20)

    assert metrics.observations == 0
    assert "no observations" in metrics.describe()


def test_ic_is_averaged_across_sessions():
    good = outcomes_where([(float(i), float(i)) for i in range(20)], date(2026, 9, 17))
    bad = outcomes_where([(float(i), float(-i)) for i in range(20)], END)

    metrics = evaluate(good + bad, 5)

    assert metrics.sessions == 2
    assert metrics.mean_ic == pytest.approx(0.0, abs=1e-9)


# --- forward returns ------------------------------------------------------


def test_forward_return_looks_ahead(cal):
    sessions = cal.sessions_in_range(date(2026, 9, 1), END)
    closes = [100.0 + i for i in range(len(sessions))]
    series = series_from(closes, sessions)

    result = series.forward_return(sessions[0], 3)

    assert result == pytest.approx(3.0, abs=0.1)


def test_forward_return_is_none_without_a_future(cal):
    sessions = cal.sessions_in_range(date(2026, 9, 1), END)
    series = series_from([100.0] * len(sessions), sessions)

    assert series.forward_return(sessions[-1], 5) is None


def test_forward_return_of_unknown_date_is_none(cal):
    sessions = cal.sessions_in_range(date(2026, 9, 1), END)
    series = series_from([100.0] * len(sessions), sessions)

    assert series.forward_return(date(2020, 1, 1), 1) is None


# --- truncation: the no-look-ahead guarantee ------------------------------


def test_truncate_drops_later_bars(cal):
    sessions = cal.sessions_in_range(date(2026, 9, 1), END)
    series = series_from([100.0] * len(sessions), sessions)

    truncated = series.truncate_to(date(2026, 9, 10))

    assert max(truncated.sessions) <= date(2026, 9, 10)
    assert truncated.as_of == date(2026, 9, 10)


def test_truncate_matches_a_fresh_load(db, cal):
    """In-memory slicing must equal what reloading would produce."""
    from algorix.series import load_series

    instrument_id = InstrumentRepository(db).upsert(
        __import__("algorix.models", fromlist=["Instrument"]).Instrument(
            symbol="TEST", exchange=Exchange.NSE,
            instrument_type=__import__(
                "algorix.models", fromlist=["InstrumentType"]
            ).InstrumentType.EQUITY,
        )
    )
    sessions = cal.sessions_in_range(date(2026, 6, 1), END)
    BarRepository(db).upsert_many(
        instrument_id,
        [Bar(session_date=d, open=100 + i, high=102 + i, low=99 + i,
             close=100 + i, volume=1000)
         for i, d in enumerate(sessions)],
        source="test",
    )

    full = load_series(db, instrument_id, END, 400, cal)
    reloaded = load_series(db, instrument_id, date(2026, 8, 3), 400, cal)

    assert full.truncate_to(date(2026, 8, 3)).sessions == reloaded.sessions


# --- end-to-end replay ----------------------------------------------------


def seed(db, cal, count=12, sessions_back=420):
    repo = InstrumentRepository(db)
    bars = BarRepository(db)
    symbols = [f"S{i}" for i in range(count)]
    ConstituencyRepository(db).sync(
        NIFTY_50,
        [ConstituentRecord(symbol=s, name=s) for s in symbols],
        date(2024, 1, 1),
    )
    sessions = cal.sessions_in_range(cal.trading_days_ago(END, sessions_back), END)
    for i, symbol in enumerate(symbols):
        instrument = repo.get(symbol, Exchange.NSE)
        rate = i / 10000.0
        bars.upsert_many(
            instrument.id,
            [Bar(session_date=d, open=p, high=p + 1, low=p - 1, close=p,
                 volume=200_000)
             for d, p in zip(sessions, [100.0 * ((1 + rate) ** j)
                                        for j in range(len(sessions))])],
            source="test",
        )
    return symbols, sessions


def test_backtest_scores_historical_sessions(db, cal):
    seed(db, cal)

    result = run_backtest(db, cal.trading_days_ago(END, 30), END, cal)

    assert result.sessions_scored > 0
    assert result.outcomes


def test_backtest_detects_a_predictive_signal(db, cal):
    """Steadily-compounding names should rank well and keep outperforming."""
    seed(db, cal)

    result = run_backtest(db, cal.trading_days_ago(END, 40), END, cal)

    assert result.metrics[5].mean_ic is not None
    assert result.metrics[5].mean_ic > 0.5


def test_backtest_reports_survivorship_caveat(db, cal):
    """Constituency recorded only from 2024 -- earlier ranges must warn."""
    repo = InstrumentRepository(db)
    symbols, _ = seed(db, cal)
    result = run_backtest(db, cal.trading_days_ago(END, 20), END, cal)

    # Membership was recorded before the range here, so no bias warning.
    assert not any("SURVIVORSHIP" in c for c in result.caveats)


def test_backtest_warns_without_recorded_membership(db, cal):
    symbols, _ = seed(db, cal)
    # Ask about a range predating the recorded membership window.
    result = run_backtest(db, date(2023, 1, 5), date(2023, 2, 5), cal)

    assert any("SURVIVORSHIP" in c for c in result.caveats)


def test_backtest_warns_about_missing_delivery(db, cal):
    seed(db, cal)

    result = run_backtest(db, cal.trading_days_ago(END, 20), END, cal)

    assert any("delivery" in c.lower() for c in result.caveats)


def test_backtest_always_flags_adjusted_prices(db, cal):
    seed(db, cal)

    result = run_backtest(db, cal.trading_days_ago(END, 20), END, cal)

    assert any("adjusted" in c for c in result.caveats)


def test_recent_sessions_have_no_future(db, cal):
    seed(db, cal)

    result = run_backtest(db, cal.trading_days_ago(END, 10), END, cal,
                          horizons=(20,))

    # The last 20 sessions cannot have 20-session forward returns.
    assert result.metrics[20].observations < len(result.outcomes) + 1


def test_backtest_on_empty_database(db, cal):
    result = run_backtest(db, cal.trading_days_ago(END, 10), END, cal)

    assert result.sessions_scored == 0
    assert result.caveats


def test_reversed_range_is_rejected(db, cal):
    with pytest.raises(ValueError, match="is after end"):
        run_backtest(db, END, cal.trading_days_ago(END, 10), cal)


def test_summary_surfaces_caveats(db, cal):
    seed(db, cal)

    summary = run_backtest(db, cal.trading_days_ago(END, 20), END, cal).summary()

    assert "CAVEATS" in summary
