"""Tests for cross-sectional ranking and multi-window momentum."""

from datetime import date, timedelta

import pytest

from algorix.cross_sectional import (
    MIN_COHORT,
    Ranking,
    compute_universe_momentum,
    rank_percentile,
)
from algorix.indicators import MOMENTUM_12M, MOMENTUM_SKIP, total_return
from algorix.models import Bar
from algorix.series import IndicatorValue, PriceSeries

AS_OF = date(2026, 9, 18)


def series_from(closes: list[float], instrument_id: int = 1) -> PriceSeries:
    start = date(2020, 1, 1)
    bars = tuple(
        Bar(
            session_date=start + timedelta(days=i),
            open=c,
            high=c + 1,
            low=c - 1,
            close=c,
            volume=1000,
        )
        for i, c in enumerate(closes)
    )
    return PriceSeries(
        instrument_id=instrument_id, as_of=bars[-1].session_date, bars=bars
    )


def flat_then(total: int, final_move: float, base: float = 100.0) -> list[float]:
    """A flat path that steps to `base * (1 + final_move)` at the end."""
    return [base] * (total - 1) + [base * (1 + final_move)]


def cohort(rates: dict[str, float], length: int = 400) -> dict[str, PriceSeries]:
    """Build a universe where each symbol compounds at its own rate.

    A steady ramp rather than a single final step: the 12-month window skips
    the most recent month, so a move parked in the final bars would be
    invisible to it and every symbol would tie.
    """
    return {
        symbol: series_from(
            [100.0 * ((1.0 + rate) ** i) for i in range(length)],
            instrument_id=i,
        )
        for i, (symbol, rate) in enumerate(rates.items(), start=1)
    }


# ---------------------------------------------------------------------------
# total_return with skip
# ---------------------------------------------------------------------------


def test_total_return_measures_the_window():
    result = total_return(series_from([100.0] * 21 + [110.0]), sessions=21)

    assert result.value == pytest.approx(10.0)


def test_skip_excludes_the_recent_window():
    """The 12M-skip-1M convention must not see the final month's move."""
    closes = [100.0] * (MOMENTUM_12M + 1) + [500.0] * MOMENTUM_SKIP

    skipped = total_return(
        series_from(closes), MOMENTUM_12M, skip=MOMENTUM_SKIP
    )
    unskipped = total_return(series_from(closes), MOMENTUM_12M)

    # The spike lands entirely inside the skipped month.
    assert skipped.value == pytest.approx(0.0)
    assert unskipped.value > 100.0


def test_skip_requires_extra_history():
    result = total_return(series_from([100.0] * 25), sessions=21, skip=21)

    assert result.available is False


def test_negative_skip_is_rejected():
    with pytest.raises(ValueError, match="skip cannot be negative"):
        total_return(series_from([100.0] * 30), 10, skip=-1)


def test_non_positive_sessions_is_rejected():
    with pytest.raises(ValueError, match="sessions must be positive"):
        total_return(series_from([100.0] * 30), 0)


# ---------------------------------------------------------------------------
# Percentile ranking
# ---------------------------------------------------------------------------


def test_best_performer_ranks_100():
    values = {
        f"S{i}": IndicatorValue.of(float(i)) for i in range(MIN_COHORT + 2)
    }

    ranking = rank_percentile(values)

    best = max(values, key=lambda s: values[s].value)
    assert ranking.percentile_of(best).value == pytest.approx(100.0)


def test_worst_performer_ranks_zero():
    values = {
        f"S{i}": IndicatorValue.of(float(i)) for i in range(MIN_COHORT + 2)
    }

    ranking = rank_percentile(values)

    assert ranking.percentile_of("S0").value == pytest.approx(0.0)


def test_ties_share_a_mid_rank():
    """Identical values must not be ordered arbitrarily."""
    values = {f"S{i}": IndicatorValue.of(50.0) for i in range(MIN_COHORT + 2)}

    ranking = rank_percentile(values)
    percentiles = {r.percentile for r in ranking.ranked.values()}

    assert len(percentiles) == 1


def test_middle_value_ranks_mid_scale():
    values = {f"S{i}": IndicatorValue.of(float(i)) for i in range(11)}

    ranking = rank_percentile(values)

    assert ranking.percentile_of("S5").value == pytest.approx(50.0)


def test_unavailable_values_are_excluded_not_ranked_low():
    """Unknown momentum is not bad momentum."""
    values = {f"S{i}": IndicatorValue.of(float(i)) for i in range(MIN_COHORT + 1)}
    values["NOHISTORY"] = IndicatorValue.unavailable("needs 252 sessions, has 30")

    ranking = rank_percentile(values)

    assert "NOHISTORY" not in ranking.ranked
    assert "NOHISTORY" in ranking.excluded
    assert ranking.percentile_of("NOHISTORY").available is False
    # Its absence must not have dragged the real cohort's ranks.
    assert ranking.percentile_of("S0").value == pytest.approx(0.0)


def test_exclusion_reason_is_preserved():
    values = {"A": IndicatorValue.unavailable("needs 252 sessions, has 30")}

    ranking = rank_percentile(values)

    assert "252" in ranking.excluded["A"]


def test_small_cohort_is_not_usable():
    """Percentiles among a handful of survivors are noise."""
    values = {f"S{i}": IndicatorValue.of(float(i)) for i in range(3)}

    ranking = rank_percentile(values)

    assert ranking.is_usable is False
    assert ranking.percentile_of("S0").available is False
    assert "too small" in ranking.percentile_of("S0").reason


def test_cohort_size_is_recorded():
    values = {f"S{i}": IndicatorValue.of(float(i)) for i in range(MIN_COHORT + 2)}

    ranking = rank_percentile(values)

    assert ranking.cohort_size == MIN_COHORT + 2
    assert ranking.ranked["S0"].cohort_size == MIN_COHORT + 2


def test_empty_input_produces_empty_ranking():
    ranking = rank_percentile({})

    assert ranking.cohort_size == 0
    assert ranking.is_usable is False


def test_all_unavailable_produces_empty_ranking():
    values = {f"S{i}": IndicatorValue.unavailable("no data") for i in range(5)}

    ranking = rank_percentile(values)

    assert ranking.ranked == {}
    assert len(ranking.excluded) == 5


def test_top_returns_highest_first():
    values = {f"S{i}": IndicatorValue.of(float(i)) for i in range(MIN_COHORT + 2)}

    top = rank_percentile(values).top(3)

    assert [r.symbol for r in top] == ["S11", "S10", "S9"]


# ---------------------------------------------------------------------------
# Universe momentum
# ---------------------------------------------------------------------------


def test_universe_momentum_ranks_the_strongest_top():
    rates = {f"S{i}": i / 10000.0 for i in range(MIN_COHORT + 2)}
    universe = compute_universe_momentum(cohort(rates), AS_OF)

    top = universe.top(1)

    assert top[0].symbol == "S11"
    assert top[0].composite.value > 90.0


def test_universe_momentum_ranks_the_weakest_bottom():
    rates = {f"S{i}": i / 10000.0 for i in range(MIN_COHORT + 2)}
    universe = compute_universe_momentum(cohort(rates), AS_OF)

    assert universe.profiles["S0"].composite.value < 10.0


def test_profile_carries_raw_returns_and_percentiles():
    rates = {f"S{i}": i / 10000.0 for i in range(MIN_COHORT + 2)}
    universe = compute_universe_momentum(cohort(rates), AS_OF)

    profile = universe.profiles["S5"]

    assert profile.return_1m.available
    assert profile.percentile_1m.available
    assert profile.is_complete is True


def test_short_history_gets_no_composite():
    """Averaging whichever windows happened to compute is not comparable."""
    series = cohort({f"S{i}": i / 10000.0 for i in range(MIN_COHORT + 2)})
    series["NEWLY_LISTED"] = series_from([100.0] * 30, instrument_id=99)

    universe = compute_universe_momentum(series, AS_OF)

    profile = universe.profiles["NEWLY_LISTED"]
    assert profile.composite.available is False
    assert "missing window" in profile.composite.reason
    assert profile.is_complete is False


def test_unrankable_instrument_does_not_distort_peers():
    """An instrument with no computable value must not shift anyone's rank.

    Note the boundary: a 30-session newcomer *does* have a real 1-month
    return and legitimately joins that cohort, shifting peers -- that is
    cross-sectional ranking working. Only an instrument that cannot produce a
    value for a window is excluded from it.
    """
    full = cohort({f"S{i}": i / 10000.0 for i in range(MIN_COHORT + 2)})
    with_stub = dict(full)
    # Too short for even the 1-month window.
    with_stub["JUST_LISTED"] = series_from([100.0] * 10, instrument_id=99)

    baseline = compute_universe_momentum(full, AS_OF)
    perturbed = compute_universe_momentum(with_stub, AS_OF)

    for symbol in full:
        assert perturbed.profiles[symbol].composite.value == pytest.approx(
            baseline.profiles[symbol].composite.value
        )
    assert perturbed.profiles["JUST_LISTED"].composite.available is False


def test_partial_history_joins_only_the_windows_it_can_fill():
    """A 30-session newcomer ranks in 1M but not 3M or 12M."""
    series = cohort({f"S{i}": i / 10000.0 for i in range(MIN_COHORT + 2)})
    series["NEWLY_LISTED"] = series_from([100.0] * 30, instrument_id=99)

    universe = compute_universe_momentum(series, AS_OF)

    assert universe.rankings["1m"].percentile_of("NEWLY_LISTED").available is True
    assert universe.rankings["3m"].percentile_of("NEWLY_LISTED").available is False
    assert universe.profiles["NEWLY_LISTED"].composite.available is False


def test_top_excludes_instruments_without_a_composite():
    series = cohort({f"S{i}": i / 10000.0 for i in range(MIN_COHORT + 2)})
    series["NEWLY_LISTED"] = series_from([100.0] * 30, instrument_id=99)

    universe = compute_universe_momentum(series, AS_OF)

    assert "NEWLY_LISTED" not in [p.symbol for p in universe.top(20)]


def test_tiny_universe_yields_no_composites():
    universe = compute_universe_momentum(cohort({"A": 0.0001, "B": 0.0002}), AS_OF)

    assert universe.profiles["A"].composite.available is False


def test_rankings_are_exposed_per_window():
    rates = {f"S{i}": i / 10000.0 for i in range(MIN_COHORT + 2)}
    universe = compute_universe_momentum(cohort(rates), AS_OF)

    assert set(universe.rankings) == {"1m", "3m", "12m"}
    assert universe.rankings["3m"].is_usable
