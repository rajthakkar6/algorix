"""Tests for cross-sectional ranking and multi-window momentum."""

from datetime import date, timedelta

import pytest

from algorix.cross_sectional import (
    MIN_COHORT,
    MIN_SECTOR_SIZE,
    Ranking,
    compute_universe_momentum,
    rank_percentile,
    sector_demean,
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


# ---------------------------------------------------------------------------
# sector_demean
# ---------------------------------------------------------------------------


def iv(x: float) -> IndicatorValue:
    return IndicatorValue.of(x)


def test_sector_demean_fixes_a_sector_pileup():
    """The concrete failure this exists for: an entire sector clustered at
    one end of the ranking, indistinguishable from a genuinely weak stock.

    Five IT names all around momentum 10 (the Sep 2026 pattern -- IT held
    the bottom five ranks together) against five other-sector names spread
    from -5 to 15. Before demeaning, every IT name loses to every spread-out
    peer. After, an IT name that is merely average *for IT* lands in the
    middle of the demeaned distribution instead of the bottom.
    """
    raw = {
        "TCS": iv(9.0), "INFY": iv(10.0), "WIPRO": iv(11.0),
        "HCLTECH": iv(10.5), "TECHM": iv(9.5),
        "RELIANCE": iv(-5.0), "ONGC": iv(0.0), "ITC": iv(5.0),
        "TITAN": iv(10.0), "MARUTI": iv(15.0),
    }
    sector = {
        "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT", "TECHM": "IT",
        "RELIANCE": "Energy", "ONGC": "Energy", "ITC": "FMCG",
        "TITAN": "Consumer", "MARUTI": "Auto",
    }

    demeaned = sector_demean(raw, sector)

    # IT's own mean (10.0) is now the IT names' zero point -- INFY, sitting
    # exactly on its sector's mean, is no longer indistinguishable from
    # RELIANCE at -5. It is now near the middle of the whole set, not the
    # bottom of it.
    it_values = sorted(demeaned[s].value for s in ("TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"))
    assert it_values == pytest.approx([-1.0, -0.5, 0.0, 0.5, 1.0])

    ranking = rank_percentile(demeaned)
    infy_percentile = ranking.percentile_of("INFY").value
    assert 30 < infy_percentile < 70, (
        f"INFY (sector-average IT) should land mid-table after demeaning, "
        f"got percentile {infy_percentile}"
    )


def test_sector_demean_preserves_real_intra_sector_spread():
    """Demeaning must not flatten genuine differences within a sector --
    only the sector's shared component should be removed."""
    raw = {"A": iv(20.0), "B": iv(10.0), "C": iv(0.0)}
    sector = {"A": "X", "B": "X", "C": "X"}

    demeaned = sector_demean(raw, sector)

    assert demeaned["A"].value > demeaned["B"].value > demeaned["C"].value
    assert demeaned["A"].value - demeaned["C"].value == pytest.approx(20.0)


def test_sector_demean_below_min_size_falls_back_to_universe_mean():
    """A sector with fewer than MIN_SECTOR_SIZE members has no meaningful
    average of its own -- but it must still be demeaned against *something*
    on the same scale as everyone else, or it silently stays on a different
    footing than sector-demeaned peers.

    This is not a hypothetical: leaving such a stock at its raw, un-demeaned
    value was the actual bug this test replaces. When a large sector's
    demeaned block sits at one extreme with nothing below it (IT's real
    Sep 2026 shape), an un-demeaned small-sector stock beside it does not
    land "unaffected" -- it ends up on the wrong side of a scale comparison
    that no longer means what it looks like it means.
    """
    assert MIN_SECTOR_SIZE >= 2  # the scenario below needs this to hold
    raw = {"LONE": iv(42.0), "A": iv(1.0), "B": iv(2.0)}
    sector = {"LONE": "Telecom", "A": "IT", "B": "IT"}

    demeaned = sector_demean(raw, sector, min_sector_size=3)

    # Universe mean of the three raw values is 15.0 -- LONE lands at
    # 42 - 15 = 27, not at its untouched raw 42.
    assert demeaned["LONE"].value == pytest.approx(27.0)
    # A and B's sector (IT) is also below the floor here, so they get the
    # same universe-mean baseline -- their relative order among themselves
    # is unchanged, which is the actual guarantee: rank preserved, scale
    # made comparable.
    assert demeaned["A"].value < demeaned["B"].value


def test_sector_demean_unknown_sector_uses_universe_mean():
    """A stock with no recorded sector is not evidence of anything, but it
    still needs a same-scale baseline rather than being left raw among
    values everyone else has had a mean subtracted from."""
    raw = {"UNCLASSIFIED": iv(7.0), "A": iv(1.0), "B": iv(2.0), "C": iv(3.0)}
    sector = {"A": "IT", "B": "IT", "C": "IT"}  # UNCLASSIFIED absent

    demeaned = sector_demean(raw, sector)

    # Universe mean of all four raw values is 3.25.
    assert demeaned["UNCLASSIFIED"].value == pytest.approx(7.0 - 3.25)


def test_sector_demean_unavailable_values_pass_through():
    """A stock that could not be scored at all stays unscored, not zeroed."""
    raw = {
        "MISSING": IndicatorValue.unavailable("insufficient history"),
        "A": iv(1.0), "B": iv(2.0), "C": iv(3.0),
    }
    sector = {"MISSING": "IT", "A": "IT", "B": "IT", "C": "IT"}

    demeaned = sector_demean(raw, sector)

    assert demeaned["MISSING"].available is False


def test_sector_demean_large_sector_at_an_extreme():
    """The actual failure shape from Sep 2026: one sector big enough to get
    its own mean (IT, 5 names) sits at the absolute bottom of the universe,
    with every other sector too small to qualify (2 or fewer members each,
    below MIN_SECTOR_SIZE) sitting entirely above it.

    IT's best name should end up comparable to -- not worse than -- at
    least one non-IT name after demeaning. Before the universe-mean
    fallback, small sectors stayed at their raw absolute values while IT
    alone got recentred near zero, which pushed IT even further below
    everyone else instead of closer to them.
    """
    raw = {
        "TCS": iv(1.0), "INFY": iv(2.0), "WIPRO": iv(3.0),
        "HCLTECH": iv(4.0), "TECHM": iv(5.0),          # IT: lowest 5, raw
        "RELIANCE": iv(10.0), "ONGC": iv(20.0),          # Energy: 2, below floor
        "ITC": iv(30.0),                                  # FMCG: 1
        "TITAN": iv(40.0),                                # Consumer: 1
        "AXISBANK": iv(60.0), "SBIN": iv(70.0),          # Financials: 2
    }
    sector = {
        "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT", "TECHM": "IT",
        "RELIANCE": "Energy", "ONGC": "Energy", "ITC": "FMCG", "TITAN": "Consumer",
        "AXISBANK": "Financials", "SBIN": "Financials",
    }

    demeaned = sector_demean(raw, sector, min_sector_size=3)

    non_it = ["RELIANCE", "ONGC", "ITC", "TITAN", "AXISBANK", "SBIN"]
    assert demeaned["TECHM"].value > min(demeaned[s].value for s in non_it), (
        "IT's strongest name must beat at least the weakest non-IT name "
        "after demeaning -- otherwise the whole sector is still buried"
    )
    # IT's internal order must survive -- demeaning removes the shared
    # component, not the genuine differences between IT names.
    assert demeaned["TECHM"].value > demeaned["TCS"].value


def test_sector_demean_with_no_sector_map_is_a_passthrough():
    """No sector data at all (None, or empty) must reproduce the old,
    plain cross-sectional behaviour exactly -- this is the backward-
    compatibility guarantee that lets every existing caller keep working."""
    raw = {"A": iv(1.0), "B": iv(2.0), "C": iv(3.0)}

    assert sector_demean(raw, None) == raw
    assert sector_demean(raw, {}) == raw
