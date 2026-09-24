"""Tests for composite scoring, eligibility and risk sizing."""

from datetime import date, timedelta

import pytest

from algorix.cross_sectional import compute_universe_momentum
from algorix.models import Bar, DeliveryRecord
from algorix.scoring import (
    DEFAULT_STOP_ATR_MULTIPLE,
    MIN_CONTRIBUTORS,
    build_risk_profile,
    check_eligibility,
    score_universe,
)
from algorix.series import PriceSeries

AS_OF = date(2026, 9, 18)


def series_from(
    closes: list[float], volume: int = 100_000, instrument_id: int = 1
) -> PriceSeries:
    start = date(2020, 1, 1)
    bars = tuple(
        Bar(
            session_date=start + timedelta(days=i),
            open=c,
            high=c + 1,
            low=c - 1,
            close=c,
            volume=volume,
        )
        for i, c in enumerate(closes)
    )
    return PriceSeries(instrument_id=instrument_id, as_of=bars[-1].session_date, bars=bars)


def ramp(rate: float, length: int = 400, base: float = 100.0) -> list[float]:
    return [base * ((1 + rate) ** i) for i in range(length)]


def universe_of(rates: dict[str, float], volume: int = 100_000):
    return {
        sym: series_from(ramp(rate), volume=volume, instrument_id=i)
        for i, (sym, rate) in enumerate(rates.items(), start=1)
    }


def delivery_for(symbols, pct: float = 50.0, sessions: int = 25):
    start = date(2026, 8, 1)
    return {
        sym: [
            DeliveryRecord(
                session_date=start + timedelta(days=i),
                traded_quantity=1000,
                delivered_quantity=int(10 * pct),
            )
            for i in range(sessions)
        ]
        for sym in symbols
    }


def scored(rates: dict[str, float], **kwargs):
    series = universe_of(rates, volume=kwargs.pop("volume", 100_000))
    momentum = compute_universe_momentum(series, AS_OF)
    return score_universe(
        series,
        momentum,
        AS_OF,
        delivery_by_symbol=delivery_for(series),
        **kwargs,
    )


STANDARD = {f"S{i}": i / 10000.0 for i in range(12)}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_strongest_stock_scores_highest():
    universe = scored(STANDARD)

    top = universe.top(1)[0]

    assert top.symbol == "S11"
    assert top.score.value > 60


def test_pullback_inversion_penalises_the_hottest_stock():
    """A5 is scored inversely -- a stock still ripping is a poor entry.

    This is what stops the model degenerating into "buy whatever rose most":
    the strongest recent performer takes the worst pullback percentile even
    while topping momentum, which is the "strong over months, weak over days"
    combination INDICATORS.md A5 is designed to find.
    """
    universe = scored(STANDARD)

    hottest = universe.scores["S11"]
    by_code = {c.code: c.percentile for c in hottest.contributions}

    assert by_code["A1"] == pytest.approx(100.0)  # best momentum
    assert by_code["A5"] == pytest.approx(0.0)  # worst pullback entry
    # The drag is real: a perfect-momentum stock still scores well short of 100.
    assert hottest.score.value < 80


def test_weakest_stock_scores_lowest():
    universe = scored(STANDARD)

    ranked = universe.ranked()

    assert ranked[-1].symbol == "S0"
    assert ranked[-1].score.value < 40


def test_scores_sit_on_a_0_100_scale():
    universe = scored(STANDARD)

    for score in universe.ranked():
        assert 0.0 <= score.score.value <= 100.0


def test_score_records_its_method_and_cohort():
    """Principle 0.2a -- differently-derived scores must never be conflated."""
    universe = scored(STANDARD)

    top = universe.top(1)[0]

    assert top.method == "cross-sectional:NIFTY50"
    assert top.cohort_size == len(STANDARD)


# ---------------------------------------------------------------------------
# Sector-neutral scoring
# ---------------------------------------------------------------------------

#: Rates shaped like the real Sep 2026 finding: an entire sector (IT) is
#: the market's five weakest names, with nothing below it, while every
#: other sector is too small (<=2 members) to get its own baseline.
SECTOR_PILEUP_RATES = {
    "TCS": 0.00001, "INFY": 0.00002, "WIPRO": 0.00003,
    "HCLTECH": 0.00004, "TECHM": 0.00005,
    "RELIANCE": 0.00010, "ONGC": 0.00020, "ITC": 0.00030,
    "TITAN": 0.00040, "MARUTI": 0.00050,
    "AXISBANK": 0.00060, "SBIN": 0.00070,
}
SECTOR_PILEUP_MAP = {
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT", "TECHM": "IT",
    "RELIANCE": "Energy", "ONGC": "Energy", "ITC": "FMCG",
    "TITAN": "Consumer", "MARUTI": "Auto",
    "AXISBANK": "Financials", "SBIN": "Financials",
}


def test_sector_neutral_method_is_labelled():
    """Principle 0.2a/5 -- a sector-neutral score is a different claim from
    a plain cross-sectional one and must say so."""
    universe = scored(SECTOR_PILEUP_RATES, industry_by_symbol=SECTOR_PILEUP_MAP)

    top = universe.top(1)[0]

    assert top.method == "cross-sectional:NIFTY50:sector-neutral"


def test_without_industry_data_score_is_unaffected():
    """No sector data supplied must reproduce the exact pre-existing
    behaviour -- every caller that does not pass industry_by_symbol keeps
    working exactly as before this feature existed."""
    plain = scored(SECTOR_PILEUP_RATES)
    explicit_none = scored(SECTOR_PILEUP_RATES, industry_by_symbol=None)

    for symbol in SECTOR_PILEUP_RATES:
        assert plain.scores[symbol].score.value == pytest.approx(
            explicit_none.scores[symbol].score.value
        )
        assert plain.scores[symbol].method == "cross-sectional:NIFTY50"


def test_sector_pileup_no_longer_buries_the_sector():
    """The concrete failure this was built for: TECHM is IT's strongest
    name, but with the whole sector sitting at the bottom of the universe
    and nothing below it, a plain cross-sectional score cannot tell "IT's
    best" from "the market's worst." Sector-neutral scoring must be able to.
    """
    plain = scored(SECTOR_PILEUP_RATES)
    neutral = scored(SECTOR_PILEUP_RATES, industry_by_symbol=SECTOR_PILEUP_MAP)

    non_it = ["RELIANCE", "ONGC", "ITC", "TITAN", "MARUTI", "AXISBANK", "SBIN"]

    # Plain: TECHM (IT's best) still loses to every single non-IT name --
    # the sector's shared weakness swamps its own relative strength.
    assert plain.scores["TECHM"].score.value < min(
        plain.scores[s].score.value for s in non_it
    )

    # Sector-neutral: TECHM now beats at least one non-IT name.
    assert neutral.scores["TECHM"].score.value > min(
        neutral.scores[s].score.value for s in non_it
    )


def test_sector_neutral_preserves_true_raw_value_for_display():
    """Demeaning must only change what feeds the ranking -- the Contribution
    breakdown shown to the user must still show the real, human-readable
    number (e.g. actual % of 52-week high), not a sector-adjusted delta
    that would be meaningless on a dashboard.
    """
    plain = scored(SECTOR_PILEUP_RATES)
    neutral = scored(SECTOR_PILEUP_RATES, industry_by_symbol=SECTOR_PILEUP_MAP)

    def raw_of(universe, code):
        contributions = universe.scores["TECHM"].contributions
        return next(c.raw.value for c in contributions if c.code == code)

    for code in ("A1", "A2", "A3", "A4"):
        assert raw_of(plain, code) == pytest.approx(raw_of(neutral, code)), (
            f"{code}'s displayed raw value must not change when the "
            "ranking underneath it is sector-demeaned"
        )
    # The percentile *is* expected to change -- that's the entire point.
    plain_a1 = next(
        c.percentile for c in plain.scores["TECHM"].contributions if c.code == "A1"
    )
    neutral_a1 = next(
        c.percentile for c in neutral.scores["TECHM"].contributions if c.code == "A1"
    )
    assert neutral_a1 != plain_a1


def test_sector_neutral_leaves_pullback_and_delivery_untouched():
    """A5 (short-term reversal) and A6 (delivery) are different effects
    from the momentum/trend family and did not show the sector-pileup
    pattern -- only A1-A4 should move when sector data is supplied."""
    plain = scored(SECTOR_PILEUP_RATES)
    neutral = scored(SECTOR_PILEUP_RATES, industry_by_symbol=SECTOR_PILEUP_MAP)

    def percentile_of(universe, symbol, code):
        return next(
            c.percentile for c in universe.scores[symbol].contributions
            if c.code == code
        )

    for symbol in SECTOR_PILEUP_RATES:
        assert percentile_of(plain, symbol, "A6") == pytest.approx(
            percentile_of(neutral, symbol, "A6")
        )


def test_contributions_are_recorded_for_explainability():
    universe = scored(STANDARD)

    top = universe.top(1)[0]
    codes = {c.code for c in top.contributions}

    assert {"A1", "A2", "A3", "A4", "A5", "A6"} <= codes


def test_explain_names_strong_signals():
    universe = scored(STANDARD)

    explanation = universe.top(1)[0].explain()

    assert "S11" in explanation
    assert "strong on" in explanation


def test_explain_reports_ineligibility():
    universe = scored(STANDARD, volume=1)

    explanation = next(iter(universe.scores.values())).explain()

    assert "INELIGIBLE" in explanation


def test_delivery_absence_is_recorded_as_missing():
    series = universe_of(STANDARD)
    momentum = compute_universe_momentum(series, AS_OF)

    universe = score_universe(series, momentum, AS_OF, delivery_by_symbol={})

    top = universe.top(1)[0]
    assert "delivery" in top.missing
    assert top.is_complete is False
    # Still scored: the remaining five signals clear the minimum.
    assert top.score.available


def test_too_few_signals_yields_no_score():
    series = {"LONE": series_from(ramp(0.001, length=30))}
    momentum = compute_universe_momentum(series, AS_OF)

    universe = score_universe(series, momentum, AS_OF)

    score = universe.scores["LONE"]
    assert score.score.available is False
    assert f"needs {MIN_CONTRIBUTORS}" in score.score.reason


def test_unscored_stocks_are_listed_separately():
    series = universe_of(STANDARD)
    series["NEWLY_LISTED"] = series_from(ramp(0.001, length=30), instrument_id=99)
    momentum = compute_universe_momentum(series, AS_OF)

    universe = score_universe(series, momentum, AS_OF)

    assert "NEWLY_LISTED" in [s.symbol for s in universe.unscored]
    assert "NEWLY_LISTED" not in [s.symbol for s in universe.top(20)]


# ---------------------------------------------------------------------------
# Eligibility -- not the same as a low score
# ---------------------------------------------------------------------------


def test_liquid_stock_is_eligible():
    assert check_eligibility(series_from(ramp(0.001), volume=100_000)) == []


def test_illiquid_stock_is_ineligible():
    reasons = check_eligibility(series_from(ramp(0.001), volume=1))

    assert reasons
    assert "illiquid" in reasons[0]


def test_ineligible_stock_is_not_scored_low_but_excluded():
    """Ineligible and 'scored badly' are different claims."""
    universe = scored(STANDARD, volume=1)

    assert universe.ranked() == []
    assert len(universe.ineligible) == len(STANDARD)
    for score in universe.ineligible:
        assert score.score.available is False
        assert "illiquid" in score.score.reason


def test_ineligible_can_still_be_listed_explicitly():
    universe = scored(STANDARD, volume=1)

    assert universe.ranked(include_ineligible=True) == []


def test_short_history_cannot_be_assessed_for_liquidity():
    reasons = check_eligibility(series_from([100.0] * 5))

    assert "insufficient history" in reasons[0]


# ---------------------------------------------------------------------------
# Risk profile -- never part of the score
# ---------------------------------------------------------------------------


def test_risk_profile_places_a_stop_below_price():
    profile = build_risk_profile(series_from(ramp(0.001)))

    assert profile.atr.available
    assert profile.stop_price.available
    assert profile.stop_price.value < ramp(0.001)[-1]


def test_stop_distance_reflects_atr_multiple():
    series = series_from([100.0] * 30)
    profile = build_risk_profile(series)

    expected = 100.0 - profile.atr.value * DEFAULT_STOP_ATR_MULTIPLE
    assert profile.stop_price.value == pytest.approx(expected)


def test_position_size_scales_inversely_with_volatility():
    """A more volatile stock must be sized smaller for the same risk."""
    calm = build_risk_profile(series_from([100.0] * 30))
    wild_closes = [100.0 + (10 if i % 2 else 0) for i in range(30)]
    wild = build_risk_profile(series_from(wild_closes))

    calm_size = calm.position_size(1_000_000).value
    wild_size = wild.position_size(1_000_000).value

    assert wild_size < calm_size


def test_position_size_respects_risk_budget():
    profile = build_risk_profile(series_from([100.0] * 30))

    size = profile.position_size(1_000_000, risk_per_trade=0.01).value
    loss_at_stop = size * profile.atr.value * DEFAULT_STOP_ATR_MULTIPLE

    assert loss_at_stop == pytest.approx(10_000.0)


def test_position_size_refuses_without_atr():
    profile = build_risk_profile(series_from([100.0] * 5))

    result = profile.position_size(1_000_000)

    assert result.available is False
    assert "cannot size safely" in result.reason


def test_position_size_rejects_non_positive_equity():
    profile = build_risk_profile(series_from([100.0] * 30))

    assert profile.position_size(0).available is False


def test_risk_is_not_a_score_contribution():
    """CLAUDE.md invariant: risk inputs never enter the score."""
    universe = scored(STANDARD)

    top = universe.top(1)[0]
    codes = {c.code for c in top.contributions}

    assert "C1" not in codes
    assert "C2" not in codes
    assert top.risk is not None  # present, but separate


# ---------------------------------------------------------------------------
# A8 -- Post-Earnings Announcement Drift
# ---------------------------------------------------------------------------


def test_a8_is_missing_without_earnings_data():
    """The default, backward-compatible case: no earnings_by_symbol means
    A8 contributes nothing, exactly as before it existed."""
    universe = scored(STANDARD)

    for score in universe.ranked():
        by_code = {c.code for c in score.contributions}
        assert "A8" not in by_code
        assert "earnings drift" in score.missing


def earnings_for(surprise_by_symbol, days_ago=5):
    """EarningsSurpriseRecord fixtures for several symbols at once --
    A8's ranking cohort needs MIN_COHORT (10) members to be usable, so a
    single-stock earnings dict cannot exercise the full scoring pipeline."""
    from algorix.models import EarningsSurpriseRecord

    return {
        symbol: [
            EarningsSurpriseRecord(
                symbol=symbol, report_date=AS_OF - timedelta(days=days_ago),
                eps_estimate=10.0, eps_actual=10.0 * (1 + surprise / 100),
                surprise_pct=surprise,
            )
        ]
        for symbol, surprise in surprise_by_symbol.items()
    }


#: 10 members (S2-S11) -- clears MIN_COHORT. S0/S1 deliberately excluded,
#: to test the "no recent earnings" case within the same scored universe.
A8_COHORT = {f"S{i}": float(i * 5) for i in range(2, 12)}


def test_a8_contributes_for_a_stock_with_a_recent_surprise():
    universe = scored(STANDARD, earnings_by_symbol=earnings_for(A8_COHORT))

    scored_stock = universe.scores["S11"]
    by_code = {c.code: c for c in scored_stock.contributions}
    assert "A8" in by_code
    assert by_code["A8"].raw.value == pytest.approx(55.0)
    assert "earnings drift" not in scored_stock.missing


def test_a8_absent_for_stocks_with_no_recent_earnings_even_when_others_have_it():
    universe = scored(STANDARD, earnings_by_symbol=earnings_for(A8_COHORT))

    s0 = universe.scores["S0"]
    assert "A8" not in {c.code for c in s0.contributions}
    assert "earnings drift" in s0.missing


def test_a8_stale_earnings_do_not_contribute():
    universe = scored(
        STANDARD, earnings_by_symbol=earnings_for(A8_COHORT, days_ago=200)
    )

    assert "A8" not in {c.code for c in universe.scores["S11"].contributions}


def test_a8_is_not_sector_demeaned():
    """A8 is a company-idiosyncratic effect (see scoring.py's
    SECTOR_DEMEANED_CONTRIBUTORS comment) -- its percentile must be
    identical with or without sector data supplied."""
    earnings = earnings_for(A8_COHORT)
    industry = {f"S{i}": "TestSector" for i in range(12)}

    plain = scored(STANDARD, earnings_by_symbol=earnings)
    neutral = scored(STANDARD, earnings_by_symbol=earnings, industry_by_symbol=industry)

    plain_a8 = next(
        c.percentile for c in plain.scores["S11"].contributions if c.code == "A8"
    )
    neutral_a8 = next(
        c.percentile for c in neutral.scores["S11"].contributions if c.code == "A8"
    )
    assert plain_a8 == pytest.approx(neutral_a8)


def test_a8_negative_surprise_scores_low():
    cohort = dict(A8_COHORT)
    cohort["S2"] = -30.0  # a clear miss, well below the rest of the cohort

    universe = scored(STANDARD, earnings_by_symbol=earnings_for(cohort))

    miss = next(c for c in universe.scores["S2"].contributions if c.code == "A8")
    beat = next(c for c in universe.scores["S11"].contributions if c.code == "A8")
    assert beat.percentile > miss.percentile
