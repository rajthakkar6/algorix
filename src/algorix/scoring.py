"""Composite scoring (INDICATORS.md Buckets A and C, design principles 0.2-0.3).

Turns indicators into a 0-100 score with a breakdown that explains it. The
score is the primary, backtestable number in this system -- CLAUDE.md
invariant 1 forbids anything, including an LLM, from silently overriding it.

Three structural rules, all from INDICATORS.md:

**Equal weights.** Contributors are averaged, not weighted by hand.
Out-of-sample, equal weighting across uncorrelated signal families reliably
beats hand-tuned weights, and there is no journal data yet to justify
anything else (open decision 3). Weights become tunable only once realized
outcomes exist to tune against.

**Risk inputs are not contributors.** ATR and realized volatility size the
position and place the stop. They are not directional and never enter the
score (principle 0.3).

**Eligibility is not a low score.** A stock that fails a liquidity or
tradeability gate is reported *ineligible*, not ranked badly -- those are
different claims, and conflating them buries a real opportunity under a
technicality or vice versa (principle 0.2b).

Every score records the method that produced it. A cross-sectional score
against 50 peers and one derived some other way are not comparable, and
principle 0.2a forbids presenting them as if they were.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from algorix.calendar import TradingCalendar
from algorix.cross_sectional import UniverseMomentum, rank_percentile, sector_demean
from algorix.indicators import (
    atr_percent,
    average_true_range,
    delivery_trend,
    donchian_position,
    pct_of_52_week_high,
    pead_signal,
    relative_volume,
    short_term_return,
    trend_state,
)
from algorix.models import DeliveryRecord, EarningsSurpriseRecord
from algorix.series import IndicatorValue, PriceSeries

#: Contributors required before a score is emitted. Below this the score
#: would rest on too few signals to mean much.
MIN_CONTRIBUTORS = 4

#: Minimum median daily turnover (Rs crore) for an instrument to be
#: tradeable. Thin names produce signals you cannot actually fill.
MIN_TURNOVER_CRORE = 1.0

#: Risk-per-trade used to size positions, as a fraction of account equity.
DEFAULT_RISK_PER_TRADE = 0.01

#: Stop distance in ATR multiples.
DEFAULT_STOP_ATR_MULTIPLE = 2.0


@dataclass(frozen=True)
class Contribution:
    """One signal's input to a score."""

    code: str
    label: str
    percentile: float
    raw: IndicatorValue
    note: str


@dataclass(frozen=True)
class RiskProfile:
    """Sizing and stop guidance (Bucket C). Never part of the score."""

    atr: IndicatorValue
    atr_pct: IndicatorValue
    stop_price: IndicatorValue
    stop_distance_pct: IndicatorValue

    def position_size(
        self,
        account_equity: float,
        risk_per_trade: float = DEFAULT_RISK_PER_TRADE,
    ) -> IndicatorValue:
        """Shares to buy so a stop-out costs `risk_per_trade` of equity."""
        if account_equity <= 0:
            return IndicatorValue.unavailable("account equity must be positive")
        if not self.atr.available or self.atr.value <= 0:
            return IndicatorValue.unavailable(
                "position size: ATR unavailable, cannot size safely"
            )

        risk_amount = account_equity * risk_per_trade
        per_share_risk = self.atr.value * DEFAULT_STOP_ATR_MULTIPLE
        return IndicatorValue.of(risk_amount / per_share_risk)


@dataclass(frozen=True)
class StockScore:
    """A scored instrument, with everything needed to explain the number."""

    symbol: str
    as_of: date
    score: IndicatorValue
    contributions: list[Contribution] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    eligible: bool = True
    ineligible_reasons: list[str] = field(default_factory=list)
    risk: RiskProfile | None = None
    #: How the score was derived -- see principle 0.2a.
    method: str = "cross-sectional"
    cohort_size: int = 0

    @property
    def is_complete(self) -> bool:
        return not self.missing

    def explain(self) -> str:
        """Plain-language account of why the score is what it is."""
        if not self.eligible:
            return f"{self.symbol}: INELIGIBLE -- " + "; ".join(
                self.ineligible_reasons
            )
        if not self.score.available:
            return f"{self.symbol}: unscored -- {self.score.reason}"

        strong = [c for c in self.contributions if c.percentile >= 70]
        weak = [c for c in self.contributions if c.percentile <= 30]

        parts = [f"{self.symbol}: {self.score.value:.0f}/100"]
        if strong:
            parts.append("strong on " + ", ".join(c.note for c in strong))
        if weak:
            parts.append("weak on " + ", ".join(c.note for c in weak))
        if self.missing:
            parts.append(f"missing {', '.join(self.missing)}")
        return " | ".join(parts)


@dataclass(frozen=True)
class ScoredUniverse:
    """Scores for a whole universe on one session."""

    as_of: date
    scores: dict[str, StockScore] = field(default_factory=dict)

    def ranked(self, include_ineligible: bool = False) -> list[StockScore]:
        candidates = [
            s
            for s in self.scores.values()
            if s.score.available and (include_ineligible or s.eligible)
        ]
        return sorted(candidates, key=lambda s: s.score.value, reverse=True)

    def top(self, n: int = 10) -> list[StockScore]:
        return self.ranked()[:n]

    @property
    def ineligible(self) -> list[StockScore]:
        return [s for s in self.scores.values() if not s.eligible]

    @property
    def unscored(self) -> list[StockScore]:
        return [
            s for s in self.scores.values() if s.eligible and not s.score.available
        ]


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


def check_eligibility(
    series: PriceSeries, min_turnover_crore: float = MIN_TURNOVER_CRORE
) -> list[str]:
    """Reasons an instrument is not tradeable, empty if it is.

    Currently checks liquidity. Circuit-lock status, F&O ban membership and
    promoter pledge (INDICATORS.md §0.2b) need NSE sources this build does
    not yet ingest -- they matter far more once the universe expands past the
    Nifty 50, where every name is large and liquid.
    """
    reasons: list[str] = []

    if len(series) < 20:
        reasons.append("insufficient history to assess liquidity")
        return reasons

    recent = series.window(20)
    turnovers = sorted(b.close * b.volume / 1e7 for b in recent)  # Rs crore
    median_turnover = turnovers[len(turnovers) // 2]

    if median_turnover < min_turnover_crore:
        reasons.append(
            f"illiquid: median turnover Rs {median_turnover:.2f} cr "
            f"is below Rs {min_turnover_crore:.2f} cr"
        )

    return reasons


# ---------------------------------------------------------------------------
# Risk profile
# ---------------------------------------------------------------------------


def build_risk_profile(
    series: PriceSeries,
    calendar: TradingCalendar | None = None,
    stop_multiple: float = DEFAULT_STOP_ATR_MULTIPLE,
) -> RiskProfile:
    atr = average_true_range(series, calendar=calendar)
    atr_p = atr_percent(series, calendar=calendar)

    stop_price = IndicatorValue.unavailable("stop: ATR unavailable")
    stop_pct = IndicatorValue.unavailable("stop: ATR unavailable")

    if atr.available and series.bars:
        close = series.bars[-1].close
        stop = close - (atr.value * stop_multiple)
        if stop > 0:
            stop_price = IndicatorValue.of(stop)
            stop_pct = IndicatorValue.of(((close - stop) / close) * 100.0)
        else:
            stop_price = IndicatorValue.unavailable(
                "stop: ATR exceeds price, instrument too volatile to size"
            )

    return RiskProfile(
        atr=atr, atr_pct=atr_p, stop_price=stop_price, stop_distance_pct=stop_pct
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _trend_to_score(series: PriceSeries, calendar) -> IndicatorValue:
    """Collapse trend state into a 0-100 reading for ranking.

    Four independent conditions, equally weighted, so a stock in a fully
    aligned uptrend scores 100 and one below everything scores 0.
    """
    state = trend_state(series, calendar)
    if state is None:
        return IndicatorValue.unavailable("trend: insufficient history")

    flags = (
        state.above_fast,
        state.above_slow,
        state.fast_above_slow,
        state.fast_rising,
    )
    return IndicatorValue.of(sum(flags) / len(flags) * 100.0)


#: Contributors that all measure some form of "price has risen recently"
#: (INDICATORS.md's own diagnosis: A2-A4 are restatements of A1, not
#: independent evidence). A concentrated sector move shows up in all four
#: together, which is what put every Nifty 50 IT name in the bottom five
#: ranks in the Sep 2026 walk-forward test. A5 (short-term reversal), A6
#: (delivery accumulation) and A8 (post-earnings drift) are different,
#: idiosyncratic-to-the-company effects that did not show the same
#: sector-wide pileup, so they are left on the raw universe-wide comparison.
SECTOR_DEMEANED_CONTRIBUTORS = ("A1", "A2", "A3", "A4")


def score_universe(
    series_by_symbol: dict[str, PriceSeries],
    momentum: UniverseMomentum,
    as_of: date,
    delivery_by_symbol: dict[str, list[DeliveryRecord]] | None = None,
    calendar: TradingCalendar | None = None,
    universe_label: str = "NIFTY50",
    industry_by_symbol: dict[str, str | None] | None = None,
    earnings_by_symbol: dict[str, list[EarningsSurpriseRecord]] | None = None,
) -> ScoredUniverse:
    """Score every instrument against its peers.

    Contributors (equal weight, each ranked cross-sectionally):
      A1 momentum composite, A2 52-week high proximity, A3 trend state,
      A4 Donchian position (confirmed by A7 relative volume),
      A5 short-term pullback (inverted), A6 delivery trend,
      A8 post-earnings drift (recent earnings surprise, when one exists).

    When `industry_by_symbol` is supplied, A1-A4 are demeaned against their
    sector's average before ranking -- see `sector_demean`. Without it, this
    behaves exactly as it did before that existed: a plain cross-sectional
    score. The two are not the same claim, so the method label records
    which one ran (design principle 0.2a/5) and the journal's
    `SCORING_VERSION` was bumped when this was introduced, so a session
    scored the old way and one scored the new way are never silently
    compared as though comparable.

    `earnings_by_symbol` (a stock's `EarningsSurpriseRecord` history, caller-
    bounded to `end=as_of` the same way `delivery_by_symbol` is) drives A8.
    Unlike A1-A6, most of the universe will not have a recent enough report
    to be scored on it at any given moment -- that is A8's correct,
    event-driven behaviour, not a data gap; see `pead_signal`.
    """
    delivery_by_symbol = delivery_by_symbol or {}
    earnings_by_symbol = earnings_by_symbol or {}
    sector_neutral = bool(industry_by_symbol)

    raw: dict[str, dict[str, IndicatorValue]] = {
        "A1": {},
        "A2": {},
        "A3": {},
        "A4": {},
        "A5": {},
        "A6": {},
        "A8": {},
    }

    for symbol, series in series_by_symbol.items():
        profile = momentum.profiles.get(symbol)
        raw["A1"][symbol] = (
            profile.composite
            if profile is not None
            else IndicatorValue.unavailable("momentum: not in universe")
        )
        raw["A2"][symbol] = pct_of_52_week_high(series, calendar)
        raw["A3"][symbol] = _trend_to_score(series, calendar)

        # A4 confirmed by A7: volume corroborates a breakout but is not a
        # standalone signal, so it scales the Donchian reading rather than
        # occupying its own slot.
        donchian = donchian_position(series, calendar=calendar)
        volume = relative_volume(series, calendar=calendar)
        if donchian.available and volume.available:
            # Cap the multiplier so one freak volume day cannot dominate.
            confirmation = min(max(volume.value, 0.5), 2.0)
            raw["A4"][symbol] = IndicatorValue.of(
                min(100.0, donchian.value * (0.75 + 0.25 * confirmation))
            )
        else:
            raw["A4"][symbol] = donchian

        # A5 inverted: within a strong universe, recent *weakness* is the
        # better entry -- short-term reversal runs against momentum.
        pullback = short_term_return(series, calendar=calendar)
        raw["A5"][symbol] = (
            IndicatorValue.of(-pullback.value) if pullback.available else pullback
        )

        raw["A6"][symbol] = delivery_trend(delivery_by_symbol.get(symbol, []))

        raw["A8"][symbol] = pead_signal(
            earnings_by_symbol.get(symbol, []), as_of, calendar
        )

    # `raw` keeps the true, human-readable values (52w-high %, trend flag
    # average, ...) for the Contribution breakdown shown to the user.
    # Demeaning only changes what feeds the percentile ranking below --
    # never what gets displayed as "why" a score is what it is.
    ranking_input = {
        code: (
            sector_demean(values, industry_by_symbol)
            if code in SECTOR_DEMEANED_CONTRIBUTORS
            else values
        )
        for code, values in raw.items()
    }
    rankings = {
        code: rank_percentile(values) for code, values in ranking_input.items()
    }

    labels = {
        "A1": ("momentum", "relative momentum"),
        "A2": ("52w high", "proximity to 52-week high"),
        "A3": ("trend", "trend alignment"),
        "A4": ("breakout", "breakout position"),
        "A5": ("pullback", "short-term pullback"),
        "A6": ("delivery", "delivery accumulation"),
        "A8": ("earnings drift", "post-earnings surprise drift"),
    }

    scores: dict[str, StockScore] = {}
    for symbol, series in series_by_symbol.items():
        ineligible = check_eligibility(series)
        risk = build_risk_profile(series, calendar)

        contributions: list[Contribution] = []
        missing: list[str] = []

        for code, (short_label, note) in labels.items():
            percentile = rankings[code].percentile_of(symbol)
            if percentile.available:
                contributions.append(
                    Contribution(
                        code=code,
                        label=short_label,
                        percentile=percentile.value,
                        raw=raw[code][symbol],
                        note=note,
                    )
                )
            else:
                missing.append(short_label)

        if ineligible:
            score = IndicatorValue.unavailable(
                "not scored: " + "; ".join(ineligible)
            )
        elif len(contributions) < MIN_CONTRIBUTORS:
            score = IndicatorValue.unavailable(
                f"only {len(contributions)} of {len(labels)} signals available "
                f"(needs {MIN_CONTRIBUTORS}); missing {', '.join(missing)}"
            )
        else:
            score = IndicatorValue.of(
                sum(c.percentile for c in contributions) / len(contributions)
            )

        cohort = max((rankings[c].cohort_size for c in rankings), default=0)
        scores[symbol] = StockScore(
            symbol=symbol,
            as_of=as_of,
            score=score,
            contributions=contributions,
            missing=missing,
            eligible=not ineligible,
            ineligible_reasons=ineligible,
            risk=risk,
            method=(
                f"cross-sectional:{universe_label}:sector-neutral"
                if sector_neutral
                else f"cross-sectional:{universe_label}"
            ),
            cohort_size=cohort,
        )

    return ScoredUniverse(as_of=as_of, scores=scores)
