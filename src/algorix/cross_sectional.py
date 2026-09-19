"""Cross-sectional ranking (INDICATORS.md A1 and design principle 0.2).

Momentum is a *relative* effect. A 6% three-month return means one thing when
the universe averaged -4% and something else entirely when it averaged +12%,
so raw returns are ranked into percentiles within the universe rather than
compared against fixed thresholds. Percentiles are also self-normalising
across volatility regimes, which absolute cutoffs are not.

Two rules protect the ranking:

**Instruments without a value are excluded, not ranked low.** A stock whose
history is too short has an *unknown* momentum, not a bad one. Ranking it at
zero would quietly assert the latter and push genuinely weak names up the
table.

**The cohort is recorded with the result.** A percentile computed against 50
peers and one computed against 8 survivors are not comparable, and
INDICATORS.md §0.2a forbids presenting them as though they were.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date

from algorix.calendar import TradingCalendar
from algorix.indicators import (
    MOMENTUM_1M,
    MOMENTUM_3M,
    MOMENTUM_12M,
    MOMENTUM_SKIP,
    total_return,
)
from algorix.series import DEFAULT_MIN_COMPLETENESS, IndicatorValue, PriceSeries

#: A ranking needs a real cohort. Percentiles among a handful of survivors are
#: noise dressed as precision.
MIN_COHORT = 10


@dataclass(frozen=True)
class RankedValue:
    """One instrument's place within a cohort."""

    symbol: str
    raw: float
    percentile: float
    cohort_size: int


@dataclass(frozen=True)
class Ranking:
    """Percentile ranks for one measure across one cohort."""

    ranked: dict[str, RankedValue] = field(default_factory=dict)
    #: Instruments excluded, with the reason their value was unavailable.
    excluded: dict[str, str] = field(default_factory=dict)

    @property
    def cohort_size(self) -> int:
        return len(self.ranked)

    @property
    def is_usable(self) -> bool:
        return self.cohort_size >= MIN_COHORT

    def percentile_of(self, symbol: str) -> IndicatorValue:
        entry = self.ranked.get(symbol)
        if entry is None:
            return IndicatorValue.unavailable(
                self.excluded.get(symbol, f"{symbol} not in cohort")
            )
        if not self.is_usable:
            return IndicatorValue.unavailable(
                f"cohort of {self.cohort_size} is too small to rank "
                f"(needs {MIN_COHORT})"
            )
        return IndicatorValue.of(entry.percentile)

    def top(self, n: int) -> list[RankedValue]:
        return sorted(
            self.ranked.values(), key=lambda r: r.percentile, reverse=True
        )[:n]


def rank_percentile(values: Mapping[str, IndicatorValue]) -> Ranking:
    """Rank available values into 0-100 percentiles.

    Ties share a mid-rank, so three identical values all receive the same
    percentile rather than being ordered arbitrarily by dictionary insertion.
    """
    usable: dict[str, float] = {}
    excluded: dict[str, str] = {}

    for symbol, value in values.items():
        if value.available:
            usable[symbol] = value.value
        else:
            excluded[symbol] = value.reason or "unavailable"

    cohort = len(usable)
    if cohort == 0:
        return Ranking(ranked={}, excluded=excluded)

    ranked: dict[str, RankedValue] = {}
    for symbol, raw in usable.items():
        below = sum(1 for other in usable.values() if other < raw)
        equal = sum(1 for other in usable.values() if other == raw)
        # Mid-rank: ties land in the middle of the band they jointly occupy.
        percentile = ((below + (equal - 1) / 2) / max(cohort - 1, 1)) * 100.0
        ranked[symbol] = RankedValue(
            symbol=symbol,
            raw=raw,
            percentile=max(0.0, min(100.0, percentile)),
            cohort_size=cohort,
        )

    return Ranking(ranked=ranked, excluded=excluded)


@dataclass(frozen=True)
class MomentumProfile:
    """Multi-window momentum for one instrument (INDICATORS.md A1)."""

    symbol: str
    return_1m: IndicatorValue
    return_3m: IndicatorValue
    return_12m_skip_1m: IndicatorValue
    percentile_1m: IndicatorValue
    percentile_3m: IndicatorValue
    percentile_12m: IndicatorValue
    composite: IndicatorValue

    @property
    def is_complete(self) -> bool:
        return all(
            v.available
            for v in (self.percentile_1m, self.percentile_3m, self.percentile_12m)
        )


@dataclass(frozen=True)
class UniverseMomentum:
    """Cross-sectional momentum for a whole universe on one session."""

    as_of: date
    profiles: dict[str, MomentumProfile] = field(default_factory=dict)
    rankings: dict[str, Ranking] = field(default_factory=dict)

    def top(self, n: int = 10) -> list[MomentumProfile]:
        scored = [p for p in self.profiles.values() if p.composite.available]
        return sorted(scored, key=lambda p: p.composite.value, reverse=True)[:n]


def compute_universe_momentum(
    series_by_symbol: Mapping[str, PriceSeries],
    as_of: date,
    calendar: TradingCalendar | None = None,
    min_completeness: float = DEFAULT_MIN_COMPLETENESS,
) -> UniverseMomentum:
    """Rank every instrument's momentum against its peers.

    The composite averages the three window percentiles. Equal weighting is
    deliberate: INDICATORS.md open decision 3 recommends it until journal data
    exists to justify anything else, because hand-tuned weights reliably
    overfit out of sample.

    An instrument missing any window gets no composite -- averaging over
    whichever windows happened to compute would make short-history names
    quietly incomparable to full-history ones.
    """
    windows = {
        "1m": (MOMENTUM_1M, 0),
        "3m": (MOMENTUM_3M, 0),
        "12m": (MOMENTUM_12M, MOMENTUM_SKIP),
    }

    raw: dict[str, dict[str, IndicatorValue]] = {}
    for label, (sessions, skip) in windows.items():
        raw[label] = {
            symbol: total_return(
                series, sessions, skip, calendar, min_completeness
            )
            for symbol, series in series_by_symbol.items()
        }

    rankings = {label: rank_percentile(values) for label, values in raw.items()}

    profiles: dict[str, MomentumProfile] = {}
    for symbol in series_by_symbol:
        percentiles = {
            label: rankings[label].percentile_of(symbol) for label in windows
        }

        if all(p.available for p in percentiles.values()):
            composite = IndicatorValue.of(
                sum(p.value for p in percentiles.values()) / len(percentiles)
            )
        else:
            missing = [
                label for label, p in percentiles.items() if not p.available
            ]
            composite = IndicatorValue.unavailable(
                f"momentum composite: missing window(s) {sorted(missing)}"
            )

        profiles[symbol] = MomentumProfile(
            symbol=symbol,
            return_1m=raw["1m"][symbol],
            return_3m=raw["3m"][symbol],
            return_12m_skip_1m=raw["12m"][symbol],
            percentile_1m=percentiles["1m"],
            percentile_3m=percentiles["3m"],
            percentile_12m=percentiles["12m"],
            composite=composite,
        )

    return UniverseMomentum(as_of=as_of, profiles=profiles, rankings=rankings)
