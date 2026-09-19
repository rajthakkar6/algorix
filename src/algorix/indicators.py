"""Single-instrument indicators (INDICATORS.md Buckets A and C).

Each function takes a PriceSeries and returns an IndicatorValue -- a number,
or an explicit reason there isn't one. Nothing here silently substitutes a
shorter window, a default, or a zero.

What is deliberately absent matters as much as what is here. RSI, MACD,
Stochastics, Bollinger Bands, ADX, OBV and the rest were rejected in
INDICATORS.md Bucket E for redundancy or for conflicting with momentum. Do
not add them back because they are familiar -- see CLAUDE.md invariant 6.

Bucket C indicators (ATR, realized volatility) are risk and sizing inputs.
They are NOT directional and must never be fed into the composite score.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from algorix.calendar import TradingCalendar
from algorix.models import DeliveryRecord
from algorix.series import DEFAULT_MIN_COMPLETENESS, IndicatorValue, PriceSeries

# -- Standard windows -------------------------------------------------------

#: Sessions in a trading year, used for the 52-week high (A2).
SESSIONS_PER_YEAR = 252

#: A3: 50 and 200 DMA. The 20 DMA was demoted in INDICATORS.md -- redundant
#: beside the 50 at a days-to-weeks horizon.
TREND_FAST = 50
TREND_SLOW = 200

#: A4: Donchian breakout windows.
DONCHIAN_SHORT = 20
DONCHIAN_LONG = 55

#: A5: short-term pullback horizon.
PULLBACK_SESSIONS = 5

#: A7: relative volume baseline.
VOLUME_BASELINE = 20

#: C1/C2: ATR window and the history used to rank volatility.
ATR_WINDOW = 14
VOL_PERCENTILE_WINDOW = 252


#: A1 momentum windows, in sessions. ~21 sessions to a month.
MOMENTUM_1M = 21
MOMENTUM_3M = 63
MOMENTUM_12M = 252
#: Sessions skipped at the near end of the 12-month window. Jegadeesh-Titman
#: convention: the most recent month is dominated by short-term reversal,
#: which runs *opposite* to momentum and contaminates the signal.
MOMENTUM_SKIP = 21

#: A6 delivery-trend windows, in published sessions.
DELIVERY_RECENT = 5
DELIVERY_BASELINE = 20


def _guard(
    series: PriceSeries,
    sessions: int,
    calendar: TradingCalendar | None,
    min_completeness: float,
) -> str | None:
    return series.require(sessions, calendar, min_completeness)


# ---------------------------------------------------------------------------
# A1 -- total return over a window (the per-instrument half of momentum)
# ---------------------------------------------------------------------------


def total_return(
    series: PriceSeries,
    sessions: int,
    skip: int = 0,
    calendar: TradingCalendar | None = None,
    min_completeness: float = DEFAULT_MIN_COMPLETENESS,
) -> IndicatorValue:
    """Percentage return over `sessions`, ending `skip` sessions ago.

    `skip` exists for the 12-month window (INDICATORS.md A1): skipping the
    most recent month removes short-term reversal, which works against
    momentum and otherwise pollutes the strongest signal in the model.

    This is only half of A1 -- momentum is a *cross-sectional* effect, so the
    raw return here is ranked against the universe in `cross_sectional.py`.
    """
    if sessions <= 0:
        raise ValueError(f"sessions must be positive, got {sessions}")
    if skip < 0:
        raise ValueError(f"skip cannot be negative, got {skip}")

    needed = sessions + skip + 1
    problem = _guard(series, needed, calendar, min_completeness)
    if problem:
        label = f"{sessions}-session return"
        if skip:
            label += f" (skip {skip})"
        return IndicatorValue.unavailable(f"{label}: {problem}")

    bars = series.bars
    end_index = len(bars) - 1 - skip
    start_index = end_index - sessions

    start_price = bars[start_index].close
    end_price = bars[end_index].close
    if start_price <= 0:
        return IndicatorValue.unavailable("return: non-positive base price")

    return IndicatorValue.of(((end_price / start_price) - 1.0) * 100.0)


# ---------------------------------------------------------------------------
# A6 -- delivery percentage trend
# ---------------------------------------------------------------------------


def delivery_trend(
    records: Sequence[DeliveryRecord],
    recent: int = DELIVERY_RECENT,
    baseline: int = DELIVERY_BASELINE,
) -> IndicatorValue:
    """Recent delivery % as a multiple of its own baseline (INDICATORS.md A6).

    Above 1.0 means delivery-based buying is picking up relative to this
    stock's own norm -- genuine accumulation rather than intraday churn.
    Measured against the stock's own history rather than a market-wide
    threshold, because normal delivery varies enormously by stock.

    Sessions where nothing traded contribute no percentage (0/0 is
    undefined) and are skipped rather than counted as zero.
    """
    if recent <= 0 or baseline <= 0:
        raise ValueError("windows must be positive")
    if recent > baseline:
        raise ValueError(
            f"recent window {recent} cannot exceed baseline {baseline}"
        )

    usable = [
        r.delivery_pct
        for r in sorted(records, key=lambda r: r.session_date)
        if r.delivery_pct is not None
    ]

    if len(usable) < baseline:
        return IndicatorValue.unavailable(
            f"delivery trend: needs {baseline} published sessions, "
            f"has {len(usable)}"
        )

    window = usable[-baseline:]
    baseline_mean = statistics.fmean(window)
    recent_mean = statistics.fmean(usable[-recent:])

    if baseline_mean <= 0:
        return IndicatorValue.unavailable(
            "delivery trend: baseline delivery is zero"
        )

    return IndicatorValue.of(recent_mean / baseline_mean)


def latest_delivery_pct(records: Sequence[DeliveryRecord]) -> IndicatorValue:
    """Most recent published delivery percentage."""
    usable = [
        r for r in sorted(records, key=lambda r: r.session_date)
        if r.delivery_pct is not None
    ]
    if not usable:
        return IndicatorValue.unavailable("delivery: no published sessions")
    return IndicatorValue.of(usable[-1].delivery_pct)


# ---------------------------------------------------------------------------
# A2 -- 52-week high proximity
# ---------------------------------------------------------------------------


def pct_of_52_week_high(
    series: PriceSeries,
    calendar: TradingCalendar | None = None,
    min_completeness: float = DEFAULT_MIN_COMPLETENESS,
) -> IndicatorValue:
    """Latest close as a percentage of the 52-week high (INDICATORS.md A2).

    A documented standalone anomaly, not merely a momentum restatement -- it
    often holds when return-based momentum is noisy.
    """
    problem = _guard(series, SESSIONS_PER_YEAR, calendar, min_completeness)
    if problem:
        return IndicatorValue.unavailable(f"52-week high: {problem}")

    window = series.window(SESSIONS_PER_YEAR)
    high = max(b.high for b in window)
    if high <= 0:
        return IndicatorValue.unavailable("52-week high: non-positive high")

    return IndicatorValue.of((window[-1].close / high) * 100.0)


# ---------------------------------------------------------------------------
# A3 -- trend state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrendState:
    """Where price sits relative to its moving averages (INDICATORS.md A3).

    Time-series momentum is a separate effect from cross-sectional momentum:
    a stock can rank well against peers while in its own downtrend. This also
    serves as the long-only veto.
    """

    above_fast: bool
    above_slow: bool
    fast_above_slow: bool
    fast_rising: bool
    pct_from_fast: float
    pct_from_slow: float

    @property
    def is_uptrend(self) -> bool:
        """Price above both averages, with the fast above the slow."""
        return self.above_fast and self.above_slow and self.fast_above_slow


def simple_moving_average(series: PriceSeries, window: int) -> float | None:
    if len(series) < window:
        return None
    return statistics.fmean(b.close for b in series.window(window))


def trend_state(
    series: PriceSeries,
    calendar: TradingCalendar | None = None,
    min_completeness: float = DEFAULT_MIN_COMPLETENESS,
) -> TrendState | None:
    """Trend state, or None when history is too short or too sparse."""
    if _guard(series, TREND_SLOW, calendar, min_completeness):
        return None

    fast = simple_moving_average(series, TREND_FAST)
    slow = simple_moving_average(series, TREND_SLOW)
    if fast is None or slow is None or fast <= 0 or slow <= 0:
        return None

    close = series.bars[-1].close

    # Slope measured against the fast average one week earlier.
    earlier = PriceSeries(
        instrument_id=series.instrument_id,
        as_of=series.as_of,
        bars=series.bars[:-5],
    )
    fast_prev = simple_moving_average(earlier, TREND_FAST)

    return TrendState(
        above_fast=close > fast,
        above_slow=close > slow,
        fast_above_slow=fast > slow,
        fast_rising=fast_prev is not None and fast > fast_prev,
        pct_from_fast=((close / fast) - 1.0) * 100.0,
        pct_from_slow=((close / slow) - 1.0) * 100.0,
    )


# ---------------------------------------------------------------------------
# A4 -- Donchian breakout
# ---------------------------------------------------------------------------


def donchian_position(
    series: PriceSeries,
    window: int = DONCHIAN_SHORT,
    calendar: TradingCalendar | None = None,
    min_completeness: float = DEFAULT_MIN_COMPLETENESS,
) -> IndicatorValue:
    """Where the close sits in its N-session range, 0-100 (INDICATORS.md A4).

    100 means a new N-session high. Chosen over subjective chart patterns
    because it is objective and backtestable.

    The window excludes the current bar, so "at a new high" means the close
    exceeded the prior range rather than trivially equalling its own high.
    """
    needed = window + 1
    problem = _guard(series, needed, calendar, min_completeness)
    if problem:
        return IndicatorValue.unavailable(f"Donchian({window}): {problem}")

    prior = series.bars[-needed:-1]
    high = max(b.high for b in prior)
    low = min(b.low for b in prior)
    close = series.bars[-1].close

    if high <= low:
        return IndicatorValue.unavailable(
            f"Donchian({window}): flat range, position undefined"
        )

    position = ((close - low) / (high - low)) * 100.0
    # A genuine breakout closes outside the prior range; clamp so the scale
    # stays 0-100 while still reporting the extreme.
    return IndicatorValue.of(max(0.0, min(100.0, position)))


def is_breakout(
    series: PriceSeries,
    window: int = DONCHIAN_SHORT,
    calendar: TradingCalendar | None = None,
) -> bool:
    """True when the close exceeds the prior `window` sessions' high."""
    needed = window + 1
    if _guard(series, needed, calendar, DEFAULT_MIN_COMPLETENESS):
        return False
    prior = series.bars[-needed:-1]
    return series.bars[-1].close > max(b.high for b in prior)


# ---------------------------------------------------------------------------
# A5 -- short-term pullback
# ---------------------------------------------------------------------------


def short_term_return(
    series: PriceSeries,
    sessions: int = PULLBACK_SESSIONS,
    calendar: TradingCalendar | None = None,
    min_completeness: float = DEFAULT_MIN_COMPLETENESS,
) -> IndicatorValue:
    """Percentage return over the last `sessions` (INDICATORS.md A5).

    Scored *inversely* by the composite: short-term reversal runs opposite to
    long-horizon momentum, so "strong over months, weak over days" is the
    combination worth buying. Returned here as a plain return; the sign
    convention belongs to the scorer, not the measurement.
    """
    needed = sessions + 1
    problem = _guard(series, needed, calendar, min_completeness)
    if problem:
        return IndicatorValue.unavailable(f"{sessions}-session return: {problem}")

    window = series.window(needed)
    start, end = window[0].close, window[-1].close
    if start <= 0:
        return IndicatorValue.unavailable(
            f"{sessions}-session return: non-positive base price"
        )

    return IndicatorValue.of(((end / start) - 1.0) * 100.0)


# ---------------------------------------------------------------------------
# A7 -- relative volume
# ---------------------------------------------------------------------------


def relative_volume(
    series: PriceSeries,
    baseline: int = VOLUME_BASELINE,
    calendar: TradingCalendar | None = None,
    min_completeness: float = DEFAULT_MIN_COMPLETENESS,
) -> IndicatorValue:
    """Latest volume as a multiple of its `baseline`-session average (A7).

    A confirmation modifier on a breakout, never a standalone score: volume
    accompanies breakdowns as readily as breakouts.
    """
    needed = baseline + 1
    problem = _guard(series, needed, calendar, min_completeness)
    if problem:
        return IndicatorValue.unavailable(f"relative volume: {problem}")

    window = series.window(needed)
    prior = window[:-1]
    average = statistics.fmean(b.volume for b in prior)

    if average <= 0:
        # An instrument that reports no volume at all (FX) has no meaningful
        # relative volume -- that is absence, not a ratio of zero.
        return IndicatorValue.unavailable(
            "relative volume: baseline volume is zero"
        )

    return IndicatorValue.of(window[-1].volume / average)


# ---------------------------------------------------------------------------
# C1 -- Average True Range  (RISK INPUT -- never a score contributor)
# ---------------------------------------------------------------------------


def average_true_range(
    series: PriceSeries,
    window: int = ATR_WINDOW,
    calendar: TradingCalendar | None = None,
    min_completeness: float = DEFAULT_MIN_COMPLETENESS,
) -> IndicatorValue:
    """ATR in price units (INDICATORS.md C1).

    Used for stop distance, volatility-normalised position sizing, and making
    breakout thresholds comparable across instruments. Not directional --
    CLAUDE.md invariant: risk inputs never enter the score.
    """
    needed = window + 1
    problem = _guard(series, needed, calendar, min_completeness)
    if problem:
        return IndicatorValue.unavailable(f"ATR({window}): {problem}")

    window_bars = series.window(needed)
    true_ranges = []
    for previous, current in zip(window_bars, window_bars[1:]):
        true_ranges.append(
            max(
                current.high - current.low,
                abs(current.high - previous.close),
                abs(current.low - previous.close),
            )
        )

    return IndicatorValue.of(statistics.fmean(true_ranges))


def atr_percent(
    series: PriceSeries,
    window: int = ATR_WINDOW,
    calendar: TradingCalendar | None = None,
    min_completeness: float = DEFAULT_MIN_COMPLETENESS,
) -> IndicatorValue:
    """ATR as a percentage of the latest close.

    The comparable form: a 50-rupee ATR means something different on a
    100-rupee stock than on a 5,000-rupee one.
    """
    atr = average_true_range(series, window, calendar, min_completeness)
    if not atr.available:
        return atr

    close = series.bars[-1].close
    if close <= 0:
        return IndicatorValue.unavailable("ATR%: non-positive close")
    return IndicatorValue.of((atr.value / close) * 100.0)


# ---------------------------------------------------------------------------
# C2 -- realized volatility percentile  (RISK INPUT)
# ---------------------------------------------------------------------------


def realized_volatility_percentile(
    series: PriceSeries,
    window: int = ATR_WINDOW,
    history: int = VOL_PERCENTILE_WINDOW,
    calendar: TradingCalendar | None = None,
    min_completeness: float = DEFAULT_MIN_COMPLETENESS,
) -> IndicatorValue:
    """Where current ATR% sits within its own past year, 0-100 (C2).

    Captures the genuinely useful part of Bollinger Bands -- volatility
    contraction and expansion -- without the redundant envelope.
    """
    needed = history + window + 1
    problem = _guard(series, needed, calendar, min_completeness)
    if problem:
        return IndicatorValue.unavailable(f"volatility percentile: {problem}")

    readings: list[float] = []
    for end in range(len(series.bars) - history, len(series.bars) + 1):
        slice_ = PriceSeries(
            instrument_id=series.instrument_id,
            as_of=series.as_of,
            bars=series.bars[:end],
        )
        value = atr_percent(slice_, window, calendar=None)
        if value.available:
            readings.append(value.value)

    if len(readings) < 2:
        return IndicatorValue.unavailable(
            "volatility percentile: not enough ATR readings"
        )

    current = readings[-1]
    below = sum(1 for r in readings[:-1] if r < current)
    return IndicatorValue.of((below / (len(readings) - 1)) * 100.0)
