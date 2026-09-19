"""Tests for single-instrument indicators.

Series are constructed from explicit price paths so each indicator's value is
verifiable by hand rather than asserted against whatever the code produced.
"""

from datetime import date, timedelta

import pytest

from algorix.calendar import TradingCalendar
from algorix.indicators import (
    ATR_WINDOW,
    DONCHIAN_SHORT,
    SESSIONS_PER_YEAR,
    TREND_SLOW,
    atr_percent,
    average_true_range,
    donchian_position,
    is_breakout,
    pct_of_52_week_high,
    realized_volatility_percentile,
    relative_volume,
    short_term_return,
    simple_moving_average,
    trend_state,
)
from algorix.models import Bar
from algorix.series import PriceSeries


def series_from(
    closes: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    volumes: list[int] | None = None,
) -> PriceSeries:
    """Build a series of consecutive weekday-ish dates from explicit prices.

    Dates are synthetic and only ordered; calendar-aware tests pass a real
    calendar explicitly.
    """
    n = len(closes)
    highs = highs or [c + 1 for c in closes]
    lows = lows or [c - 1 for c in closes]
    volumes = volumes or [1000] * n
    start = date(2020, 1, 1)

    bars = tuple(
        Bar(
            session_date=start + timedelta(days=i),
            open=closes[i],
            high=highs[i],
            low=lows[i],
            close=closes[i],
            volume=volumes[i],
        )
        for i in range(n)
    )
    return PriceSeries(instrument_id=1, as_of=bars[-1].session_date, bars=bars)


# ---------------------------------------------------------------------------
# A2 -- 52-week high proximity
# ---------------------------------------------------------------------------


def test_at_the_high_is_100_pct():
    closes = [100.0] * (SESSIONS_PER_YEAR - 1) + [110.0]
    highs = [101.0] * (SESSIONS_PER_YEAR - 1) + [110.0]

    result = pct_of_52_week_high(series_from(closes, highs=highs))

    assert result.available
    assert result.value == pytest.approx(100.0)


def test_half_way_down_from_the_high():
    closes = [200.0] + [100.0] * (SESSIONS_PER_YEAR - 1)
    highs = [200.0] + [100.0] * (SESSIONS_PER_YEAR - 1)

    result = pct_of_52_week_high(series_from(closes, highs=highs))

    assert result.value == pytest.approx(50.0)


def test_52_week_high_needs_a_full_year():
    result = pct_of_52_week_high(series_from([100.0] * 100))

    assert result.available is False
    assert "needs 252 sessions" in result.reason


# ---------------------------------------------------------------------------
# A3 -- trend state
# ---------------------------------------------------------------------------


def test_moving_average_is_the_mean_of_the_window():
    series = series_from([10.0, 20.0, 30.0, 40.0])

    assert simple_moving_average(series, 4) == pytest.approx(25.0)
    assert simple_moving_average(series, 2) == pytest.approx(35.0)


def test_moving_average_is_none_without_enough_history():
    assert simple_moving_average(series_from([10.0, 20.0]), 5) is None


def test_rising_series_is_an_uptrend():
    closes = [float(i) for i in range(100, TREND_SLOW + 150)]

    state = trend_state(series_from(closes))

    assert state is not None
    assert state.is_uptrend is True
    assert state.above_fast and state.above_slow and state.fast_above_slow
    assert state.fast_rising is True


def test_falling_series_is_not_an_uptrend():
    closes = [float(i) for i in range(TREND_SLOW + 150, 99, -1)]

    state = trend_state(series_from(closes))

    assert state is not None
    assert state.is_uptrend is False
    assert state.above_slow is False
    assert state.fast_rising is False


def test_trend_reports_distance_from_averages():
    closes = [100.0] * (TREND_SLOW + 50)
    closes[-1] = 110.0

    state = trend_state(series_from(closes))

    assert state is not None
    assert state.pct_from_slow > 0


def test_trend_needs_200_sessions():
    assert trend_state(series_from([100.0] * 199)) is None


def test_trend_available_at_exactly_200_sessions():
    closes = [float(i) for i in range(100, TREND_SLOW + 100)]

    assert trend_state(series_from(closes)) is not None


# ---------------------------------------------------------------------------
# A4 -- Donchian breakout
# ---------------------------------------------------------------------------


def test_new_high_is_a_breakout():
    closes = [100.0] * DONCHIAN_SHORT + [120.0]
    highs = [101.0] * DONCHIAN_SHORT + [120.0]

    series = series_from(closes, highs=highs)

    assert is_breakout(series) is True
    assert donchian_position(series).value == pytest.approx(100.0)


def test_close_inside_the_range_is_not_a_breakout():
    closes = [100.0] * DONCHIAN_SHORT + [100.0]
    highs = [110.0] * DONCHIAN_SHORT + [101.0]
    lows = [90.0] * DONCHIAN_SHORT + [99.0]

    series = series_from(closes, highs=highs, lows=lows)

    assert is_breakout(series) is False
    assert 0.0 < donchian_position(series).value < 100.0


def test_at_the_bottom_of_the_range():
    closes = [100.0] * DONCHIAN_SHORT + [90.0]
    highs = [110.0] * DONCHIAN_SHORT + [91.0]
    lows = [90.0] * DONCHIAN_SHORT + [89.0]

    result = donchian_position(series_from(closes, highs=highs, lows=lows))

    assert result.value == pytest.approx(0.0)


def test_breakout_excludes_the_current_bar():
    """Otherwise a bar trivially equals its own high and always 'breaks out'."""
    closes = [100.0] * (DONCHIAN_SHORT + 1)
    highs = [100.0] * DONCHIAN_SHORT + [200.0]

    assert is_breakout(series_from(closes, highs=highs)) is False


def test_flat_range_has_no_defined_position():
    closes = [100.0] * (DONCHIAN_SHORT + 1)
    result = donchian_position(
        series_from(closes, highs=[100.0] * (DONCHIAN_SHORT + 1),
                    lows=[100.0] * (DONCHIAN_SHORT + 1))
    )

    assert result.available is False
    assert "flat range" in result.reason


def test_donchian_needs_history():
    result = donchian_position(series_from([100.0] * 5))

    assert result.available is False


def test_breakout_is_false_without_history():
    assert is_breakout(series_from([100.0] * 3)) is False


# ---------------------------------------------------------------------------
# A5 -- short-term return
# ---------------------------------------------------------------------------


def test_short_term_gain():
    result = short_term_return(series_from([100.0, 101, 102, 103, 104, 110.0]))

    assert result.value == pytest.approx(10.0)


def test_short_term_loss():
    result = short_term_return(series_from([100.0, 99, 98, 97, 96, 90.0]))

    assert result.value == pytest.approx(-10.0)


def test_flat_short_term_return_is_zero_and_available():
    result = short_term_return(series_from([100.0] * 6))

    assert result.available is True
    assert result.value == pytest.approx(0.0)


def test_short_term_return_needs_enough_sessions():
    result = short_term_return(series_from([100.0, 101.0]))

    assert result.available is False


# ---------------------------------------------------------------------------
# A7 -- relative volume
# ---------------------------------------------------------------------------


def test_double_average_volume():
    volumes = [1000] * 20 + [2000]
    result = relative_volume(series_from([100.0] * 21, volumes=volumes))

    assert result.value == pytest.approx(2.0)


def test_average_volume_is_one():
    volumes = [1000] * 21
    result = relative_volume(series_from([100.0] * 21, volumes=volumes))

    assert result.value == pytest.approx(1.0)


def test_zero_baseline_volume_is_unavailable_not_zero():
    """FX reports no volume -- that is absence, not a ratio."""
    volumes = [0] * 21
    result = relative_volume(series_from([100.0] * 21, volumes=volumes))

    assert result.available is False
    assert "baseline volume is zero" in result.reason


def test_relative_volume_needs_a_baseline():
    result = relative_volume(series_from([100.0] * 5, volumes=[100] * 5))

    assert result.available is False


# ---------------------------------------------------------------------------
# C1 -- ATR
# ---------------------------------------------------------------------------


def test_atr_of_constant_range():
    """Every bar spans exactly 2.0 with no gaps -- ATR must be 2.0."""
    closes = [100.0] * (ATR_WINDOW + 1)
    result = average_true_range(series_from(closes))

    assert result.value == pytest.approx(2.0)


def test_atr_accounts_for_gaps():
    """A gap up makes true range exceed the bar's own high-low."""
    closes = [100.0] * ATR_WINDOW + [120.0]
    highs = [101.0] * ATR_WINDOW + [121.0]
    lows = [99.0] * ATR_WINDOW + [119.0]

    result = average_true_range(series_from(closes, highs=highs, lows=lows))

    assert result.value > 2.0


def test_atr_percent_normalises_by_price():
    """A 2-rupee range means more on a 100 stock than on a 1000 one."""
    cheap = atr_percent(series_from([100.0] * (ATR_WINDOW + 1)))
    dear = atr_percent(series_from([1000.0] * (ATR_WINDOW + 1)))

    assert cheap.value > dear.value
    assert cheap.value == pytest.approx(2.0)


def test_atr_needs_history():
    result = average_true_range(series_from([100.0] * 5))

    assert result.available is False
    assert "ATR(14)" in result.reason


def test_atr_percent_propagates_unavailability():
    result = atr_percent(series_from([100.0] * 5))

    assert result.available is False


# ---------------------------------------------------------------------------
# C2 -- volatility percentile
# ---------------------------------------------------------------------------


def test_volatility_percentile_high_when_range_expands():
    quiet = [100.0] * 300
    highs = [100.5] * 300
    lows = [99.5] * 300
    # Final bar has a much wider range.
    highs[-1], lows[-1] = 130.0, 70.0

    result = realized_volatility_percentile(
        series_from(quiet, highs=highs, lows=lows)
    )

    assert result.available
    assert result.value > 90.0


def test_volatility_percentile_needs_a_year_of_history():
    result = realized_volatility_percentile(series_from([100.0] * 100))

    assert result.available is False


# ---------------------------------------------------------------------------
# Sparse-data policy
# ---------------------------------------------------------------------------


def test_indicators_refuse_a_sparse_window():
    """Feed gaps must not silently produce a confident number."""
    cal = TradingCalendar()
    full = cal.sessions_in_range(date(2025, 1, 1), date(2026, 9, 18))
    sparse = full[::2]

    bars = tuple(
        Bar(session_date=d, open=100, high=101, low=99, close=100, volume=1000)
        for d in sparse
    )
    series = PriceSeries(instrument_id=1, as_of=date(2026, 9, 18), bars=bars)

    result = average_true_range(series, calendar=cal)

    assert result.available is False
    assert "complete" in result.reason


# ---------------------------------------------------------------------------
# A6 -- delivery percentage trend
# ---------------------------------------------------------------------------


def _delivery(pcts: list[float | None], traded: int = 1000):
    """Build delivery records; None means a session where nothing traded."""
    from algorix.models import DeliveryRecord

    start = date(2020, 1, 1)
    records = []
    for i, pct in enumerate(pcts):
        if pct is None:
            records.append(
                DeliveryRecord(
                    session_date=start + timedelta(days=i),
                    traded_quantity=0,
                    delivered_quantity=0,
                )
            )
        else:
            records.append(
                DeliveryRecord(
                    session_date=start + timedelta(days=i),
                    traded_quantity=traded,
                    delivered_quantity=int(traded * pct / 100),
                )
            )
    return records


def test_rising_delivery_scores_above_one():
    from algorix.indicators import delivery_trend

    result = delivery_trend(_delivery([40.0] * 15 + [80.0] * 5))

    assert result.available
    assert result.value > 1.0


def test_falling_delivery_scores_below_one():
    from algorix.indicators import delivery_trend

    result = delivery_trend(_delivery([60.0] * 15 + [30.0] * 5))

    assert result.value < 1.0


def test_steady_delivery_scores_one():
    from algorix.indicators import delivery_trend

    result = delivery_trend(_delivery([50.0] * 20))

    assert result.value == pytest.approx(1.0)


def test_delivery_trend_is_relative_to_the_stocks_own_norm():
    """High absolute delivery is not the signal -- a change from norm is."""
    from algorix.indicators import delivery_trend

    always_high = delivery_trend(_delivery([85.0] * 20))
    rising_low = delivery_trend(_delivery([20.0] * 15 + [40.0] * 5))

    assert always_high.value == pytest.approx(1.0)
    assert rising_low.value > 1.0


def test_unpublished_sessions_are_skipped_not_counted_as_zero():
    """0/0 is undefined; treating it as 0% would fake a delivery collapse."""
    from algorix.indicators import delivery_trend

    with_gaps = _delivery([50.0] * 10 + [None] * 3 + [50.0] * 10)

    result = delivery_trend(with_gaps)

    assert result.available
    assert result.value == pytest.approx(1.0)


def test_delivery_trend_needs_a_baseline():
    from algorix.indicators import delivery_trend

    result = delivery_trend(_delivery([50.0] * 5))

    assert result.available is False
    assert "needs 20 published sessions" in result.reason


def test_delivery_trend_rejects_inverted_windows():
    from algorix.indicators import delivery_trend

    with pytest.raises(ValueError, match="cannot exceed baseline"):
        delivery_trend(_delivery([50.0] * 30), recent=30, baseline=20)


def test_latest_delivery_pct():
    from algorix.indicators import latest_delivery_pct

    result = latest_delivery_pct(_delivery([40.0, 50.0, 77.0]))

    assert result.value == pytest.approx(77.0, abs=0.2)


def test_latest_delivery_pct_with_no_published_sessions():
    from algorix.indicators import latest_delivery_pct

    result = latest_delivery_pct(_delivery([None, None]))

    assert result.available is False
