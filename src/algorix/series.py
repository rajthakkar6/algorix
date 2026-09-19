"""Point-in-time price series for indicator computation.

Two rules govern everything here, and both exist to stop a signal looking
valid when it is not.

**No look-ahead.** A series loaded `as_of` date D contains nothing after D.
Every indicator therefore sees only what was knowable at the time. Without
this, a backtest quietly reports the future and every result is optimistic.

**Insufficient history yields nothing, never an approximation.** A 200-day
moving average over 150 sessions is not a 200-day moving average; it is a
different, shorter indicator wearing the same name. Indicators return an
explicit "unavailable, and here is why" rather than a number computed on
whatever happened to be there.

Feed gaps are real (see PROJECT_SCOPE §5.1 -- yfinance drops individual
sessions), so a series also knows how complete it is relative to the trading
calendar, and indicators can refuse data that is too sparse to trust.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from algorix.calendar import TradingCalendar
from algorix.models import Bar
from algorix.storage import BarRepository, Database

#: Fraction of expected sessions that must be present for a window to be
#: usable. Feeds drop the odd session; a window missing more than this is not
#: measuring what it claims to.
DEFAULT_MIN_COMPLETENESS = 0.95


@dataclass(frozen=True)
class IndicatorValue:
    """An indicator result, or an explicit statement that there isn't one.

    Indicators never return a bare float, because "unavailable" and "zero"
    must not be confusable at any call site.
    """

    value: float | None = None
    reason: str | None = None

    @property
    def available(self) -> bool:
        return self.value is not None

    @classmethod
    def of(cls, value: float) -> "IndicatorValue":
        return cls(value=value)

    @classmethod
    def unavailable(cls, reason: str) -> "IndicatorValue":
        return cls(value=None, reason=reason)

    def __bool__(self) -> bool:
        # Guards `if value:` from reading a legitimate 0.0 as absent.
        raise TypeError(
            "IndicatorValue has no truth value -- check .available explicitly"
        )


@dataclass(frozen=True)
class PriceSeries:
    """Bars for one instrument, oldest first, none later than `as_of`."""

    instrument_id: int
    as_of: date
    bars: tuple[Bar, ...]

    def __len__(self) -> int:
        return len(self.bars)

    @property
    def closes(self) -> list[float]:
        return [b.close for b in self.bars]

    @property
    def highs(self) -> list[float]:
        return [b.high for b in self.bars]

    @property
    def lows(self) -> list[float]:
        return [b.low for b in self.bars]

    @property
    def volumes(self) -> list[int]:
        return [b.volume for b in self.bars]

    @property
    def sessions(self) -> list[date]:
        return [b.session_date for b in self.bars]

    @property
    def latest(self) -> Bar | None:
        return self.bars[-1] if self.bars else None

    def window(self, n: int) -> tuple[Bar, ...]:
        """The most recent `n` bars. Shorter than `n` if history is short."""
        if n <= 0:
            raise ValueError(f"window size must be positive, got {n}")
        return self.bars[-n:]

    def truncate_to(self, as_of: date) -> "PriceSeries":
        """This series as it stood on `as_of` -- nothing later is visible.

        The in-memory equivalent of reloading with an earlier `as_of`, so a
        historical replay can slice one loaded history thousands of times
        instead of re-querying. The no-look-ahead guarantee is identical:
        bars after `as_of` are dropped, not hidden.
        """
        kept = tuple(b for b in self.bars if b.session_date <= as_of)
        return PriceSeries(
            instrument_id=self.instrument_id, as_of=as_of, bars=kept
        )

    def forward_return(self, from_date: date, sessions: int) -> float | None:
        """Percentage return over the `sessions` bars after `from_date`.

        Used only for evaluation, never for scoring -- this deliberately
        looks ahead, which is exactly why it must never be reachable from an
        indicator. Returns None when the future has not happened yet.
        """
        if sessions <= 0:
            raise ValueError(f"sessions must be positive, got {sessions}")

        index = next(
            (i for i, b in enumerate(self.bars) if b.session_date == from_date),
            None,
        )
        if index is None:
            return None

        target = index + sessions
        if target >= len(self.bars):
            return None

        start = self.bars[index].close
        if start <= 0:
            return None
        return ((self.bars[target].close / start) - 1.0) * 100.0

    def require(
        self,
        sessions: int,
        calendar: TradingCalendar | None = None,
        min_completeness: float = DEFAULT_MIN_COMPLETENESS,
    ) -> str | None:
        """Why a window of `sessions` cannot be trusted, or None if it can.

        Checks both depth (enough bars at all) and density (not too many
        missing sessions inside the window).
        """
        if len(self.bars) < sessions:
            return (
                f"needs {sessions} sessions, has {len(self.bars)}"
            )

        if calendar is None:
            return None

        window = self.window(sessions)
        expected = calendar.sessions_in_range(
            window[0].session_date, window[-1].session_date
        )
        if not expected:
            return None

        completeness = len(window) / len(expected)
        if completeness < min_completeness:
            missing = len(expected) - len(window)
            return (
                f"window is {completeness:.0%} complete "
                f"({missing} of {len(expected)} sessions missing)"
            )
        return None


def load_series(
    db: Database,
    instrument_id: int,
    as_of: date,
    lookback_sessions: int,
    calendar: TradingCalendar | None = None,
) -> PriceSeries:
    """Load up to `lookback_sessions` bars ending at or before `as_of`.

    The start date is computed generously in calendar days, because sessions
    are sparser than days and a feed may have gaps -- then the result is
    trimmed to the requested session count.
    """
    if lookback_sessions <= 0:
        raise ValueError(
            f"lookback_sessions must be positive, got {lookback_sessions}"
        )

    # ~252 sessions a year; 1.6x plus a buffer comfortably covers weekends,
    # holidays and feed gaps without a second query.
    span_days = int(lookback_sessions * 1.6) + 30
    start = as_of - timedelta(days=span_days)

    bars = BarRepository(db).get_range(instrument_id, start, as_of)
    trimmed = tuple(bars[-lookback_sessions:]) if bars else ()

    return PriceSeries(instrument_id=instrument_id, as_of=as_of, bars=trimmed)
