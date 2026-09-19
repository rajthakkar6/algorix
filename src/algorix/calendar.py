"""NSE/BSE trading calendar.

Every date question in this system is an IST question, and most of them are
really one question: *what is the latest session whose data actually exists
yet?* The morning scanner runs around 07:30 IST, before the 09:15 open, so
"today" has no data -- it must score against the previous close. Getting that
boundary wrong does not crash anything; it silently scores stale or missing
data, which is exactly the failure CLAUDE.md's data rules exist to prevent.

Holiday data comes from `exchange_calendars`' XBOM calendar. XBOM is the BSE
calendar, but NSE and BSE observe identical holidays and identical hours
(09:15-15:30 IST), so it is authoritative for NSE too.

Holiday lists are not hardcoded here on purpose. NSE publishes them annually
and they shift with the lunar calendar; a hand-written list would be wrong in
a way nothing would catch. Where NSE announces an unscheduled closure or a
special session (Muhurat trading), use the `extra_holidays` / `extra_sessions`
overrides rather than editing this module.
"""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

from algorix.config import MARKET_TIMEZONE
from algorix.exceptions import AlgorixError

#: `exchange_calendars` code for the Indian equity market.
CALENDAR_CODE = "XBOM"

#: Regular session close, IST. A session is only "complete" once this passes.
SESSION_CLOSE = time(15, 30)

IST = ZoneInfo(MARKET_TIMEZONE)


class CalendarRangeError(AlgorixError):
    """A date was requested outside the calendar's known coverage.

    Raised rather than guessed. Returning a plausible answer for a year whose
    holidays are not yet published would be silently wrong.
    """


class TradingCalendar:
    """Trading-day arithmetic for NSE/BSE.

    Args:
        extra_holidays: dates to treat as closed despite the base calendar
            (unscheduled closures announced by NSE).
        extra_sessions: dates to treat as open despite the base calendar
            (e.g. a special Muhurat session the library does not carry).
    """

    def __init__(
        self,
        extra_holidays: set[date] | None = None,
        extra_sessions: set[date] | None = None,
    ) -> None:
        self._calendar = xcals.get_calendar(CALENDAR_CODE)
        self._extra_holidays = set(extra_holidays or ())
        self._extra_sessions = set(extra_sessions or ())

        overlap = self._extra_holidays & self._extra_sessions
        if overlap:
            raise ValueError(
                "A date cannot be both an extra holiday and an extra session: "
                f"{sorted(overlap)}"
            )

    # -- coverage ---------------------------------------------------------

    @property
    def first_supported_date(self) -> date:
        return self._calendar.first_session.date()

    @property
    def last_supported_date(self) -> date:
        return self._calendar.last_session.date()

    def _check_in_range(self, day: date) -> None:
        if not (self.first_supported_date <= day <= self.last_supported_date):
            raise CalendarRangeError(
                f"{day} is outside calendar coverage "
                f"({self.first_supported_date} to {self.last_supported_date})"
            )

    # -- queries ----------------------------------------------------------

    def is_trading_day(self, day: date) -> bool:
        """True if `day` is a trading session."""
        self._check_in_range(day)
        if day in self._extra_holidays:
            return False
        if day in self._extra_sessions:
            return True
        return self._calendar.is_session(pd.Timestamp(day))

    def previous_trading_day(self, day: date) -> date:
        """The latest session strictly before `day`."""
        self._check_in_range(day)
        cursor = day
        # Bounded scan: the longest NSE closure is a handful of days, but the
        # range check terminates this even against a pathological calendar.
        while True:
            cursor = cursor - pd.Timedelta(days=1).to_pytimedelta()
            self._check_in_range(cursor)
            if self.is_trading_day(cursor):
                return cursor

    def next_trading_day(self, day: date) -> date:
        """The earliest session strictly after `day`."""
        self._check_in_range(day)
        cursor = day
        while True:
            cursor = cursor + pd.Timedelta(days=1).to_pytimedelta()
            self._check_in_range(cursor)
            if self.is_trading_day(cursor):
                return cursor

    def last_completed_session(self, now: datetime) -> date:
        """The most recent session whose closing data exists as of `now`.

        This is the method the morning scanner depends on. At 07:30 IST on a
        trading day the session has not opened, so the answer is the previous
        session -- not today.

        Args:
            now: a timezone-aware datetime. Naive datetimes are rejected: the
                answer differs by a full session depending on the zone, so
                guessing would be a silent correctness bug.
        """
        if now.tzinfo is None:
            raise ValueError(
                "last_completed_session() requires a timezone-aware datetime; "
                "a naive one is ambiguous and can be a full session wrong"
            )

        local = now.astimezone(IST)
        today = local.date()
        self._check_in_range(today)

        if self.is_trading_day(today) and local.time() >= SESSION_CLOSE:
            return today
        return self.previous_trading_day(today)

    def sessions_in_range(self, start: date, end: date) -> list[date]:
        """All sessions from `start` to `end`, inclusive."""
        if start > end:
            raise ValueError(f"start {start} is after end {end}")
        self._check_in_range(start)
        self._check_in_range(end)

        sessions = [
            ts.date()
            for ts in self._calendar.sessions_in_range(
                pd.Timestamp(start), pd.Timestamp(end)
            )
        ]
        # Apply overrides, then re-sort so callers always get ascending order.
        result = {s for s in sessions if s not in self._extra_holidays}
        result |= {s for s in self._extra_sessions if start <= s <= end}
        return sorted(result)

    def trading_days_ago(self, day: date, n: int) -> date:
        """The session `n` trading days before `day`.

        `n=0` returns `day` itself. Used by lookback windows -- a 20-day
        moving average means 20 *sessions*, not 20 calendar days.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        self._check_in_range(day)

        cursor = day
        for _ in range(n):
            cursor = self.previous_trading_day(cursor)
        return cursor
