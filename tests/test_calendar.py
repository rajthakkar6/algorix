"""Tests for the NSE/BSE trading calendar.

Dates used here are verified against the XBOM calendar rather than asserted
from memory. Weekend/holiday facts (Republic Day, Independence Day) are
fixed-date national holidays and safe to assert directly.
"""

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from algorix.calendar import CalendarRangeError, TradingCalendar

IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture
def cal():
    return TradingCalendar()


# --------------------------------------------------------------------------
# Positive cases
# --------------------------------------------------------------------------


def test_weekday_is_a_trading_day(cal):
    # Friday 18 Sep 2026, an ordinary weekday.
    assert cal.is_trading_day(date(2026, 9, 18)) is True


def test_saturday_is_not_a_trading_day(cal):
    assert cal.is_trading_day(date(2026, 9, 19)) is False


def test_sunday_is_not_a_trading_day(cal):
    assert cal.is_trading_day(date(2026, 9, 20)) is False


def test_republic_day_is_a_holiday(cal):
    """26 January is a fixed national holiday."""
    assert cal.is_trading_day(date(2026, 1, 26)) is False


def test_independence_day_is_a_holiday(cal):
    """15 August is a fixed national holiday (2026 falls on a Saturday)."""
    assert cal.is_trading_day(date(2026, 8, 15)) is False


def test_previous_trading_day_skips_the_weekend(cal):
    # Monday -> previous session is the preceding Friday.
    assert cal.previous_trading_day(date(2026, 9, 21)) == date(2026, 9, 18)


def test_next_trading_day_skips_the_weekend(cal):
    assert cal.next_trading_day(date(2026, 9, 18)) == date(2026, 9, 21)


def test_previous_trading_day_is_strict(cal):
    """Called on a trading day, it returns the day before -- not the same day."""
    result = cal.previous_trading_day(date(2026, 9, 18))

    assert result < date(2026, 9, 18)
    assert cal.is_trading_day(result)


def test_midweek_holiday_is_not_a_trading_day(cal):
    """Mon 14 Sep 2026 is a market holiday, not a weekend.

    Regression guard: the original version of these tests assumed every
    Mon-Fri is a session. Mid-week holidays are common on the Indian calendar
    and shift year to year with the lunar calendar -- which is why holiday
    data is read from the calendar library, never hardcoded.
    """
    assert cal.is_trading_day(date(2026, 9, 14)) is False


def test_sessions_in_range_excludes_weekends(cal):
    # Mon 21 - Fri 25 Sep 2026 is a clean week with no holiday.
    sessions = cal.sessions_in_range(date(2026, 9, 21), date(2026, 9, 27))

    # Mon-Fri only; Sat 26th and Sun 27th excluded.
    assert sessions == [
        date(2026, 9, 21),
        date(2026, 9, 22),
        date(2026, 9, 23),
        date(2026, 9, 24),
        date(2026, 9, 25),
    ]


def test_sessions_in_range_excludes_midweek_holiday(cal):
    sessions = cal.sessions_in_range(date(2026, 9, 14), date(2026, 9, 20))

    # Mon 14th is a holiday; Sat 19th and Sun 20th are the weekend.
    assert sessions == [
        date(2026, 9, 15),
        date(2026, 9, 16),
        date(2026, 9, 17),
        date(2026, 9, 18),
    ]


def test_sessions_in_range_single_day(cal):
    assert cal.sessions_in_range(date(2026, 9, 18), date(2026, 9, 18)) == [
        date(2026, 9, 18)
    ]


def test_trading_days_ago_zero_returns_same_day(cal):
    assert cal.trading_days_ago(date(2026, 9, 18), 0) == date(2026, 9, 18)


def test_trading_days_ago_counts_sessions_not_calendar_days(cal):
    """5 sessions back from Fri 18 Sep skips the weekend *and* Mon 14th.

    Sessions walking back: 17th, 16th, 15th, (14th is a holiday), 11th, 10th.
    A naive calendar-day subtraction would land on the 13th -- a Sunday.
    """
    result = cal.trading_days_ago(date(2026, 9, 18), 5)

    assert result == date(2026, 9, 10)
    assert cal.is_trading_day(result)


def test_previous_trading_day_skips_a_midweek_holiday(cal):
    """Tue 15th -> previous session is Fri 11th, skipping the Mon 14th holiday."""
    assert cal.previous_trading_day(date(2026, 9, 15)) == date(2026, 9, 11)


# -- last_completed_session: the scanner's critical path --------------------


def test_before_open_returns_previous_session(cal):
    """07:30 IST -- the scanner's actual run time. Today has no data yet."""
    now = datetime(2026, 9, 18, 7, 30, tzinfo=IST)

    assert cal.last_completed_session(now) == date(2026, 9, 17)


def test_during_session_returns_previous_session(cal):
    """Mid-session at 12:00, today's close does not exist yet."""
    now = datetime(2026, 9, 18, 12, 0, tzinfo=IST)

    assert cal.last_completed_session(now) == date(2026, 9, 17)


def test_exactly_at_close_counts_as_complete(cal):
    """15:30:00 IST is the close -- today's data now exists."""
    now = datetime(2026, 9, 18, 15, 30, tzinfo=IST)

    assert cal.last_completed_session(now) == date(2026, 9, 18)


def test_after_close_returns_today(cal):
    now = datetime(2026, 9, 18, 18, 0, tzinfo=IST)

    assert cal.last_completed_session(now) == date(2026, 9, 18)


def test_one_minute_before_close_is_not_complete(cal):
    """Boundary: 15:29 is still mid-session."""
    now = datetime(2026, 9, 18, 15, 29, tzinfo=IST)

    assert cal.last_completed_session(now) == date(2026, 9, 17)


def test_on_a_weekend_returns_fridays_session(cal):
    now = datetime(2026, 9, 19, 10, 0, tzinfo=IST)  # Saturday

    assert cal.last_completed_session(now) == date(2026, 9, 18)


def test_on_a_holiday_returns_previous_session(cal):
    now = datetime(2026, 1, 26, 10, 0, tzinfo=IST)  # Republic Day

    result = cal.last_completed_session(now)

    assert result < date(2026, 1, 26)
    assert cal.is_trading_day(result)


def test_non_ist_timezone_is_converted_not_misread(cal):
    """03:00 UTC is 08:30 IST -- same day, still before the open."""
    now = datetime(2026, 9, 18, 3, 0, tzinfo=timezone.utc)

    assert cal.last_completed_session(now) == date(2026, 9, 17)


def test_utc_late_evening_crosses_into_next_ist_day(cal):
    """20:00 UTC Thu = 01:30 IST Fri -- before Friday's open."""
    now = datetime(2026, 9, 17, 20, 0, tzinfo=timezone.utc)

    assert cal.last_completed_session(now) == date(2026, 9, 17)


# -- overrides --------------------------------------------------------------


def test_extra_holiday_closes_a_normal_session():
    """An unscheduled NSE closure can be injected without editing the module."""
    cal = TradingCalendar(extra_holidays={date(2026, 9, 18)})

    assert cal.is_trading_day(date(2026, 9, 18)) is False
    assert cal.last_completed_session(
        datetime(2026, 9, 18, 18, 0, tzinfo=IST)
    ) == date(2026, 9, 17)


def test_extra_session_opens_a_normal_holiday():
    """A special session (e.g. Muhurat trading) can be injected."""
    cal = TradingCalendar(extra_sessions={date(2026, 9, 19)})  # a Saturday

    assert cal.is_trading_day(date(2026, 9, 19)) is True


def test_extra_holiday_removed_from_sessions_in_range():
    cal = TradingCalendar(extra_holidays={date(2026, 9, 16)})

    sessions = cal.sessions_in_range(date(2026, 9, 14), date(2026, 9, 18))

    # Base sessions for that week are the 15th-18th (the 14th is a holiday);
    # overriding the 16th leaves three.
    assert date(2026, 9, 16) not in sessions
    assert sessions == [date(2026, 9, 15), date(2026, 9, 17), date(2026, 9, 18)]


def test_extra_session_added_to_sessions_in_range_in_order():
    cal = TradingCalendar(extra_sessions={date(2026, 9, 19)})

    sessions = cal.sessions_in_range(date(2026, 9, 14), date(2026, 9, 20))

    assert sessions[-1] == date(2026, 9, 19)
    assert sessions == sorted(sessions)


# --------------------------------------------------------------------------
# Negative cases
# --------------------------------------------------------------------------


def test_naive_datetime_is_rejected(cal):
    """A naive datetime can be a full session wrong -- never guess the zone."""
    with pytest.raises(ValueError, match="timezone-aware"):
        cal.last_completed_session(datetime(2026, 9, 18, 7, 30))


def test_date_before_coverage_raises(cal):
    with pytest.raises(CalendarRangeError, match="outside calendar coverage"):
        cal.is_trading_day(date(1800, 1, 1))


def test_date_after_coverage_raises(cal):
    """Querying an unpublished year must raise, not invent an answer."""
    with pytest.raises(CalendarRangeError, match="outside calendar coverage"):
        cal.is_trading_day(date(2200, 1, 1))


def test_reversed_range_is_rejected(cal):
    with pytest.raises(ValueError, match="is after end"):
        cal.sessions_in_range(date(2026, 9, 18), date(2026, 9, 14))


def test_negative_lookback_is_rejected(cal):
    with pytest.raises(ValueError, match="non-negative"):
        cal.trading_days_ago(date(2026, 9, 18), -1)


def test_contradictory_overrides_are_rejected():
    """A date cannot be both closed and open -- fail loudly at construction."""
    with pytest.raises(ValueError, match="cannot be both"):
        TradingCalendar(
            extra_holidays={date(2026, 9, 18)},
            extra_sessions={date(2026, 9, 18)},
        )
