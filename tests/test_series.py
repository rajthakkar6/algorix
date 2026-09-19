"""Tests for point-in-time series loading."""

from datetime import date, timedelta

import pytest

from algorix.calendar import TradingCalendar
from algorix.models import Bar, Exchange, Instrument, InstrumentType
from algorix.series import IndicatorValue, PriceSeries, load_series
from algorix.storage import BarRepository, Database, InstrumentRepository

AS_OF = date(2026, 9, 18)


@pytest.fixture
def cal():
    return TradingCalendar()


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "test.db")
    database.migrate()
    return database


@pytest.fixture
def instrument_id(db):
    return InstrumentRepository(db).upsert(
        Instrument(
            symbol="TEST", exchange=Exchange.NSE, instrument_type=InstrumentType.EQUITY
        )
    )


def seed(db, instrument_id, sessions, close=100.0):
    BarRepository(db).upsert_many(
        instrument_id,
        [
            Bar(
                session_date=d,
                open=close,
                high=close + 1,
                low=close - 1,
                close=close,
                volume=1000,
            )
            for d in sessions
        ],
        source="test",
    )


def make_series(sessions, closes=None, instrument_id=1):
    closes = closes or [100.0] * len(sessions)
    return PriceSeries(
        instrument_id=instrument_id,
        as_of=sessions[-1],
        bars=tuple(
            Bar(
                session_date=d,
                open=c,
                high=c + 1,
                low=c - 1,
                close=c,
                volume=1000,
            )
            for d, c in zip(sessions, closes)
        ),
    )


# --------------------------------------------------------------------------
# IndicatorValue
# --------------------------------------------------------------------------


def test_available_value():
    value = IndicatorValue.of(42.0)

    assert value.available is True
    assert value.value == 42.0


def test_unavailable_carries_a_reason():
    value = IndicatorValue.unavailable("needs 200 sessions, has 12")

    assert value.available is False
    assert "200" in value.reason


def test_zero_is_available_not_absent():
    """0.0 is a real reading -- it must not read as missing."""
    value = IndicatorValue.of(0.0)

    assert value.available is True
    assert value.value == 0.0


def test_truthiness_is_refused():
    """`if value:` would silently treat 0.0 as absent -- so it raises."""
    with pytest.raises(TypeError, match="no truth value"):
        bool(IndicatorValue.of(0.0))


# --------------------------------------------------------------------------
# Point-in-time loading -- the look-ahead guard
# --------------------------------------------------------------------------


def test_series_excludes_bars_after_as_of(db, instrument_id, cal):
    """The core correctness rule: nothing later than as_of is visible."""
    sessions = cal.sessions_in_range(date(2026, 9, 1), date(2026, 9, 18))
    seed(db, instrument_id, sessions)

    series = load_series(db, instrument_id, date(2026, 9, 16), 50)

    assert max(series.sessions) == date(2026, 9, 16)
    assert date(2026, 9, 17) not in series.sessions


def test_series_is_oldest_first(db, instrument_id, cal):
    sessions = cal.sessions_in_range(date(2026, 9, 1), AS_OF)
    seed(db, instrument_id, sessions)

    series = load_series(db, instrument_id, AS_OF, 50)

    assert series.sessions == sorted(series.sessions)


def test_series_trims_to_lookback(db, instrument_id, cal):
    sessions = cal.sessions_in_range(date(2026, 1, 1), AS_OF)
    seed(db, instrument_id, sessions)

    series = load_series(db, instrument_id, AS_OF, 10)

    assert len(series) == 10
    assert series.sessions[-1] == AS_OF


def test_series_keeps_the_most_recent_bars(db, instrument_id, cal):
    sessions = cal.sessions_in_range(date(2026, 6, 1), AS_OF)
    seed(db, instrument_id, sessions)

    series = load_series(db, instrument_id, AS_OF, 5)

    assert series.sessions == sessions[-5:]


def test_empty_history_gives_empty_series(db, instrument_id):
    series = load_series(db, instrument_id, AS_OF, 50)

    assert len(series) == 0
    assert series.latest is None


def test_load_rejects_non_positive_lookback(db, instrument_id):
    with pytest.raises(ValueError, match="must be positive"):
        load_series(db, instrument_id, AS_OF, 0)


def test_long_lookback_spans_enough_calendar_days(db, instrument_id, cal):
    """252 sessions is ~1 year of calendar days -- the span must reach back."""
    sessions = cal.sessions_in_range(date(2025, 6, 1), AS_OF)
    seed(db, instrument_id, sessions)

    series = load_series(db, instrument_id, AS_OF, 252)

    assert len(series) == 252


# --------------------------------------------------------------------------
# Window and sufficiency
# --------------------------------------------------------------------------


def test_window_returns_most_recent(cal):
    sessions = cal.sessions_in_range(date(2026, 9, 1), AS_OF)
    series = make_series(sessions)

    assert len(series.window(3)) == 3
    assert series.window(3)[-1].session_date == AS_OF


def test_window_rejects_non_positive():
    series = make_series([AS_OF])

    with pytest.raises(ValueError, match="must be positive"):
        series.window(0)


def test_require_passes_with_enough_history(cal):
    sessions = cal.sessions_in_range(date(2026, 8, 1), AS_OF)
    series = make_series(sessions)

    assert series.require(5, cal) is None


def test_require_reports_insufficient_depth(cal):
    series = make_series(cal.sessions_in_range(date(2026, 9, 14), AS_OF))

    problem = series.require(200, cal)

    assert problem is not None
    assert "needs 200 sessions" in problem


def test_require_reports_a_sparse_window(cal):
    """A window riddled with feed gaps is not measuring what it claims."""
    full = cal.sessions_in_range(date(2026, 6, 1), AS_OF)
    sparse = full[::2]  # drop every other session
    series = make_series(sparse)

    problem = series.require(20, cal)

    assert problem is not None
    assert "complete" in problem


def test_require_tolerates_a_small_gap(cal):
    """One dropped session in twenty is within tolerance."""
    full = cal.sessions_in_range(date(2026, 8, 1), AS_OF)
    with_gap = [d for i, d in enumerate(full) if i != len(full) - 5]
    series = make_series(with_gap)

    assert series.require(len(with_gap), cal, min_completeness=0.9) is None


def test_require_without_calendar_checks_depth_only(cal):
    sparse = cal.sessions_in_range(date(2026, 6, 1), AS_OF)[::2]
    series = make_series(sparse)

    assert series.require(10, calendar=None) is None
