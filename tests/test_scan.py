"""Tests for the morning scanner end to end."""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from algorix.calendar import TradingCalendar
from algorix.exceptions import SourceUnreachableError
from algorix.journal import Journal
from algorix.models import Bar, DeliveryRecord, Exchange, Instrument, InstrumentType
from algorix.notify import NotConfiguredError
from algorix.regime import FlowRecord, seed_regime_instruments
from algorix.scan import run_scan
from algorix.storage import (
    BarRepository,
    Database,
    DeliveryRepository,
    InstrumentRepository,
)
from algorix.universe import NIFTY_50, ConstituencyRepository, ConstituentRecord

IST = ZoneInfo("Asia/Kolkata")
TARGET = date(2026, 9, 18)
NOW = datetime(2026, 9, 19, 8, 0, tzinfo=IST)


@pytest.fixture
def cal():
    return TradingCalendar()


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "test.db")
    database.migrate()
    return database


class FakeNotifier:
    def __init__(self, error=None):
        self.error = error
        self.sent = []

    def send(self, message, timeout_seconds=30.0):
        if self.error:
            raise self.error
        self.sent.append(message)
        return True


class FakeFlowClient:
    def fetch(self):
        return [FlowRecord(TARGET, "FII/FPI", 100.0, 50.0, 50.0),
                FlowRecord(TARGET, "DII", 100.0, 50.0, 50.0)]


def seed_universe(db, cal, count=12, sessions=400):
    """Populate a scannable universe with price and delivery history."""
    repo = InstrumentRepository(db)
    bars = BarRepository(db)
    delivery = DeliveryRepository(db)
    constituency = ConstituencyRepository(db)
    seed_regime_instruments(db)

    session_dates = cal.sessions_in_range(
        cal.trading_days_ago(TARGET, sessions - 1), TARGET
    )
    symbols = [f"S{i}" for i in range(count)]
    constituency.sync(
        NIFTY_50,
        [ConstituentRecord(symbol=s, name=f"{s} Ltd.") for s in symbols],
        date(2025, 1, 1),
    )

    for i, symbol in enumerate(symbols):
        instrument = repo.get(symbol, Exchange.NSE)
        rate = i / 10000.0
        bars.upsert_many(
            instrument.id,
            [
                Bar(session_date=d, open=p, high=p + 1, low=p - 1, close=p,
                    volume=200_000)
                for d, p in zip(
                    session_dates,
                    [100.0 * ((1 + rate) ** j) for j in range(len(session_dates))],
                )
            ],
            source="test",
        )
        delivery.upsert_many(
            instrument.id,
            [
                DeliveryRecord(session_date=d, traded_quantity=1000,
                               delivered_quantity=500)
                for d in session_dates[-25:]
            ],
            source="test",
        )
    return symbols


def scan(db, cal, **kwargs):
    defaults = dict(now=NOW, calendar=cal, notifier=FakeNotifier(),
                    flow_client=FakeFlowClient())
    defaults.update(kwargs)
    return run_scan(db, **defaults)


# --- happy path -----------------------------------------------------------


def test_scan_targets_the_last_completed_session(db, cal):
    seed_universe(db, cal)

    assert scan(db, cal).session_date == TARGET


def test_scan_scores_the_universe(db, cal):
    seed_universe(db, cal)

    result = scan(db, cal)

    assert len(result.universe.ranked()) > 0
    assert result.universe.top(1)[0].symbol == "S11"


def test_scan_produces_a_digest(db, cal):
    seed_universe(db, cal)

    result = scan(db, cal)

    assert "Algorix" in result.digest
    assert "S11" in result.digest


def test_scan_writes_to_the_journal(db, cal):
    seed_universe(db, cal)

    result = scan(db, cal)

    assert result.journal_run_id is not None
    assert Journal(db).scores_for(TARGET)


def test_scan_delivers_the_digest(db, cal):
    seed_universe(db, cal)
    notifier = FakeNotifier()

    result = scan(db, cal, notifier=notifier)

    assert result.delivered is True
    assert len(notifier.sent) == 1


def test_scan_assesses_the_regime(db, cal):
    seed_universe(db, cal)

    result = scan(db, cal)

    assert result.regime is not None


def test_scan_is_repeatable(db, cal):
    seed_universe(db, cal)

    scan(db, cal)
    scan(db, cal)

    assert len(Journal(db).scores_for(TARGET)) == 12


def test_no_send_skips_delivery(db, cal):
    seed_universe(db, cal)
    notifier = FakeNotifier()

    result = scan(db, cal, deliver=False, notifier=notifier)

    assert result.delivered is False
    assert notifier.sent == []
    assert result.journal_run_id is not None  # still journalled


# --- failure handling -----------------------------------------------------


def test_journal_is_written_before_delivery(db, cal):
    """A Telegram outage must not cost the record."""
    seed_universe(db, cal)

    result = scan(
        db, cal, notifier=FakeNotifier(error=SourceUnreachableError("down"))
    )

    assert result.delivered is False
    assert result.journal_run_id is not None
    assert Journal(db).scores_for(TARGET)
    assert any("delivery failed" in e for e in result.errors)


def test_unconfigured_telegram_is_reported_not_fatal(db, cal):
    seed_universe(db, cal)

    result = scan(
        db, cal, notifier=FakeNotifier(error=NotConfiguredError("no token"))
    )

    assert result.delivered is False
    assert result.journal_run_id is not None
    assert any("no token" in e for e in result.errors)


def test_flow_failure_does_not_stop_the_scan(db, cal):
    seed_universe(db, cal)

    class DeadFlows:
        def fetch(self):
            raise SourceUnreachableError("NSE down")

    result = scan(db, cal, flow_client=DeadFlows())

    assert result.universe.ranked()
    assert any("FII/DII unavailable" in e for e in result.errors)


def test_empty_database_reports_clearly(db, cal):
    result = scan(db, cal)

    assert result.universe.scores == {}
    assert any("run `python -m algorix.refresh` first" in e for e in result.errors)
    assert "No instruments scored" in result.digest


def test_summary_is_human_readable(db, cal):
    seed_universe(db, cal)

    summary = scan(db, cal).summary()

    assert "Session:" in summary
    assert "Journalled: yes" in summary
