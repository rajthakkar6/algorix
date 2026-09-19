"""Tests for the web UI.

Route-level tests against a seeded database: the UI must render real scores,
and -- the rule carried from the rest of the codebase -- must show missing
data as missing rather than as zero.
"""

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from algorix.journal import Journal
from algorix.models import Bar, DeliveryRecord, Exchange
from algorix.regime import seed_regime_instruments
from algorix.storage import (
    BarRepository,
    Database,
    DeliveryRepository,
    InstrumentRepository,
)
from algorix.universe import NIFTY_50, ConstituencyRepository, ConstituentRecord
from algorix.web import app, configure

TARGET = date(2026, 9, 18)


@pytest.fixture
def seeded(tmp_path, monkeypatch):
    from algorix.calendar import TradingCalendar

    cal = TradingCalendar()
    db = Database(tmp_path / "web.db")
    db.migrate()
    seed_regime_instruments(db)

    repo = InstrumentRepository(db)
    bars = BarRepository(db)
    delivery = DeliveryRepository(db)

    symbols = [f"S{i}" for i in range(12)]
    ConstituencyRepository(db).sync(
        NIFTY_50,
        [ConstituentRecord(symbol=s, name=f"{s} Ltd.") for s in symbols],
        date(2024, 1, 1),
    )
    sessions = cal.sessions_in_range(cal.trading_days_ago(TARGET, 400), TARGET)

    for i, symbol in enumerate(symbols):
        instrument = repo.get(symbol, Exchange.NSE)
        rate = i / 10000.0
        bars.upsert_many(
            instrument.id,
            [Bar(session_date=d, open=p, high=p + 1, low=p - 1, close=p,
                 volume=200_000)
             for d, p in zip(sessions,
                             [100.0 * ((1 + rate) ** j) for j in range(len(sessions))])],
            source="test",
        )
        delivery.upsert_many(
            instrument.id,
            [DeliveryRecord(session_date=d, traded_quantity=1000,
                            delivered_quantity=500) for d in sessions[-25:]],
            source="test",
        )

    # Pin "now" so the UI resolves to the seeded session.
    import algorix.web.server as web_app

    monkeypatch.setattr(web_app, "_current_session", lambda: TARGET)
    configure(tmp_path / "web.db", NIFTY_50)
    return db


@pytest.fixture
def client(seeded):
    return TestClient(app)


# --- dashboard ------------------------------------------------------------


def test_dashboard_renders(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "Morning scan" in response.text


def test_dashboard_lists_scored_symbols(client):
    text = client.get("/").text

    assert "S11" in text
    assert "/stock/S11" in text


def test_dashboard_shows_the_regime_banner(client):
    text = client.get("/").text

    assert "regime" in text.lower()


def test_dashboard_states_the_scoring_method(client):
    """Principle 0.2a -- method shown, never implied."""
    assert "cross-sectional" in client.get("/").text


def test_dashboard_respects_top_parameter(client):
    few = client.get("/?top=3").text

    assert few.count('class="sym"') == 3


def test_dashboard_on_empty_database(tmp_path, monkeypatch):
    import algorix.web.server as web_app

    empty = Database(tmp_path / "empty.db")
    empty.migrate()
    monkeypatch.setattr(web_app, "_current_session", lambda: TARGET)
    configure(tmp_path / "empty.db", NIFTY_50)

    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert "Nothing to show" in response.text


# --- stock detail ---------------------------------------------------------


def test_stock_page_renders(client):
    response = client.get("/stock/S5")

    assert response.status_code == 200
    assert "S5" in response.text


def test_stock_page_lists_indicators(client):
    text = client.get("/stock/S5").text

    assert "52-week high proximity" in text
    assert "delivery trend" in text


def test_stock_page_draws_a_sparkline(client):
    assert "<polyline" in client.get("/stock/S5").text


def test_unknown_symbol_is_handled(client):
    response = client.get("/stock/NOSUCH")

    assert response.status_code == 200
    assert "not in the database" in response.text


def test_symbol_is_case_insensitive(client):
    assert client.get("/stock/s5").status_code == 200


def test_missing_indicator_shows_a_dash_not_zero(seeded, tmp_path, monkeypatch):
    """The core display rule: absence must not render as 0."""
    import algorix.web.server as web_app

    repo = InstrumentRepository(seeded)
    bars = BarRepository(seeded)
    # A stock with only a handful of bars: most indicators cannot compute.
    from algorix.models import Instrument, InstrumentType

    stub_id = repo.upsert(
        Instrument(symbol="STUB", exchange=Exchange.NSE,
                   instrument_type=InstrumentType.EQUITY)
    )
    bars.upsert_many(
        stub_id,
        [Bar(session_date=TARGET - timedelta(days=i), open=10, high=11, low=9,
             close=10, volume=100) for i in range(3)],
        source="test",
    )
    monkeypatch.setattr(web_app, "_current_session", lambda: TARGET)

    text = TestClient(app).get("/stock/STUB").text

    assert "--" in text
    assert "never shown as zero" in text


# --- backtest -------------------------------------------------------------


def test_backtest_page_renders(client):
    response = client.get("/backtest?sessions=30")

    assert response.status_code == 200
    assert "Backtest" in response.text


def test_backtest_page_shows_caveats(client):
    """Caveats must be visible in the UI, not just the CLI."""
    text = client.get("/backtest?sessions=30").text

    assert "Caveats" in text


def test_backtest_page_shows_per_signal_table(client):
    text = client.get("/backtest?sessions=40").text

    assert "Per-signal" in text


# --- journal --------------------------------------------------------------


def test_journal_page_empty_state(client):
    text = client.get("/journal").text

    assert "No scans recorded yet" in text


def test_journal_page_lists_recorded_scans(seeded, client):
    from algorix.cross_sectional import compute_universe_momentum
    from algorix.scoring import score_universe
    from algorix.series import load_series
    from algorix.calendar import TradingCalendar

    cal = TradingCalendar()
    repo = InstrumentRepository(seeded)
    series = {}
    for symbol in [f"S{i}" for i in range(12)]:
        instrument = repo.get(symbol, Exchange.NSE)
        series[symbol] = load_series(seeded, instrument.id, TARGET, 400, cal)
    universe = score_universe(
        series, compute_universe_momentum(series, TARGET, cal), TARGET, calendar=cal
    )
    Journal(seeded).record(universe)

    text = client.get("/journal").text

    assert str(TARGET) in text
    assert "S11" in text
