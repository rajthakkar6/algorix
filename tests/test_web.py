"""Tests for the web UI.

Route-level tests against a seeded database: the UI must render real scores,
and -- the rule carried from the rest of the codebase -- must show missing
data as missing rather than as zero.
"""

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from algorix.journal import Journal
from algorix.models import Bar, ChartDrawing, DeliveryRecord, Exchange
from algorix.regime import seed_regime_instruments
from algorix.storage import (
    BarRepository,
    ChartDrawingRepository,
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


def test_stock_page_embeds_the_chart(client):
    text = client.get("/stock/S5").text

    assert 'id="price-chart"' in text
    assert "/stock/S5/bars" in text


def test_stock_page_embeds_drawing_controls(client):
    text = client.get("/stock/S5").text

    assert 'data-tool="trendline"' in text
    assert 'data-tool="breakout"' in text
    assert 'data-tool="dip"' in text
    assert 'id="drawing-list"' in text
    assert "/static/chart-drawings.js" in text
    assert "/stock/S5/drawings" in text


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


# --- auto markers (breakout/dip highlights) ---------------------------------


def _trend(is_uptrend: bool):
    from algorix.indicators import TrendState

    return TrendState(
        above_fast=is_uptrend, above_slow=is_uptrend, fast_above_slow=is_uptrend,
        fast_rising=is_uptrend, pct_from_fast=2.0 if is_uptrend else -2.0,
        pct_from_slow=4.0 if is_uptrend else -4.0,
    )


def test_auto_markers_flags_a_breakout():
    from algorix.series import IndicatorValue
    from algorix.web.server import _auto_markers

    markers = _auto_markers(
        IndicatorValue.of(97.0), IndicatorValue.of(0.5), _trend(True), TARGET
    )

    assert len(markers) == 1
    assert markers[0]["text"] == "Breakout"
    assert markers[0]["time"] == TARGET.isoformat()


def test_auto_markers_flags_a_dip_within_an_uptrend():
    from algorix.series import IndicatorValue
    from algorix.web.server import _auto_markers

    markers = _auto_markers(
        IndicatorValue.of(50.0), IndicatorValue.of(-5.0), _trend(True), TARGET
    )

    assert len(markers) == 1
    assert markers[0]["text"] == "Dip"


def test_auto_markers_ignores_a_pullback_outside_an_uptrend():
    """A falling stock pulling back further is not a 'dip worth watching' --
    the dip marker only means something inside an established uptrend."""
    from algorix.series import IndicatorValue
    from algorix.web.server import _auto_markers

    markers = _auto_markers(
        IndicatorValue.of(50.0), IndicatorValue.of(-5.0), _trend(False), TARGET
    )

    assert markers == []


def test_auto_markers_handles_unavailable_indicators():
    """A missing indicator must not crash the marker computation or be
    silently treated as satisfying the threshold."""
    from algorix.series import IndicatorValue
    from algorix.web.server import _auto_markers

    markers = _auto_markers(
        IndicatorValue.unavailable("not enough history"),
        IndicatorValue.unavailable("not enough history"),
        _trend(True), TARGET,
    )

    assert markers == []


def test_auto_markers_neither_condition_is_empty():
    from algorix.series import IndicatorValue
    from algorix.web.server import _auto_markers

    markers = _auto_markers(
        IndicatorValue.of(50.0), IndicatorValue.of(0.5), _trend(True), TARGET
    )

    assert markers == []


def test_auto_markers_with_no_trend_state_is_empty():
    """trend_state() returns None when history is too short -- must not
    crash, and a dip can't be evaluated without knowing the trend."""
    from algorix.series import IndicatorValue
    from algorix.web.server import _auto_markers

    markers = _auto_markers(
        IndicatorValue.of(50.0), IndicatorValue.of(-5.0), None, TARGET
    )

    assert markers == []


# --- stock bars (chart data) API -------------------------------------------


def test_bars_route_returns_ohlcv_json(client):
    response = client.get("/stock/S5/bars")
    bars = response.json()

    assert response.status_code == 200
    assert len(bars) > 0
    first = bars[0]
    assert set(first) == {"time", "open", "high", "low", "close", "volume"}
    assert first["time"] == first["time"][:10]  # YYYY-MM-DD, no time component
    # Ascending by date, matching the price series order.
    assert [b["time"] for b in bars] == sorted(b["time"] for b in bars)


def test_bars_route_case_insensitive(client):
    assert client.get("/stock/s5/bars").json() == client.get("/stock/S5/bars").json()


def test_bars_route_for_unknown_symbol_returns_empty_list(client):
    response = client.get("/stock/NOSUCH/bars")

    assert response.status_code == 200
    assert response.json() == []


def test_bars_route_for_instrument_with_no_price_history(seeded, monkeypatch):
    """An instrument can exist without ever having ingested bars for it."""
    import algorix.web.server as web_app
    from algorix.models import Instrument, InstrumentType

    InstrumentRepository(seeded).upsert(
        Instrument(symbol="NOBARS", exchange=Exchange.NSE,
                   instrument_type=InstrumentType.EQUITY)
    )
    monkeypatch.setattr(web_app, "_current_session", lambda: TARGET)

    response = TestClient(app).get("/stock/NOBARS/bars")

    assert response.status_code == 200
    assert response.json() == []


# --- static assets -----------------------------------------------------------


def test_chart_drawings_js_is_served(client):
    response = client.get("/static/chart-drawings.js")

    assert response.status_code == 200
    assert "AlgorixDrawingTools" in response.text


# --- chart drawings API -----------------------------------------------------


def _instrument_id(db, symbol):
    return InstrumentRepository(db).get(symbol, Exchange.NSE).id


def test_list_drawings_returns_seeded_rows(client, seeded):
    repo = ChartDrawingRepository(seeded)
    repo.create(ChartDrawing(
        instrument_id=_instrument_id(seeded, "S5"), tool_type="trendline",
        points=[{"time": "2026-09-01", "price": 100.0},
                {"time": "2026-09-18", "price": 110.0}],
    ))
    repo.create(ChartDrawing(
        instrument_id=_instrument_id(seeded, "S5"), tool_type="breakout",
        points=[{"time": "2026-09-18", "price": 110.0}],
    ))

    bodies = client.get("/stock/S5/drawings").json()

    assert len(bodies) == 2
    assert {d["tool_type"] for d in bodies} == {"trendline", "breakout"}
    assert all(set(d) == {"id", "tool_type", "points", "created_at"} for d in bodies)


def test_list_drawings_when_none_stored_is_empty(client):
    assert client.get("/stock/S5/drawings").json() == []


def test_list_drawings_for_unknown_symbol_returns_empty_list(client):
    response = client.get("/stock/NOSUCH/drawings")

    assert response.status_code == 200
    assert response.json() == []


def test_list_drawings_case_insensitive(client, seeded):
    ChartDrawingRepository(seeded).create(ChartDrawing(
        instrument_id=_instrument_id(seeded, "S5"), tool_type="dip",
        points=[{"time": "2026-09-18", "price": 90.0}],
    ))

    assert client.get("/stock/s5/drawings").json() == client.get("/stock/S5/drawings").json()


def test_create_trendline_persists_and_is_listed(client):
    response = client.post("/stock/S5/drawings", json={
        "tool_type": "trendline",
        "points": [{"time": "2026-09-01", "price": 100.0},
                   {"time": "2026-09-18", "price": 110.0}],
    })

    assert response.status_code == 201
    body = response.json()
    assert body["tool_type"] == "trendline"
    assert body["id"] is not None

    listed = client.get("/stock/S5/drawings").json()
    assert [d["id"] for d in listed] == [body["id"]]


def test_create_drawing_for_unknown_symbol_is_rejected(client):
    response = client.post("/stock/NOSUCH/drawings", json={
        "tool_type": "trendline",
        "points": [{"time": "2026-09-01", "price": 100.0},
                   {"time": "2026-09-18", "price": 110.0}],
    })

    assert response.status_code == 404


def test_create_drawing_with_unregistered_tool_type_is_rejected(client):
    response = client.post("/stock/S5/drawings", json={
        "tool_type": "rectangle",
        "points": [{"time": "2026-09-01", "price": 100.0}],
    })

    assert response.status_code == 400


def test_create_trendline_with_wrong_point_count_is_rejected(client):
    too_few = client.post("/stock/S5/drawings", json={
        "tool_type": "trendline",
        "points": [{"time": "2026-09-01", "price": 100.0}],
    })
    too_many = client.post("/stock/S5/drawings", json={
        "tool_type": "trendline",
        "points": [{"time": "2026-09-01", "price": 100.0}] * 3,
    })

    assert too_few.status_code == 400
    assert too_many.status_code == 400


def test_create_drawing_with_missing_field_is_rejected(client):
    response = client.post("/stock/S5/drawings", json={"tool_type": "trendline"})

    assert response.status_code == 422


def test_create_drawing_with_non_numeric_price_is_rejected(client):
    response = client.post("/stock/S5/drawings", json={
        "tool_type": "breakout",
        "points": [{"time": "2026-09-18", "price": "abc"}],
    })

    assert response.status_code == 422


def test_delete_drawing_removes_it(client):
    created = client.post("/stock/S5/drawings", json={
        "tool_type": "breakout",
        "points": [{"time": "2026-09-18", "price": 110.0}],
    }).json()

    response = client.delete(f"/stock/S5/drawings/{created['id']}")

    assert response.status_code == 204
    assert client.get("/stock/S5/drawings").json() == []


def test_delete_drawing_for_unknown_symbol_is_rejected(client):
    response = client.delete("/stock/NOSUCH/drawings/1")

    assert response.status_code == 404


def test_delete_nonexistent_drawing_id_is_rejected(client):
    response = client.delete("/stock/S5/drawings/999999")

    assert response.status_code == 404


def test_delete_drawing_under_wrong_symbol_is_rejected(client):
    """A drawing id that is real, but belongs to a different instrument,
    must not be deletable by guessing the id under another symbol."""
    created = client.post("/stock/S5/drawings", json={
        "tool_type": "breakout",
        "points": [{"time": "2026-09-18", "price": 110.0}],
    }).json()

    response = client.delete(f"/stock/S6/drawings/{created['id']}")

    assert response.status_code == 404
    assert [d["id"] for d in client.get("/stock/S5/drawings").json()] == [created["id"]]


def test_deleting_the_same_drawing_twice_is_rejected_the_second_time(client):
    created = client.post("/stock/S5/drawings", json={
        "tool_type": "breakout",
        "points": [{"time": "2026-09-18", "price": 110.0}],
    }).json()

    first = client.delete(f"/stock/S5/drawings/{created['id']}")
    second = client.delete(f"/stock/S5/drawings/{created['id']}")

    assert first.status_code == 204
    assert second.status_code == 404


def test_delete_drawing_with_non_integer_id_is_rejected(client):
    response = client.delete("/stock/S5/drawings/abc")

    assert response.status_code == 422


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
