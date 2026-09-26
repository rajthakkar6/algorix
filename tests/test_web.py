"""Tests for the web UI.

Route-level tests against a seeded database: the UI must render real scores,
and -- the rule carried from the rest of the codebase -- must show missing
data as missing rather than as zero.
"""

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

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
    WatchlistRepository,
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


def test_dashboard_embeds_the_run_scan_control(client):
    text = client.get("/").text

    assert 'id="run-scan-btn"' in text
    assert 'id="run-scan-status"' in text
    assert "/scan/run" in text
    assert "/scan/status" in text


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


def test_stock_page_shows_add_to_watchlist_by_default(client):
    text = client.get("/stock/S5").text

    assert 'data-on="false"' in text
    assert "Add to watchlist" in text


def test_stock_page_shows_on_watchlist_state(client):
    client.post("/stock/S5/watchlist")

    text = client.get("/stock/S5").text

    assert 'data-on="true"' in text
    assert "On watchlist" in text


def test_stock_page_embeds_the_qa_panel(client):
    text = client.get("/stock/S5").text

    assert 'id="qa-input"' in text
    assert 'id="qa-ask"' in text
    assert "/stock/S5/ask" in text
    assert "AI-generated commentary, not the quant score" in text


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


# --- watchlist API -----------------------------------------------------------


def test_add_to_watchlist_persists(client, seeded):
    response = client.post("/stock/S5/watchlist")

    assert response.status_code == 200
    assert response.json()["symbol"] == "S5"
    instrument_id = _instrument_id(seeded, "S5")
    assert WatchlistRepository(seeded).contains(instrument_id) is True


def test_add_to_watchlist_is_idempotent(client):
    first = client.post("/stock/S5/watchlist")
    second = client.post("/stock/S5/watchlist")

    assert first.status_code == 200
    assert second.status_code == 200


def test_add_to_watchlist_unknown_symbol_is_rejected(client):
    assert client.post("/stock/NOSUCH/watchlist").status_code == 404


def test_remove_from_watchlist(client, seeded):
    client.post("/stock/S5/watchlist")

    response = client.delete("/stock/S5/watchlist")

    assert response.status_code == 204
    instrument_id = _instrument_id(seeded, "S5")
    assert WatchlistRepository(seeded).contains(instrument_id) is False


def test_remove_from_watchlist_unknown_symbol_is_rejected(client):
    assert client.delete("/stock/NOSUCH/watchlist").status_code == 404


def test_remove_from_watchlist_never_added_is_rejected(client):
    assert client.delete("/stock/S5/watchlist").status_code == 404


# --- AI Q&A API ----------------------------------------------------------------


def test_ask_stock_question_success(client, monkeypatch):
    import algorix.web.server as web_app

    fake_query = SimpleNamespace(
        answer="Delivery% has been trending down over the last week.",
        model_id="claude-sonnet-5", prompt_version=1,
        asked_at=datetime(2026, 9, 18, 10, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(web_app, "ask_about_stock", lambda *a, **kw: fake_query)

    response = client.post("/stock/S5/ask", json={"question": "Why the pullback?"})

    assert response.status_code == 200
    body = response.json()
    assert body["configured"] is True
    assert body["answer"] == fake_query.answer
    assert body["model_id"] == "claude-sonnet-5"
    assert body["prompt_version"] == 1


def test_ask_stock_question_unknown_symbol_is_rejected(client):
    response = client.post("/stock/NOSUCH/ask", json={"question": "Why?"})

    assert response.status_code == 404


def test_ask_stock_question_blank_is_rejected(client):
    response = client.post("/stock/S5/ask", json={"question": "   "})

    assert response.status_code == 422


def test_ask_stock_question_missing_field_is_rejected(client):
    response = client.post("/stock/S5/ask", json={})

    assert response.status_code == 422


def test_ask_stock_question_not_configured_is_reported_not_a_crash(client, monkeypatch):
    import algorix.web.server as web_app
    from algorix.sentiment import NotConfiguredError

    def _raise(*a, **kw):
        raise NotConfiguredError("ANTHROPIC_API_KEY is not set.")

    monkeypatch.setattr(web_app, "ask_about_stock", _raise)

    response = client.post("/stock/S5/ask", json={"question": "Why?"})

    assert response.status_code == 200
    body = response.json()
    assert body["configured"] is False
    assert "ANTHROPIC_API_KEY" in body["reason"]


def test_ask_stock_question_source_failure_is_a_loud_502(client, monkeypatch):
    import algorix.web.server as web_app
    from algorix.exceptions import SourceUnreachableError

    def _raise(*a, **kw):
        raise SourceUnreachableError("Anthropic call failed: timeout")

    monkeypatch.setattr(web_app, "ask_about_stock", _raise)

    response = client.post("/stock/S5/ask", json={"question": "Why?"})

    assert response.status_code == 502


# --- watchlist page -----------------------------------------------------------


def test_watchlist_page_shows_scored_symbol(client):
    client.post("/stock/S5/watchlist")

    text = client.get("/watchlist").text

    assert "S5" in text
    assert "score" in text.lower()


def test_watchlist_page_when_empty(client):
    text = client.get("/watchlist").text

    assert "Nothing on your watchlist yet" in text
    assert "Run `python -m algorix.refresh`" not in text  # not empty.html's message


def test_watchlist_page_for_symbol_outside_tracked_universe(client, seeded, monkeypatch):
    """A watchlisted instrument that isn't part of the currently-tracked
    index (or was removed from it) must be shown with a reason, not
    silently dropped from the page or given a fabricated score."""
    import algorix.web.server as web_app
    from algorix.models import Instrument, InstrumentType

    untracked_id = InstrumentRepository(seeded).upsert(
        Instrument(symbol="UNTRACKED", exchange=Exchange.NSE,
                   instrument_type=InstrumentType.EQUITY)
    )
    WatchlistRepository(seeded).add(untracked_id)
    monkeypatch.setattr(web_app, "_current_session", lambda: TARGET)

    text = TestClient(app).get("/watchlist").text

    assert "UNTRACKED" in text
    assert "not in the currently tracked universe" in text


def test_watchlist_row_shows_ineligible_reason_not_a_score():
    """Direct unit test of the row-shaping helper for the ineligible case --
    reproducing real ineligibility (liquidity/circuit-lock/F&O-ban/pledge
    gates) through the full scan pipeline is not worth the fragility this
    pure function can be tested against directly."""
    from datetime import date as _date

    from algorix.scoring import StockScore
    from algorix.series import IndicatorValue
    from algorix.web.server import _watchlist_row

    score = StockScore(
        symbol="S5", as_of=TARGET, score=IndicatorValue.unavailable("ineligible"),
        eligible=False, ineligible_reasons=["circuit-locked"],
    )

    row = _watchlist_row("S5", type("I", (), {"name": None})(), None, score)

    assert row["score"] is None
    assert row["reason"] == "circuit-locked"
    assert row["contributions"] == {}


# --- scan trigger API ---------------------------------------------------------


def test_trigger_scan_runs_pipeline_and_updates_status(client, monkeypatch):
    import algorix.web.server as web_app

    calls = []
    monkeypatch.setattr(
        web_app, "refresh",
        lambda db, **kw: calls.append("refresh") or SimpleNamespace(errors=[]),
    )
    monkeypatch.setattr(
        web_app, "run_scan",
        lambda db, **kw: calls.append("scan") or SimpleNamespace(errors=[]),
    )
    monkeypatch.setattr(web_app, "last_refresh_started_at", lambda data_dir: None)

    response = client.post("/scan/run")

    assert response.status_code == 202
    assert calls == ["refresh", "scan"]
    assert client.get("/scan/status").json()["status"] == "success"


def test_trigger_scan_surfaces_collected_errors_not_just_exceptions(client, monkeypatch):
    """refresh()/run_scan() collect per-instrument failures into `.errors`
    instead of raising -- a route that only catches exceptions would
    report "success" on a run that silently failed to fetch part of the
    universe, violating the never-silently-swallow rule."""
    import algorix.web.server as web_app

    monkeypatch.setattr(
        web_app, "refresh",
        lambda db, **kw: SimpleNamespace(errors=["delivery 2026-09-18: timeout"]),
    )
    monkeypatch.setattr(web_app, "run_scan", lambda db, **kw: SimpleNamespace(errors=[]))
    monkeypatch.setattr(web_app, "last_refresh_started_at", lambda data_dir: None)

    client.post("/scan/run")

    status = client.get("/scan/status").json()
    assert status["status"] == "error"
    assert "timeout" in status["error"]


def test_trigger_scan_during_cooldown_is_rejected(client, monkeypatch):
    import algorix.web.server as web_app

    monkeypatch.setattr(
        web_app, "last_refresh_started_at",
        lambda data_dir: datetime.now(timezone.utc),
    )

    assert client.post("/scan/run").status_code == 429


def test_trigger_scan_when_lock_held_elsewhere_reports_error(client, monkeypatch):
    """Hold a raw OS lock on the same file refresh_lock() uses, without
    going through refresh_lock() itself -- that would also stamp a fresh
    "started at" timestamp and trip the cooldown check before the
    lock-contention path this test targets is ever reached."""
    import fcntl

    import algorix.web.server as web_app
    from algorix.refresh import REFRESH_LOCK_FILENAME

    monkeypatch.setattr(web_app, "last_refresh_started_at", lambda data_dir: None)

    data_dir = web_app._config().data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    fd = open(data_dir / REFRESH_LOCK_FILENAME, "a+")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        client.post("/scan/run")  # background task hits the held lock
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()

    status = client.get("/scan/status").json()
    assert status["status"] == "error"
    assert "already running" in status["error"]


def test_scan_status_shape_when_idle(client, monkeypatch):
    import algorix.web.server as web_app

    monkeypatch.setattr(web_app, "_scan_state", {"status": "idle"})
    monkeypatch.setattr(web_app, "last_refresh_started_at", lambda data_dir: None)

    body = client.get("/scan/status").json()

    assert body["status"] == "idle"
    assert set(body) == {
        "status", "started_at", "finished_at", "error",
        "last_run_started_at", "cooldown_remaining_seconds",
    }


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
