"""Local web UI.

Deliberately boring technology: FastAPI serving server-rendered Jinja
templates, reading the same SQLite file the scanner writes. No build step, no
npm, no client framework. A single-user tool reading a local database gains
nothing from a JS toolchain, and the pages are simple enough that HTML the
server already has is faster than HTML the browser has to assemble.

The one design rule carried over from the rest of the codebase: **unavailable
is shown, never hidden.** A missing indicator renders as a dash with its
reason on hover, not as a zero or a blank -- the whole system is built on the
distinction, and a UI that quietly smooths it over would undo that.

Run with:  python -m algorix.web
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from algorix.backtest import evaluate_signals, run_backtest
from algorix.calendar import IST, TradingCalendar
from algorix.config import Config
from algorix.cross_sectional import compute_universe_momentum
from algorix.indicators import (
    DELIVERY_BASELINE,
    PEAD_WINDOW_SESSIONS,
    atr_percent,
    delivery_trend,
    donchian_position,
    latest_delivery_pct,
    pct_of_52_week_high,
    relative_volume,
    short_term_return,
    trend_state,
)
from algorix.journal import Journal
from algorix.metals import metal_snapshot
from algorix.models import ChartDrawing, Exchange
from algorix.regime import (
    INDIA_VIX,
    NIFTY_INDEX,
    FlowRepository,
    assess_regime,
)
from algorix.scan import SCAN_LOOKBACK_SESSIONS
from algorix.scoring import score_universe
from algorix.series import load_series
from algorix.storage import (
    ChartDrawingRepository,
    DeliveryRepository,
    Database,
    EarningsSurpriseRepository,
    InstrumentRepository,
)
from algorix.universe import NIFTY_50, ConstituencyRepository

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

app = FastAPI(title="Algorix")
# First static mount in the app -- holds chart-drawings.js, the ported
# lightweight-charts primitive for trendlines/markers. Kept out of the
# templates' inline <script> because it's ~150 lines of self-contained
# canvas code with zero Jinja dependency.
app.mount(
    "/static",
    StaticFiles(directory=str(Path(__file__).parent / "static")),
    name="static",
)

_state: dict[str, object] = {}


def configure(db_path: str | Path | None = None, index_symbol: str = NIFTY_50) -> None:
    config = Config.from_env()
    _state["db"] = Database(db_path or config.db_path)
    _state["calendar"] = TradingCalendar()
    _state["index"] = index_symbol


def _db() -> Database:
    if "db" not in _state:
        configure()
    return _state["db"]  # type: ignore[return-value]


def _calendar() -> TradingCalendar:
    if "calendar" not in _state:
        configure()
    return _state["calendar"]  # type: ignore[return-value]


def _index() -> str:
    if "index" not in _state:
        configure()
    return str(_state["index"])


def _fmt(value, spec: str = ".1f") -> dict:
    """Render an IndicatorValue for a template: value, or a dash plus reason."""
    if value is None:
        return {"text": "--", "title": "not computed", "available": False}
    if getattr(value, "available", False):
        return {
            "text": format(value.value, spec),
            "title": "",
            "available": True,
            "raw": value.value,
        }
    return {
        "text": "--",
        "title": value.reason or "unavailable",
        "available": False,
    }


# UI-only heuristic thresholds for the stock chart's auto-highlight markers
# -- never read by scoring.py. A first pass, easy to retune; not scoring
# rules, just "is this worth a visual flag on the chart right now."
_BREAKOUT_DONCHIAN_THRESHOLD = 95.0  # top of the N-session Donchian channel
_DIP_PULLBACK_THRESHOLD = -3.0       # % short-term return, while in an uptrend


def _auto_markers(donchian_pos, short_return, trend, latest_date) -> list[dict]:
    """System-detected breakout/dip highlights for the chart.

    Computed fresh on every page load from indicators this page already
    displays (A4 Donchian position, A5 short-term return, A3 trend state)
    -- never persisted, so they can't go stale and need no sync logic. Only
    the latest session is evaluated, because that is the only session these
    indicators are computed for here; a full historical breakout/dip series
    would need rolling recomputation across the whole loaded window, which
    is out of scope for a chart overlay.

    Distinct from ChartDrawing: a user can also *manually* place a
    breakout/dip marker (persisted, tool_type "breakout"/"dip"), but that is
    a different code path (create_drawing) -- this function only ever
    produces the automatic, unsaved kind.
    """
    markers = []
    if donchian_pos.available and donchian_pos.value >= _BREAKOUT_DONCHIAN_THRESHOLD:
        markers.append({
            "time": latest_date.isoformat(), "position": "belowBar",
            "shape": "arrowUp", "color": "--good", "text": "Breakout",
        })
    if (
        trend is not None and trend.is_uptrend
        and short_return.available and short_return.value <= _DIP_PULLBACK_THRESHOLD
    ):
        markers.append({
            "time": latest_date.isoformat(), "position": "aboveBar",
            "shape": "arrowDown", "color": "--warn", "text": "Dip",
        })
    return markers


def _load_universe(as_of: date, index_symbol: str):
    db, calendar = _db(), _calendar()
    instrument_repo = InstrumentRepository(db)
    delivery_repo = DeliveryRepository(db)
    earnings_repo = EarningsSurpriseRepository(db)

    symbols = ConstituencyRepository(db).current_constituents(
        index_symbol, on=as_of
    )
    series, delivery, ids, industry, earnings = {}, {}, {}, {}, {}
    delivery_start = calendar.trading_days_ago(as_of, DELIVERY_BASELINE * 2)
    earnings_start = calendar.trading_days_ago(as_of, PEAD_WINDOW_SESSIONS)

    for symbol in symbols:
        instrument = instrument_repo.get(symbol, Exchange.NSE)
        if instrument is None or instrument.id is None:
            continue
        ids[symbol] = instrument.id
        series[symbol] = load_series(
            db, instrument.id, as_of, SCAN_LOOKBACK_SESSIONS, calendar
        )
        delivery[symbol] = delivery_repo.get_range(
            instrument.id, delivery_start, as_of
        )
        industry[symbol] = instrument.industry
        earnings[symbol] = earnings_repo.get_range(
            instrument.id, earnings_start, as_of
        )
    return series, delivery, ids, industry, earnings


def _current_session() -> date:
    return _calendar().last_completed_session(datetime.now(IST))


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, top: int = 25):
    """Latest scan: regime banner plus the ranked table."""
    db, calendar = _db(), _calendar()
    index_symbol = _index()
    as_of = _current_session()

    series, delivery, _, industry, earnings = _load_universe(as_of, index_symbol)
    if not series:
        return TEMPLATES.TemplateResponse(
            request=request,
            name="empty.html",
            context={"message": "No data yet. Run `python -m algorix.refresh` first."},
        )

    momentum = compute_universe_momentum(series, as_of, calendar)
    universe = score_universe(
        series, momentum, as_of, delivery_by_symbol=delivery,
        calendar=calendar, universe_label=index_symbol,
        industry_by_symbol=industry, earnings_by_symbol=earnings,
    )

    instrument_repo = InstrumentRepository(db)

    def regime_series(instrument):
        stored = instrument_repo.get(instrument.symbol, instrument.exchange)
        if stored is None or stored.id is None:
            return None
        loaded = load_series(db, stored.id, as_of, 300, calendar)
        return loaded if loaded.bars else None

    regime = assess_regime(
        as_of,
        series,
        index_series=regime_series(NIFTY_INDEX),
        vix_series=regime_series(INDIA_VIX),
        flows=FlowRepository(db).get_for(as_of),
        calendar=calendar,
    )

    rows = []
    for rank, score in enumerate(universe.top(top), start=1):
        contributions = {c.code: c.percentile for c in score.contributions}
        rows.append(
            {
                "rank": rank,
                "symbol": score.symbol,
                "score": score.score.value,
                "contributions": contributions,
                "missing": score.missing,
                "stop": _fmt(score.risk.stop_price) if score.risk else None,
                "atr_pct": _fmt(score.risk.atr_pct, ".2f") if score.risk else None,
                "close": series[score.symbol].bars[-1].close
                if series[score.symbol].bars
                else None,
            }
        )

    metals = None
    try:
        snapshot = metal_snapshot(db, as_of)
        metals = {
            "gold": snapshot.gold_usd,
            "silver": snapshot.silver_usd,
            "ratio": snapshot.gold_silver_ratio,
            "usdinr": snapshot.usd_inr,
        }
    except Exception:
        metals = None

    return TEMPLATES.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "as_of": as_of,
            "index": index_symbol,
            "regime": regime,
            "rows": rows,
            "universe": universe,
            "metals": metals,
            "signal_labels": {
                "A1": "momentum",
                "A2": "52w high",
                "A3": "trend",
                "A4": "breakout",
                "A5": "pullback",
                "A6": "delivery",
            },
        },
    )


@app.get("/stock/{symbol}", response_class=HTMLResponse)
def stock_detail(request: Request, symbol: str):
    """One instrument: every indicator, the price path, and journal history."""
    db, calendar = _db(), _calendar()
    symbol = symbol.upper()
    as_of = _current_session()

    instrument = InstrumentRepository(db).get(symbol, Exchange.NSE)
    if instrument is None or instrument.id is None:
        return TEMPLATES.TemplateResponse(
            request=request, name="empty.html",
            context={"message": f"{symbol} is not in the database."},
        )

    series = load_series(db, instrument.id, as_of, SCAN_LOOKBACK_SESSIONS, calendar)
    records = DeliveryRepository(db).get_range(
        instrument.id, calendar.trading_days_ago(as_of, DELIVERY_BASELINE * 2), as_of
    )

    state = trend_state(series, calendar)
    donchian_pos = donchian_position(series, calendar=calendar)
    short_return = short_term_return(series, calendar=calendar)
    indicators = [
        ("A2", "52-week high proximity", _fmt(pct_of_52_week_high(series, calendar)), "%"),
        ("A4", "Donchian position", _fmt(donchian_pos), "%"),
        ("A5", "5-session return", _fmt(short_return), "%"),
        ("A6", "delivery %", _fmt(latest_delivery_pct(records)), "%"),
        ("A6", "delivery trend", _fmt(delivery_trend(records), ".2f"), "x"),
        ("A7", "relative volume", _fmt(relative_volume(series, calendar=calendar), ".2f"), "x"),
        ("C1", "ATR", _fmt(atr_percent(series, calendar=calendar), ".2f"), "%"),
    ]

    history = Journal(db).history_for(symbol, limit=60)
    latest = series.bars[-1] if series.bars else None
    auto_markers = (
        _auto_markers(donchian_pos, short_return, state, latest.session_date)
        if latest else []
    )

    return TEMPLATES.TemplateResponse(
        request=request,
        name="stock.html",
        context={
            "symbol": symbol,
            "name": instrument.name,
            "as_of": as_of,
            "series": series,
            "trend": state,
            "indicators": indicators,
            "history": history,
            "latest": latest,
            "auto_markers": auto_markers,
        },
    )


@app.get("/stock/{symbol}/bars")
def stock_bars(symbol: str):
    """OHLCV for the interactive chart on the stock detail page.

    The only JSON route in this app -- every other page is server-rendered
    (see this module's own docstring on why). A pannable, zoomable candle
    chart is the one thing server-rendered SVG genuinely cannot do well;
    everything else on the page stays server-rendered. `time` is the
    session date as `YYYY-MM-DD`, the format lightweight-charts' business-
    day mode expects directly, no client-side date parsing needed.
    """
    db, calendar = _db(), _calendar()
    symbol = symbol.upper()
    as_of = _current_session()

    instrument = InstrumentRepository(db).get(symbol, Exchange.NSE)
    if instrument is None or instrument.id is None:
        return []

    series = load_series(db, instrument.id, as_of, SCAN_LOOKBACK_SESSIONS, calendar)
    return [
        {
            "time": b.session_date.isoformat(),
            "open": b.open, "high": b.high, "low": b.low, "close": b.close,
            "volume": b.volume,
        }
        for b in series.bars
    ]


class DrawingPoint(BaseModel):
    time: str
    price: float


class DrawingCreate(BaseModel):
    tool_type: str
    points: list[DrawingPoint]


# Known drawing tools and how many points each needs. Adding a future tool
# (horizontal line, rectangle, fib retracement) is one entry here plus a
# client-side renderer -- never a schema change, never a new route.
_DRAWING_POINT_COUNTS: dict[str, int] = {"trendline": 2, "breakout": 1, "dip": 1}


def _serialize_drawing(d: ChartDrawing) -> dict:
    return {
        "id": d.id,
        "tool_type": d.tool_type,
        "points": d.points,
        "created_at": d.created_at.isoformat() if d.created_at else None,
    }


@app.get("/stock/{symbol}/drawings")
def list_drawings(symbol: str):
    """Persisted chart annotations for this instrument (user-drawn
    trendlines, manually-placed breakout/dip markers).

    Same convention as /bars: an unknown symbol is informational-empty
    (200 []), not an error -- "no drawings exist" and "the symbol doesn't
    exist" both mean the chart overlays nothing, and the page itself
    already empty-states an unknown symbol before any client script runs.
    """
    db = _db()
    instrument = InstrumentRepository(db).get(symbol.upper(), Exchange.NSE)
    if instrument is None or instrument.id is None:
        return []
    drawings = ChartDrawingRepository(db).list_for(instrument.id)
    return [_serialize_drawing(d) for d in drawings]


@app.post("/stock/{symbol}/drawings", status_code=201)
def create_drawing(symbol: str, body: DrawingCreate):
    """Persist one drawn annotation.

    Unlike the GET route, an unknown symbol here IS a real error: creating
    a drawing means attaching it to a specific instrument, and there is
    nothing to attach it to -- a write against a nonexistent resource is
    rejected (404), not silently accepted or silently dropped.
    """
    db = _db()
    instrument = InstrumentRepository(db).get(symbol.upper(), Exchange.NSE)
    if instrument is None or instrument.id is None:
        raise HTTPException(404, f"{symbol.upper()} is not in the database.")

    expected = _DRAWING_POINT_COUNTS.get(body.tool_type)
    if expected is None:
        raise HTTPException(400, f"unknown tool_type {body.tool_type!r}")
    if len(body.points) != expected:
        raise HTTPException(
            400,
            f"{body.tool_type} requires exactly {expected} point(s), "
            f"got {len(body.points)}",
        )

    drawing = ChartDrawing(
        instrument_id=instrument.id,
        tool_type=body.tool_type,
        points=[{"time": p.time, "price": p.price} for p in body.points],
    )
    created = ChartDrawingRepository(db).create(drawing)
    return _serialize_drawing(created)


@app.delete("/stock/{symbol}/drawings/{drawing_id}", status_code=204)
def delete_drawing(symbol: str, drawing_id: int):
    """Delete one annotation, scoped to its instrument.

    Unknown symbol, unknown drawing id, and a real drawing id that belongs
    to a *different* instrument all 404 -- from the caller's perspective
    all three mean "there is no such drawing under this symbol." The
    instrument-scoped delete in ChartDrawingRepository.delete() makes the
    cross-instrument case structurally impossible, not just checked for.
    """
    db = _db()
    instrument = InstrumentRepository(db).get(symbol.upper(), Exchange.NSE)
    if instrument is None or instrument.id is None:
        raise HTTPException(404, f"{symbol.upper()} is not in the database.")
    if not ChartDrawingRepository(db).delete(drawing_id, instrument.id):
        raise HTTPException(404, f"drawing {drawing_id} not found for {symbol.upper()}")


@app.get("/backtest", response_class=HTMLResponse)
def backtest_view(request: Request, sessions: int = 250, horizon: int = 20):
    """Replay results -- does the score actually predict returns?"""
    db, calendar = _db(), _calendar()
    end = _current_session()
    start = calendar.trading_days_ago(end, sessions)

    result = run_backtest(db, start, end, calendar, index_symbol=_index())
    signals = evaluate_signals(result.outcomes, horizon) if result.outcomes else {}

    return TEMPLATES.TemplateResponse(
        request=request,
        name="backtest.html",
        context={
            "result": result,
            "signals": sorted(
                signals.items(),
                key=lambda kv: kv[1].mean_ic if kv[1].mean_ic is not None else -99,
                reverse=True,
            ),
            "horizon": horizon,
            "sessions": sessions,
            "labels": {
                "A1": "momentum",
                "A2": "52w high",
                "A3": "trend",
                "A4": "breakout",
                "A5": "pullback (inverted)",
                "A6": "delivery",
            },
        },
    )


@app.get("/journal", response_class=HTMLResponse)
def journal_view(request: Request):
    """Recorded scans -- what the scanner actually said, and when."""
    db = _db()
    journal = Journal(db)
    sessions = list(reversed(journal.recorded_sessions()))

    latest = journal.scores_for(sessions[0]) if sessions else []

    return TEMPLATES.TemplateResponse(
        request=request,
        name="journal.html",
        context={
            "sessions": sessions,
            "entries": latest[:50],
            "latest_session": sessions[0] if sessions else None,
        },
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    import uvicorn

    from algorix.config import load_dotenv_if_present

    load_dotenv_if_present()

    parser = argparse.ArgumentParser(description="Serve the Algorix web UI.")
    parser.add_argument("--db", help="database path (overrides config)")
    parser.add_argument("--index", default=NIFTY_50, help="index to display")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    configure(args.db, args.index.upper())
    print(f"Algorix UI on http://{args.host}:{args.port}  (index: {args.index})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0
