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

from dataclasses import replace as _dc_replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from algorix.backtest import evaluate_signals, run_backtest
from algorix.calendar import IST, TradingCalendar
from algorix.config import Config
from algorix.cross_sectional import compute_universe_momentum
from algorix.exceptions import AlgorixError, RefreshLockError
from algorix.indicators import (
    DELIVERY_BASELINE,
    PEAD_WINDOW_SESSIONS,
    stock_indicator_rows,
)
from algorix.journal import Journal
from algorix.metals import metal_snapshot
from algorix.models import ChartDrawing, Exchange
from algorix.qa import ask_about_stock
from algorix.refresh import last_refresh_started_at, refresh, refresh_lock
from algorix.regime import (
    INDIA_VIX,
    NIFTY_INDEX,
    FlowRepository,
    assess_regime,
)
from algorix.scan import SCAN_LOOKBACK_SESSIONS, run_scan
from algorix.scoring import SIGNAL_LABELS, score_universe
from algorix.sentiment import NotConfiguredError
from algorix.series import load_series
from algorix.storage import (
    ChartDrawingRepository,
    DeliveryRepository,
    Database,
    EarningsSurpriseRepository,
    InstrumentRepository,
    WatchlistRepository,
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
    if db_path is not None:
        # Colocate the lock file with the given database, matching how a
        # real Config's db_path already sits inside its own data_dir --
        # otherwise a test-supplied db_path would leave config.data_dir
        # pointing at the real environment's data directory, and
        # refresh_lock()/last_refresh_started_at() would read/write the
        # real ~/.algorix/refresh.lock instead of the test's own tmp path.
        config = _dc_replace(config, db_path=Path(db_path), data_dir=Path(db_path).parent)
    _state["db"] = Database(db_path or config.db_path)
    _state["calendar"] = TradingCalendar()
    _state["index"] = index_symbol
    _state["config"] = config


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


def _config() -> Config:
    if "config" not in _state:
        configure()
    return _state["config"]  # type: ignore[return-value]


#: UI-only throttle -- each manual run resubmits a G4 sentiment batch and
#: re-hits NSE/yfinance rate limits; this keeps a click-happy user from
#: multiplying that cost/rate-limit exposure past the once-daily cron
#: cadence. Not in Config: this is a UI behaviour, not a data-layer
#: setting, same reasoning as _BREAKOUT_DONCHIAN_THRESHOLD above.
SCAN_COOLDOWN_SECONDS = 45 * 60

#: In-process only -- lost on restart, which is fine: the actual safety
#: mechanism against overlap is refresh_lock's flock, not this dict. A
#: fresh process starting at "idle" is correct, since nothing is actually
#: still running after a restart.
_scan_state: dict[str, object] = {"status": "idle"}


def _run_pipeline(db_path) -> None:
    """Runs refresh() then run_scan() -- the identical whole-universe
    pipeline cron already runs, just invoked on demand.

    refresh()/run_scan() both *collect* per-instrument failures into
    `.errors` rather than raising (see their own docstrings) -- so
    surfacing a failure here means checking `.errors`, not only catching
    exceptions, or a data-source failure would finish "successfully" and
    violate CLAUDE.md's never-silently-swallow rule.
    """
    config = _config()
    db = Database(db_path)
    try:
        with refresh_lock(config.data_dir):
            refresh_report = refresh(db)
            scan_result = run_scan(db, deliver=True)
    except RefreshLockError as exc:
        _scan_state.update(
            status="error", error=str(exc),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        return
    except Exception as exc:  # truly unexpected -- must not vanish silently
        _scan_state.update(
            status="error",
            error=f"unexpected failure: {type(exc).__name__}: {exc}",
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        return

    errors = list(refresh_report.errors) + list(scan_result.errors)
    _scan_state.update(
        status="error" if errors else "success",
        error="; ".join(errors) if errors else None,
        finished_at=datetime.now(timezone.utc).isoformat(),
    )


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


def _score_current_session(as_of: date, index_symbol: str):
    """Load + cross-sectionally score the full universe for `as_of`.

    Shared by dashboard() and watchlist_view() -- a watchlist subset is
    always filtered out of this same scored universe, never scored in
    isolation. Scoring only the watchlisted symbols would compute
    different, incomparable percentiles than what the dashboard shows for
    the same stock (invariant 8: rank cross-sectionally, within one
    cohort) -- silently showing two different numbers for the same symbol
    under the same label would be exactly the kind of un-labelled method
    mismatch invariant 5 forbids.

    Returns None when there is no data at all (mirrors dashboard()'s own
    empty-database check).
    """
    calendar = _calendar()
    series, delivery, _, industry, earnings = _load_universe(as_of, index_symbol)
    if not series:
        return None
    momentum = compute_universe_momentum(series, as_of, calendar)
    universe = score_universe(
        series, momentum, as_of, delivery_by_symbol=delivery,
        calendar=calendar, universe_label=index_symbol,
        industry_by_symbol=industry, earnings_by_symbol=earnings,
    )
    return series, universe


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, top: int = 25):
    """Latest scan: regime banner plus the ranked table."""
    db, calendar = _db(), _calendar()
    index_symbol = _index()
    as_of = _current_session()

    loaded = _score_current_session(as_of, index_symbol)
    if loaded is None:
        return TEMPLATES.TemplateResponse(
            request=request,
            name="empty.html",
            context={"message": "No data yet. Run `python -m algorix.refresh` first."},
        )
    series, universe = loaded

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
            "signal_labels": SIGNAL_LABELS,
        },
    )


@app.post("/scan/run", status_code=202)
def trigger_scan(background_tasks: BackgroundTasks):
    """Kick off the same whole-universe refresh()+run_scan() pipeline cron
    runs at 07:30, on demand. Runs in the background -- refresh() alone
    makes ~57 sequential external HTTP calls, so blocking the request on
    it would be a poor UX and risks a client-side timeout.
    """
    if _scan_state.get("status") == "running":
        raise HTTPException(409, "a scan is already running")

    started = last_refresh_started_at(_config().data_dir)
    if started is not None:
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        if elapsed < SCAN_COOLDOWN_SECONDS:
            remaining = int(SCAN_COOLDOWN_SECONDS - elapsed)
            raise HTTPException(429, f"cooldown active -- try again in {remaining // 60}m")

    _scan_state.update(
        status="running", started_at=datetime.now(timezone.utc).isoformat(),
        finished_at=None, error=None,
    )
    background_tasks.add_task(_run_pipeline, _db().path)
    return {"status": "started"}


@app.get("/scan/status")
def scan_status():
    started = last_refresh_started_at(_config().data_dir)
    remaining = None
    if started is not None:
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        remaining = max(0, int(SCAN_COOLDOWN_SECONDS - elapsed))
    return {
        "status": _scan_state.get("status", "idle"),
        "started_at": _scan_state.get("started_at"),
        "finished_at": _scan_state.get("finished_at"),
        "error": _scan_state.get("error"),
        "last_run_started_at": started.isoformat() if started else None,
        "cooldown_remaining_seconds": remaining,
    }


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

    state, donchian_pos, short_return, raw_rows = stock_indicator_rows(
        series, records, calendar
    )
    indicators = [
        (code, label, _fmt(value, spec), unit)
        for code, label, value, unit, spec in raw_rows
    ]

    history = Journal(db).history_for(symbol, limit=60)
    latest = series.bars[-1] if series.bars else None
    auto_markers = (
        _auto_markers(donchian_pos, short_return, state, latest.session_date)
        if latest else []
    )
    on_watchlist = WatchlistRepository(db).contains(instrument.id)

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
            "on_watchlist": on_watchlist,
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


@app.post("/stock/{symbol}/watchlist")
def add_to_watchlist(symbol: str):
    """Add an instrument to the watchlist. Idempotent -- 200, not 201, since
    a repeat call isn't creating anything new (matches
    WatchlistRepository.add()'s own upsert-style idempotency)."""
    db = _db()
    instrument = InstrumentRepository(db).get(symbol.upper(), Exchange.NSE)
    if instrument is None or instrument.id is None:
        raise HTTPException(404, f"{symbol.upper()} is not in the database.")
    entry = WatchlistRepository(db).add(instrument.id)
    return {"symbol": symbol.upper(), "added_at": entry.added_at.isoformat()}


@app.delete("/stock/{symbol}/watchlist", status_code=204)
def remove_from_watchlist(symbol: str):
    db = _db()
    instrument = InstrumentRepository(db).get(symbol.upper(), Exchange.NSE)
    if instrument is None or instrument.id is None:
        raise HTTPException(404, f"{symbol.upper()} is not in the database.")
    if not WatchlistRepository(db).remove(instrument.id):
        raise HTTPException(404, f"{symbol.upper()} is not on the watchlist.")


class QuestionRequest(BaseModel):
    question: str


@app.post("/stock/{symbol}/ask")
def ask_stock_question(symbol: str, body: QuestionRequest):
    """One-shot AI Q&A about a stock -- see qa.py for the full design
    rationale (synchronous, one-shot, invariant 1/2 boundaries).

    Missing credentials degrade to 200 {"configured": false}, the same
    "opt-in feature, not a failure" shape NotConfiguredError already has
    for G4/Telegram elsewhere in this app. Anything else real (SDK
    failure, malformed LLM response) is a loud 502, never swallowed.
    """
    db = _db()
    symbol = symbol.upper()
    if not body.question.strip():
        raise HTTPException(422, "question cannot be blank")

    instrument = InstrumentRepository(db).get(symbol, Exchange.NSE)
    if instrument is None or instrument.id is None:
        raise HTTPException(404, f"{symbol} is not in the database.")

    try:
        query = ask_about_stock(db, instrument, body.question, calendar=_calendar())
    except NotConfiguredError as exc:
        return {"configured": False, "reason": str(exc)}
    except AlgorixError as exc:
        raise HTTPException(502, f"AI Q&A failed: {type(exc).__name__}: {exc}")

    return {
        "configured": True,
        "answer": query.answer,
        "model_id": query.model_id,
        "prompt_version": query.prompt_version,
        "asked_at": query.asked_at.isoformat(),
    }


def _watchlist_row(symbol, instrument, added_at, score) -> dict:
    """One /watchlist row. "Unavailable is shown, never hidden" applied to
    three distinct cases a watchlisted symbol can be in that universe.top()
    never has to handle: dropped from the tracked index, ineligible
    (circuit-lock/F&O-ban/etc, invariant 4), or genuinely unscored."""
    base = {"symbol": symbol, "name": instrument.name, "added_at": added_at}
    if score is None:
        return {
            **base, "score": None, "contributions": {},
            "reason": "not in the currently tracked universe, or no scan yet",
        }
    if not score.eligible:
        return {
            **base, "score": None, "contributions": {},
            "reason": "; ".join(score.ineligible_reasons) or "ineligible",
        }
    return {
        **base,
        "score": _fmt(score.score),
        "contributions": {c.code: c.percentile for c in score.contributions},
        "reason": None if score.score.available else score.score.reason,
    }


@app.get("/watchlist", response_class=HTMLResponse)
def watchlist_view(request: Request):
    """Watchlisted stocks, scored exactly like the dashboard -- see
    _score_current_session's docstring for why this never scores the
    watchlist subset in isolation."""
    db = _db()
    as_of = _current_session()
    instrument_repo = InstrumentRepository(db)

    entries = WatchlistRepository(db).list_all()
    watched = [
        (instrument.symbol, instrument, e.added_at)
        for e in entries
        if (instrument := instrument_repo.get_by_id(e.instrument_id)) is not None
    ]
    if not watched:
        return TEMPLATES.TemplateResponse(
            request=request, name="watchlist.html",
            context={"as_of": as_of, "rows": []},
        )

    loaded = _score_current_session(as_of, _index())
    universe = loaded[1] if loaded is not None else None

    rows = [
        _watchlist_row(symbol, instrument, added_at,
                       universe.scores.get(symbol) if universe else None)
        for symbol, instrument, added_at in watched
    ]

    return TEMPLATES.TemplateResponse(
        request=request, name="watchlist.html",
        context={"as_of": as_of, "rows": rows},
    )


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
            "labels": {**SIGNAL_LABELS, "A5": "pullback (inverted)"},
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
