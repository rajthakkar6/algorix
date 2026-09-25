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

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

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
from algorix.models import Exchange
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
    DeliveryRepository,
    Database,
    EarningsSurpriseRepository,
    InstrumentRepository,
)
from algorix.universe import NIFTY_50, ConstituencyRepository

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

app = FastAPI(title="Algorix")

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
    indicators = [
        ("A2", "52-week high proximity", _fmt(pct_of_52_week_high(series, calendar)), "%"),
        ("A4", "Donchian position", _fmt(donchian_position(series, calendar=calendar)), "%"),
        ("A5", "5-session return", _fmt(short_term_return(series, calendar=calendar)), "%"),
        ("A6", "delivery %", _fmt(latest_delivery_pct(records)), "%"),
        ("A6", "delivery trend", _fmt(delivery_trend(records), ".2f"), "x"),
        ("A7", "relative volume", _fmt(relative_volume(series, calendar=calendar), ".2f"), "x"),
        ("C1", "ATR", _fmt(atr_percent(series, calendar=calendar), ".2f"), "%"),
    ]

    closes = [b.close for b in series.window(120)] if series.bars else []
    history = Journal(db).history_for(symbol, limit=60)

    return TEMPLATES.TemplateResponse(
        request=request,
        name="stock.html",
        context={
            "symbol": symbol,
            "name": instrument.name,
            "as_of": as_of,
            "series": series,
            "closes": closes,
            "sparkline": _sparkline(closes),
            "trend": state,
            "indicators": indicators,
            "history": history,
            "latest": series.bars[-1] if series.bars else None,
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


def _sparkline(values: list[float], width: int = 560, height: int = 90) -> str:
    """Inline SVG polyline -- a chart without a charting dependency."""
    if len(values) < 2:
        return ""
    low, high = min(values), max(values)
    span = (high - low) or 1.0
    step = width / (len(values) - 1)
    points = " ".join(
        f"{i * step:.1f},{height - ((v - low) / span) * height:.1f}"
        for i, v in enumerate(values)
    )
    return points


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
