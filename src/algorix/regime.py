"""Market regime gates (INDICATORS.md Bucket B).

These do not rank stocks. They answer a prior question: *is this a market
where the scores are worth acting on at all?*

That question is not academic here. An 18-year NSE backtest of the Nifty 200
Momentum 30 index beat the Nifty 50 (14.01% vs 10.42% CAGR) while passing
through a **-70.5% drawdown with a 65-month recovery**, and through much of
2025-26 Indian momentum sat flat-to-negative while global momentum was the
best-performing factor anywhere. A scanner that ranks momentum without asking
whether momentum is working will hand you its most confident picks directly
into a drawdown. The gate is what separates this from a screener.

Four independent readings, deliberately not collapsed into one number:

- **B1 breadth** -- share of the universe above its 50 DMA. Narrow breadth is
  the classic precondition for a momentum unwind.
- **B2 India VIX** -- level and percentile. High-volatility regimes are where
  momentum historically breaks.
- **B3 efficiency ratio** (Kaufman) -- trending versus chopping. Chosen over
  ADX: same property, less lag.
- **B4 FII/DII net flow** -- institutional direction, which moves Indian
  markets regardless of individual stock quality.

Per PROJECT_SCOPE open decision 4, a hostile regime **warns rather than
suppresses**: silence is ambiguous, a warning is information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import StrEnum

import requests

from algorix.calendar import TradingCalendar
from algorix.exceptions import (
    DataIntegrityError,
    DataUnavailableError,
    SourceUnreachableError,
)
from algorix.indicators import TREND_FAST, simple_moving_average
from algorix.models import CalendarPolicy, Exchange, Instrument, InstrumentType
from algorix.series import IndicatorValue, PriceSeries
from algorix.storage import BarRepository, Database, InstrumentRepository

SOURCE_NSE_FLOWS = "nse-fiidii"

NIFTY_INDEX = Instrument(
    symbol="NIFTY50",
    exchange=Exchange.NSE,
    instrument_type=InstrumentType.INDEX,
    name="NIFTY 50 Index",
    yahoo_symbol="^NSEI",
    calendar_policy=CalendarPolicy.NSE,
)

INDIA_VIX = Instrument(
    symbol="INDIAVIX",
    exchange=Exchange.NSE,
    instrument_type=InstrumentType.INDEX,
    name="India VIX",
    yahoo_symbol="^INDIAVIX",
    calendar_policy=CalendarPolicy.NSE,
)

REGIME_INSTRUMENTS: tuple[Instrument, ...] = (NIFTY_INDEX, INDIA_VIX)

#: B1: below this share above the 50 DMA, the advance is too narrow to trust.
BREADTH_WEAK = 30.0
BREADTH_STRONG = 55.0

#: B2: VIX percentile above this marks an elevated-risk regime.
VIX_ELEVATED_PERCENTILE = 80.0
VIX_PERCENTILE_WINDOW = 252

#: B3: efficiency ratio below this is a chopping market, where momentum
#: signals whipsaw rather than persist.
EFFICIENCY_CHOPPY = 0.30
EFFICIENCY_TRENDING = 0.50
EFFICIENCY_WINDOW = 20


class RegimeVerdict(StrEnum):
    FAVOURABLE = "FAVOURABLE"
    MIXED = "MIXED"
    HOSTILE = "HOSTILE"
    UNKNOWN = "UNKNOWN"


def seed_regime_instruments(db: Database) -> dict[str, int]:
    repository = InstrumentRepository(db)
    return {i.symbol: repository.upsert(i) for i in REGIME_INSTRUMENTS}


# ---------------------------------------------------------------------------
# B1 -- market breadth
# ---------------------------------------------------------------------------


def market_breadth(
    series_by_symbol: dict[str, PriceSeries],
    window: int = TREND_FAST,
    min_cohort: int = 10,
) -> IndicatorValue:
    """Percentage of the universe trading above its `window`-session average.

    Instruments without enough history are excluded from both numerator and
    denominator -- counting them as "below" would fake a breadth collapse
    every time the universe gained a recent listing.
    """
    above = 0
    counted = 0

    for series in series_by_symbol.values():
        average = simple_moving_average(series, window)
        if average is None or average <= 0 or not series.bars:
            continue
        counted += 1
        if series.bars[-1].close > average:
            above += 1

    if counted < min_cohort:
        return IndicatorValue.unavailable(
            f"breadth: only {counted} instruments have {window} sessions "
            f"(needs {min_cohort})"
        )

    return IndicatorValue.of((above / counted) * 100.0)


# ---------------------------------------------------------------------------
# B3 -- Kaufman efficiency ratio
# ---------------------------------------------------------------------------


def efficiency_ratio(
    series: PriceSeries,
    window: int = EFFICIENCY_WINDOW,
    calendar: TradingCalendar | None = None,
) -> IndicatorValue:
    """Net directional movement divided by total movement, 0-1.

    1.0 is a straight line; near 0 is a market covering the same ground
    repeatedly. This is the single condition determining whether momentum
    signals persist or whipsaw.
    """
    needed = window + 1
    problem = series.require(needed, calendar)
    if problem:
        return IndicatorValue.unavailable(f"efficiency ratio: {problem}")

    closes = [b.close for b in series.window(needed)]
    net = abs(closes[-1] - closes[0])
    total = sum(abs(b - a) for a, b in zip(closes, closes[1:]))

    if total <= 0:
        return IndicatorValue.unavailable(
            "efficiency ratio: no movement in window"
        )

    return IndicatorValue.of(net / total)


# ---------------------------------------------------------------------------
# B2 -- VIX percentile
# ---------------------------------------------------------------------------


def percentile_of_latest(
    series: PriceSeries, window: int = VIX_PERCENTILE_WINDOW
) -> IndicatorValue:
    """Where the latest close sits within its own trailing `window`, 0-100."""
    problem = series.require(window)
    if problem:
        return IndicatorValue.unavailable(f"percentile: {problem}")

    closes = [b.close for b in series.window(window)]
    current = closes[-1]
    history = closes[:-1]
    below = sum(1 for c in history if c < current)
    return IndicatorValue.of((below / len(history)) * 100.0)


# ---------------------------------------------------------------------------
# B4 -- FII/DII flows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlowRecord:
    """One institutional category's activity for a session, in Rs crore."""

    session_date: date
    category: str
    buy_value: float
    sell_value: float
    net_value: float


_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

FII_DII_URL = "https://www.nseindia.com/api/fiidiiTradeReact"


def parse_flows(payload: object) -> list[FlowRecord]:
    """Parse NSE's FII/DII payload.

    Values are strings in the response and occasionally '-' when a category
    did not report; those rows are skipped rather than read as zero, since
    zero flow is a real and different claim.
    """
    if not isinstance(payload, list) or not payload:
        raise DataUnavailableError("FII/DII payload is empty or not a list")

    records: list[FlowRecord] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        raw_date = str(entry.get("date", "")).strip()
        session = _parse_flow_date(raw_date)
        if session is None:
            raise DataIntegrityError(
                f"FII/DII row has an unparseable date: {raw_date!r}"
            )

        try:
            buy = float(str(entry.get("buyValue", "")).replace(",", ""))
            sell = float(str(entry.get("sellValue", "")).replace(",", ""))
            net = float(str(entry.get("netValue", "")).replace(",", ""))
        except ValueError:
            continue

        records.append(
            FlowRecord(
                session_date=session,
                category=str(entry.get("category", "")).strip().upper(),
                buy_value=buy,
                sell_value=sell,
                net_value=net,
            )
        )

    if not records:
        raise DataUnavailableError("FII/DII payload contained no usable rows")
    return records


def _parse_flow_date(raw: str) -> date | None:
    for fmt in ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


class NseFlowClient:
    """Fetches FII/DII flows. NSE requires a primed session cookie."""

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        self.timeout_seconds = timeout_seconds

    def fetch(self) -> list[FlowRecord]:
        session = requests.Session()
        try:
            # The API rejects requests without a cookie from the main site.
            session.get(
                "https://www.nseindia.com",
                headers=_NSE_HEADERS,
                timeout=self.timeout_seconds,
            )
            response = session.get(
                FII_DII_URL, headers=_NSE_HEADERS, timeout=self.timeout_seconds
            )
        except requests.RequestException as exc:
            raise SourceUnreachableError(
                f"Could not reach NSE FII/DII endpoint: {exc}"
            ) from exc

        if response.status_code != 200:
            raise SourceUnreachableError(
                f"NSE FII/DII returned HTTP {response.status_code}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise SourceUnreachableError(
                "NSE FII/DII returned non-JSON (likely a block page)"
            ) from exc

        return parse_flows(payload)


class FlowRepository:
    """Storage for market-level institutional flows."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def upsert_many(self, records: list[FlowRecord], source: str) -> int:
        if not records:
            return 0
        rows = [
            (
                r.session_date.isoformat(),
                r.category,
                r.buy_value,
                r.sell_value,
                r.net_value,
                source,
                datetime.now(timezone.utc).isoformat(),
            )
            for r in records
        ]
        with self.db.connect() as conn:
            conn.executemany(
                """
                INSERT INTO market_flows
                    (session_date, category, buy_value, sell_value, net_value,
                     source, ingested_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (session_date, category) DO UPDATE SET
                    buy_value   = excluded.buy_value,
                    sell_value  = excluded.sell_value,
                    net_value   = excluded.net_value,
                    source      = excluded.source,
                    ingested_at = excluded.ingested_at
                """,
                rows,
            )
        return len(rows)

    def get_for(self, session_date: date) -> dict[str, FlowRecord]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM market_flows WHERE session_date = ?",
                (session_date.isoformat(),),
            ).fetchall()
        return {
            r["category"]: FlowRecord(
                session_date=date.fromisoformat(r["session_date"]),
                category=r["category"],
                buy_value=r["buy_value"],
                sell_value=r["sell_value"],
                net_value=r["net_value"],
            )
            for r in rows
        }


def _flow_net(flows: dict[str, FlowRecord], *names: str) -> IndicatorValue:
    for name in names:
        if name in flows:
            return IndicatorValue.of(flows[name].net_value)
    return IndicatorValue.unavailable(
        f"no flow data for {names[0]} on this session"
    )


# ---------------------------------------------------------------------------
# Composite regime assessment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketRegime:
    """The four gates for one session, plus an overall verdict."""

    as_of: date
    breadth_pct: IndicatorValue
    vix_level: IndicatorValue
    vix_percentile: IndicatorValue
    efficiency_ratio: IndicatorValue
    fii_net: IndicatorValue
    dii_net: IndicatorValue
    warnings: list[str] = field(default_factory=list)
    positives: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> RegimeVerdict:
        """Overall stance.

        Gates are counted, not averaged: two independent warnings are a
        materially different situation from one, and averaging would let a
        strong reading mask a genuine danger signal.
        """
        known = [
            v
            for v in (
                self.breadth_pct,
                self.vix_percentile,
                self.efficiency_ratio,
            )
            if v.available
        ]
        if not known:
            return RegimeVerdict.UNKNOWN
        if len(self.warnings) >= 2:
            return RegimeVerdict.HOSTILE
        if self.warnings:
            return RegimeVerdict.MIXED
        return RegimeVerdict.FAVOURABLE

    @property
    def is_favourable(self) -> bool:
        return self.verdict is RegimeVerdict.FAVOURABLE

    def summary(self) -> str:
        parts = [f"Regime: {self.verdict}"]
        if self.breadth_pct.available:
            parts.append(f"breadth {self.breadth_pct.value:.0f}%")
        if self.vix_level.available:
            vix = f"VIX {self.vix_level.value:.1f}"
            if self.vix_percentile.available:
                vix += f" ({self.vix_percentile.value:.0f}th pct)"
            parts.append(vix)
        if self.efficiency_ratio.available:
            parts.append(f"efficiency {self.efficiency_ratio.value:.2f}")
        if self.fii_net.available:
            parts.append(f"FII {self.fii_net.value:+,.0f} cr")
        if self.dii_net.available:
            parts.append(f"DII {self.dii_net.value:+,.0f} cr")
        return " | ".join(parts)


def assess_regime(
    as_of: date,
    universe_series: dict[str, PriceSeries],
    index_series: PriceSeries | None = None,
    vix_series: PriceSeries | None = None,
    flows: dict[str, FlowRecord] | None = None,
    calendar: TradingCalendar | None = None,
) -> MarketRegime:
    """Evaluate all four gates and produce a verdict with reasons."""
    warnings: list[str] = []
    positives: list[str] = []

    breadth = market_breadth(universe_series)
    if breadth.available:
        if breadth.value < BREADTH_WEAK:
            warnings.append(
                f"narrow breadth: only {breadth.value:.0f}% of the universe is "
                "above its 50 DMA -- the classic precondition for a momentum unwind"
            )
        elif breadth.value >= BREADTH_STRONG:
            positives.append(f"broad participation ({breadth.value:.0f}% above 50 DMA)")

    vix_level = IndicatorValue.unavailable("no VIX series")
    vix_pct = IndicatorValue.unavailable("no VIX series")
    if vix_series is not None and vix_series.bars:
        vix_level = IndicatorValue.of(vix_series.bars[-1].close)
        vix_pct = percentile_of_latest(vix_series)
        if vix_pct.available and vix_pct.value >= VIX_ELEVATED_PERCENTILE:
            warnings.append(
                f"elevated volatility: India VIX at {vix_level.value:.1f} is in "
                f"the {vix_pct.value:.0f}th percentile of its past year"
            )

    efficiency = IndicatorValue.unavailable("no index series")
    if index_series is not None:
        efficiency = efficiency_ratio(index_series, calendar=calendar)
        if efficiency.available:
            if efficiency.value < EFFICIENCY_CHOPPY:
                warnings.append(
                    f"choppy market: efficiency ratio {efficiency.value:.2f} -- "
                    "momentum signals whipsaw rather than persist"
                )
            elif efficiency.value >= EFFICIENCY_TRENDING:
                positives.append(f"trending market (efficiency {efficiency.value:.2f})")

    flows = flows or {}
    fii = _flow_net(flows, "FII/FPI", "FII")
    dii = _flow_net(flows, "DII")
    if fii.available and dii.available and fii.value < 0 and dii.value < 0:
        warnings.append(
            f"both FII ({fii.value:+,.0f} cr) and DII ({dii.value:+,.0f} cr) "
            "were net sellers"
        )

    return MarketRegime(
        as_of=as_of,
        breadth_pct=breadth,
        vix_level=vix_level,
        vix_percentile=vix_pct,
        efficiency_ratio=efficiency,
        fii_net=fii,
        dii_net=dii,
        warnings=warnings,
        positives=positives,
    )
