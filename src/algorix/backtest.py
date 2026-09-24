"""Historical replay and score evaluation.

Answers the only question that matters about this system: **do high scores
actually precede high returns?** Until that is measured, every score is an
untested opinion and the weights rest on nothing.

Replay is legitimate here because scoring is point-in-time by construction
(`series.py`): a series truncated to date D contains nothing after D, so
re-scoring a past session reproduces what the scanner would genuinely have
said that morning -- not a result contaminated by hindsight.

Forward returns are the one place in the codebase that deliberately looks
ahead. They live here, in the evaluation layer, and are never reachable from
an indicator.

**Read the caveats on every result.** A backtest run over history the
journal did not record carries real distortions -- survivorship bias above
all -- and reporting a number without them would be worse than reporting
nothing.

Metrics:

- **Information Coefficient (IC)** -- rank correlation between score and
  subsequent return, computed per session then averaged. The standard factor
  test. Sustained IC above ~0.03 is meaningful for a cross-sectional signal;
  IC near zero means the score carries no information about what follows.
- **Quintile spread** -- mean forward return of the top fifth minus the
  bottom fifth. The practical read: what the ranking is worth.
- **Hit rate** -- how often the top quintile beat the universe median.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

from algorix.calendar import TradingCalendar
from algorix.cross_sectional import compute_universe_momentum
from algorix.indicators import DELIVERY_BASELINE
from algorix.models import DeliveryRecord, EarningsSurpriseRecord, Exchange
from algorix.scoring import ScoredUniverse, score_universe
from algorix.series import PriceSeries, load_series
from algorix.storage import (
    DeliveryRepository,
    Database,
    EarningsSurpriseRepository,
    InstrumentRepository,
)
from algorix.universe import NIFTY_50, ConstituencyRepository

#: Horizons evaluated, in sessions. The swing-trading thesis is days-to-weeks,
#: so 5 and 20 sessions bracket the intended holding period.
DEFAULT_HORIZONS: tuple[int, ...] = (5, 10, 20)

#: Sessions of history each scored date needs behind it (the 12-month
#: momentum window plus its skip, plus headroom for feed gaps).
WARMUP_SESSIONS = 300

#: Minimum scored names before a session contributes to the metrics.
MIN_SESSION_COHORT = 10


@dataclass(frozen=True)
class ScoredOutcome:
    """One stock's score on one session, and what followed."""

    session_date: date
    symbol: str
    score: float
    forward_returns: dict[int, float] = field(default_factory=dict)
    #: Each contributor's percentile, so signals can be evaluated separately.
    #: A composite IC hides which components earn their place and which drag.
    signal_percentiles: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class HorizonMetrics:
    """Evaluation of one forward horizon."""

    horizon: int
    observations: int
    sessions: int
    mean_ic: float | None
    ic_std: float | None
    top_quintile_return: float | None
    bottom_quintile_return: float | None
    hit_rate: float | None

    @property
    def quintile_spread(self) -> float | None:
        if self.top_quintile_return is None or self.bottom_quintile_return is None:
            return None
        return self.top_quintile_return - self.bottom_quintile_return

    def describe(self) -> str:
        if self.observations == 0:
            return f"{self.horizon}d: no observations"
        parts = [f"{self.horizon}d over {self.sessions} sessions"]
        if self.mean_ic is not None:
            parts.append(f"IC {self.mean_ic:+.3f}")
        if self.quintile_spread is not None:
            parts.append(
                f"Q5-Q1 {self.quintile_spread:+.2f}% "
                f"({self.top_quintile_return:+.2f}% vs "
                f"{self.bottom_quintile_return:+.2f}%)"
            )
        if self.hit_rate is not None:
            parts.append(f"hit {self.hit_rate:.0%}")
        return " | ".join(parts)


@dataclass(frozen=True)
class BacktestResult:
    """Replay outcome, with the caveats needed to read it honestly."""

    start: date
    end: date
    sessions_scored: int
    outcomes: list[ScoredOutcome] = field(default_factory=list)
    metrics: dict[int, HorizonMetrics] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Backtest {self.start} to {self.end}",
            f"Sessions scored: {self.sessions_scored}",
            f"Observations:    {len(self.outcomes)}",
            "",
        ]
        for horizon in sorted(self.metrics):
            lines.append("  " + self.metrics[horizon].describe())
        if self.skipped:
            lines.append("")
            lines.append(
                "Skipped: "
                + ", ".join(f"{k} ({v})" for k, v in sorted(self.skipped.items()))
            )
        if self.caveats:
            lines.append("")
            lines.append("CAVEATS -- read before trusting any number above:")
            lines.extend(f"  ! {c}" for c in self.caveats)
        return "\n".join(lines)


def evaluate_signals(
    outcomes: Sequence[ScoredOutcome], horizon: int
) -> dict[str, HorizonMetrics]:
    """Evaluate each contributor separately at one horizon.

    A composite IC says whether the blend works; it cannot say which parts
    carry it. Equal weighting only makes sense if the components are
    individually informative -- a signal with persistently negative IC is
    actively subtracting, and averaging hides that.
    """
    codes = sorted({c for o in outcomes for c in o.signal_percentiles})
    results: dict[str, HorizonMetrics] = {}

    for code in codes:
        projected = [
            ScoredOutcome(
                session_date=o.session_date,
                symbol=o.symbol,
                score=o.signal_percentiles[code],
                forward_returns=o.forward_returns,
            )
            for o in outcomes
            if code in o.signal_percentiles
        ]
        results[code] = evaluate(projected, horizon)

    return results


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _ranks(values: Sequence[float]) -> list[float]:
    """Mid-ranks, so ties do not get an arbitrary order."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Rank correlation between two series, or None if undefined.

    Implemented directly rather than pulling in scipy: it is a Pearson
    correlation over mid-ranks, and one dependency is not worth ten lines.
    """
    if len(xs) != len(ys):
        raise ValueError("series must be the same length")
    if len(xs) < 3:
        return None

    rx, ry = _ranks(xs), _ranks(ys)
    mean_x, mean_y = statistics.fmean(rx), statistics.fmean(ry)

    covariance = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    var_x = sum((a - mean_x) ** 2 for a in rx)
    var_y = sum((b - mean_y) ** 2 for b in ry)

    if var_x <= 0 or var_y <= 0:
        # Every value tied on one side -- correlation is undefined, not zero.
        return None
    return covariance / ((var_x * var_y) ** 0.5)


def _quintile_means(
    pairs: list[tuple[float, float]]
) -> tuple[float | None, float | None]:
    """Mean forward return of the top and bottom fifths by score."""
    if len(pairs) < 5:
        return None, None
    ordered = sorted(pairs, key=lambda p: p[0])
    size = max(1, len(ordered) // 5)
    bottom = statistics.fmean(r for _, r in ordered[:size])
    top = statistics.fmean(r for _, r in ordered[-size:])
    return top, bottom


def evaluate(
    outcomes: Sequence[ScoredOutcome], horizon: int
) -> HorizonMetrics:
    """Compute IC, quintile spread and hit rate for one horizon."""
    by_session: dict[date, list[tuple[float, float]]] = {}
    for outcome in outcomes:
        value = outcome.forward_returns.get(horizon)
        if value is None:
            continue
        by_session.setdefault(outcome.session_date, []).append(
            (outcome.score, value)
        )

    ics: list[float] = []
    tops: list[float] = []
    bottoms: list[float] = []
    hits = 0
    counted_sessions = 0
    observations = 0

    for session, pairs in sorted(by_session.items()):
        if len(pairs) < MIN_SESSION_COHORT:
            continue
        counted_sessions += 1
        observations += len(pairs)

        ic = spearman([p[0] for p in pairs], [p[1] for p in pairs])
        if ic is not None:
            ics.append(ic)

        top, bottom = _quintile_means(pairs)
        if top is not None and bottom is not None:
            tops.append(top)
            bottoms.append(bottom)
            median = statistics.median(r for _, r in pairs)
            if top > median:
                hits += 1

    return HorizonMetrics(
        horizon=horizon,
        observations=observations,
        sessions=counted_sessions,
        mean_ic=statistics.fmean(ics) if ics else None,
        ic_std=statistics.stdev(ics) if len(ics) > 1 else None,
        top_quintile_return=statistics.fmean(tops) if tops else None,
        bottom_quintile_return=statistics.fmean(bottoms) if bottoms else None,
        hit_rate=(hits / len(tops)) if tops else None,
    )


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def replay_session(
    session: date,
    full_series: dict[str, PriceSeries],
    delivery: dict[str, list[DeliveryRecord]],
    calendar: TradingCalendar | None = None,
    universe_label: str = NIFTY_50,
    industry_by_symbol: dict[str, str | None] | None = None,
    earnings: dict[str, list[EarningsSurpriseRecord]] | None = None,
) -> ScoredUniverse:
    """Score one historical session using only data available then."""
    truncated = {
        symbol: series.truncate_to(session)
        for symbol, series in full_series.items()
    }
    truncated = {s: v for s, v in truncated.items() if v.bars}

    trimmed_delivery = {
        symbol: [r for r in records if r.session_date <= session]
        for symbol, records in delivery.items()
    }
    # A8's point-in-time discipline: a report from after `session` must not
    # be visible -- otherwise a backtest would credit the score with an
    # earnings surprise it could not actually have known about yet.
    trimmed_earnings = {
        symbol: [r for r in records if r.report_date <= session]
        for symbol, records in (earnings or {}).items()
    }

    momentum = compute_universe_momentum(truncated, session, calendar)
    return score_universe(
        truncated,
        momentum,
        session,
        delivery_by_symbol=trimmed_delivery,
        calendar=calendar,
        universe_label=universe_label,
        industry_by_symbol=industry_by_symbol,
        earnings_by_symbol=trimmed_earnings,
    )


def run_backtest(
    db: Database,
    start: date,
    end: date,
    calendar: TradingCalendar | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    symbols: list[str] | None = None,
    index_symbol: str = NIFTY_50,
) -> BacktestResult:
    """Replay scoring across a date range and evaluate what followed."""
    calendar = calendar or TradingCalendar()
    if start > end:
        raise ValueError(f"start {start} is after end {end}")

    instrument_repo = InstrumentRepository(db)
    delivery_repo = DeliveryRepository(db)
    earnings_repo = EarningsSurpriseRepository(db)
    constituency = ConstituencyRepository(db)

    caveats: list[str] = []

    # Constituency: point-in-time where recorded, otherwise today's members.
    recorded = constituency.current_constituents(index_symbol, on=start)
    if symbols is None:
        symbols = recorded or constituency.current_constituents(index_symbol)
        if not recorded:
            caveats.append(
                "SURVIVORSHIP BIAS: index membership was not recorded for the "
                "start of this range, so today's constituents were used "
                "throughout. Stocks that left the index are missing entirely, "
                "which flatters results. Membership is recorded from now on, "
                "so future backtests over this period will be clean."
            )

    if not symbols:
        return BacktestResult(
            start=start,
            end=end,
            sessions_scored=0,
            caveats=[*caveats, "no universe constituents are registered"],
        )

    # Load each instrument's full history once, then slice per session.
    full_series: dict[str, PriceSeries] = {}
    delivery: dict[str, list[DeliveryRecord]] = {}
    earnings: dict[str, list[EarningsSurpriseRecord]] = {}
    industry_by_symbol: dict[str, str | None] = {}
    earliest_delivery: date | None = None

    for symbol in symbols:
        instrument = instrument_repo.get(symbol, Exchange.NSE)
        if instrument is None or instrument.id is None:
            continue
        industry_by_symbol[symbol] = instrument.industry
        series = load_series(
            db,
            instrument.id,
            end,
            WARMUP_SESSIONS + len(calendar.sessions_in_range(start, end)) + 50,
            calendar,
        )
        if series.bars:
            full_series[symbol] = series
        records = delivery_repo.get_range(
            instrument.id, date(2000, 1, 1), end
        )
        if records:
            delivery[symbol] = records
            first = records[0].session_date
            earliest_delivery = (
                first if earliest_delivery is None else min(earliest_delivery, first)
            )
        earnings_records = earnings_repo.get_range(instrument.id, date(2000, 1, 1), end)
        if earnings_records:
            earnings[symbol] = earnings_records

    if not full_series:
        # Caveats accumulated above must survive an early exit -- a result
        # that drops its survivorship warning is worse than no result.
        return BacktestResult(
            start=start, end=end, sessions_scored=0,
            caveats=[
                *caveats,
                "no price history is stored for the requested universe",
            ],
        )

    if earliest_delivery is None:
        caveats.append(
            "No delivery data stored: the A6 delivery signal was absent "
            "throughout, so this tests a five-signal model, not the six-signal "
            "one the scanner runs."
        )
    elif earliest_delivery > start:
        caveats.append(
            f"Delivery data begins {earliest_delivery}; sessions before that "
            "were scored without the A6 signal, and are not directly "
            "comparable to later ones."
        )

    if not earnings:
        caveats.append(
            "No earnings-surprise data stored: A8 (PEAD) was absent "
            "throughout, and unlike A6 it is only ever active for the "
            "minority of the universe with a recent report even when data "
            "exists, so its absence here is easy to miss in the numbers."
        )

    caveats.append(
        "Prices are split/dividend-adjusted as of today, so historical bars "
        "reflect corporate actions that had not yet happened at the time."
    )

    sessions = calendar.sessions_in_range(start, end)
    outcomes: list[ScoredOutcome] = []
    skipped: dict[str, int] = {}
    scored_sessions = 0

    for session in sessions:
        universe = replay_session(
            session, full_series, delivery, calendar, index_symbol,
            industry_by_symbol=industry_by_symbol,
            earnings=earnings,
        )
        ranked = [s for s in universe.scores.values() if s.score.available]

        if len(ranked) < MIN_SESSION_COHORT:
            skipped["thin cohort"] = skipped.get("thin cohort", 0) + 1
            continue
        scored_sessions += 1

        for score in ranked:
            series = full_series.get(score.symbol)
            if series is None:
                continue
            forward = {}
            for horizon in horizons:
                value = series.forward_return(session, horizon)
                if value is not None:
                    forward[horizon] = value
            if forward:
                outcomes.append(
                    ScoredOutcome(
                        session_date=session,
                        symbol=score.symbol,
                        score=score.score.value,
                        forward_returns=forward,
                        signal_percentiles={
                            c.code: c.percentile for c in score.contributions
                        },
                    )
                )

    metrics = {h: evaluate(outcomes, h) for h in horizons}

    incomplete = [h for h, m in metrics.items() if m.sessions == 0]
    if incomplete:
        caveats.append(
            f"No completed {incomplete} session horizon(s) in range -- the "
            "most recent sessions have no future yet."
        )

    return BacktestResult(
        start=start,
        end=end,
        sessions_scored=scored_sessions,
        outcomes=outcomes,
        metrics=metrics,
        caveats=caveats,
        skipped=skipped,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m algorix.backtest``."""
    import argparse
    from datetime import datetime

    from algorix.config import Config

    parser = argparse.ArgumentParser(
        description="Replay scoring over past sessions and evaluate it."
    )
    parser.add_argument("--db", help="database path (overrides config)")
    parser.add_argument("--start", help="first session, YYYY-MM-DD")
    parser.add_argument("--end", help="last session, YYYY-MM-DD")
    parser.add_argument("--index", default=NIFTY_50, help="index to backtest")
    parser.add_argument(
        "--sessions", type=int, default=120,
        help="sessions back from --end when --start is omitted",
    )
    parser.add_argument(
        "--signals", action="store_true",
        help="also report each contributor's IC separately",
    )
    parser.add_argument("--signal-horizon", type=int, default=20)
    args = parser.parse_args(argv)

    config = Config.from_env()
    database = Database(args.db or config.db_path)
    calendar = TradingCalendar()

    end = (
        datetime.strptime(args.end, "%Y-%m-%d").date()
        if args.end
        else calendar.last_completed_session(datetime.now(tz=None).astimezone())
    )
    start = (
        datetime.strptime(args.start, "%Y-%m-%d").date()
        if args.start
        else calendar.trading_days_ago(end, args.sessions)
    )

    result = run_backtest(
        database, start, end, calendar, index_symbol=args.index.upper()
    )
    print(result.summary())
    if args.signals and result.outcomes:
        print()
        print(signal_report(result, horizon=args.signal_horizon))
    return 0




def signal_report(
    result: BacktestResult, horizon: int = 20
) -> str:
    """Per-signal IC table -- which contributors earn their place."""
    metrics = evaluate_signals(result.outcomes, horizon)
    if not metrics:
        return "No per-signal data available."

    labels = {
        "A1": "momentum",
        "A2": "52w high",
        "A3": "trend",
        "A4": "breakout",
        "A5": "pullback (inverted)",
        "A6": "delivery",
    }

    lines = [
        f"Per-signal IC at {horizon} sessions "
        f"(positive = the signal predicts returns):",
        "",
        f"  {'code':5}{'signal':22}{'IC':>8}{'Q5-Q1':>10}{'sessions':>10}",
        "  " + "-" * 55,
    ]
    ordered = sorted(
        metrics.items(),
        key=lambda kv: kv[1].mean_ic if kv[1].mean_ic is not None else -99,
        reverse=True,
    )
    for code, m in ordered:
        ic = f"{m.mean_ic:+.3f}" if m.mean_ic is not None else "--"
        spread = f"{m.quintile_spread:+.2f}%" if m.quintile_spread is not None else "--"
        lines.append(
            f"  {code:5}{labels.get(code, code):22}{ic:>8}{spread:>10}{m.sessions:>10}"
        )
    return "\n".join(lines)




def evaluate_by_regime(
    db: Database,
    result: BacktestResult,
    calendar: TradingCalendar | None = None,
    horizon: int = 20,
) -> dict[str, HorizonMetrics]:
    """Split evaluation by the regime in force on each session.

    This is the test of the gate itself (INDICATORS.md §0.4). If scores
    predict returns in favourable regimes and fail in hostile ones, the gate
    is doing its job and the composite IC is simply averaging two different
    worlds together. If performance is equally poor in both, the gate is not
    earning its place.
    """
    from algorix.regime import (
        INDIA_VIX,
        NIFTY_INDEX,
        FlowRepository,
        assess_regime,
    )

    calendar = calendar or TradingCalendar()
    instrument_repo = InstrumentRepository(db)

    def series_for(instrument):
        stored = instrument_repo.get(instrument.symbol, instrument.exchange)
        if stored is None or stored.id is None:
            return None
        loaded = load_series(db, stored.id, result.end, 400, calendar)
        return loaded if loaded.bars else None

    index_full = series_for(NIFTY_INDEX)
    vix_full = series_for(INDIA_VIX)
    flow_repo = FlowRepository(db)

    # Universe series, loaded once and sliced per session.
    symbols = sorted({o.symbol for o in result.outcomes})
    universe_full: dict[str, PriceSeries] = {}
    for symbol in symbols:
        stored = instrument_repo.get(symbol, Exchange.NSE)
        if stored is None or stored.id is None:
            continue
        loaded = load_series(db, stored.id, result.end, 400, calendar)
        if loaded.bars:
            universe_full[symbol] = loaded

    verdict_by_session: dict[date, str] = {}
    for session in sorted({o.session_date for o in result.outcomes}):
        regime = assess_regime(
            session,
            {s: v.truncate_to(session) for s, v in universe_full.items()},
            index_series=index_full.truncate_to(session) if index_full else None,
            vix_series=vix_full.truncate_to(session) if vix_full else None,
            flows=flow_repo.get_for(session),
            calendar=calendar,
        )
        verdict_by_session[session] = str(regime.verdict)

    grouped: dict[str, list[ScoredOutcome]] = {}
    for outcome in result.outcomes:
        verdict = verdict_by_session.get(outcome.session_date, "UNKNOWN")
        grouped.setdefault(verdict, []).append(outcome)

    return {v: evaluate(outcomes, horizon) for v, outcomes in grouped.items()}


if __name__ == "__main__":
    raise SystemExit(main())
