"""The morning scanner -- the anchor feature (PROJECT_SCOPE §3.1).

One command, run before the 09:15 open: refresh data, score the universe,
assess the regime, journal the result, deliver the digest.

Ordering is deliberate. The journal is written *before* delivery, so a
Telegram outage never costs you the record -- the digest can be re-sent from
stored data, but a scan that was never recorded is gone, and the journal is
the only thing that can ever validate the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

from algorix.calendar import IST, TradingCalendar
from algorix.cross_sectional import compute_universe_momentum
from algorix.exceptions import AlgorixError
from algorix.indicators import DELIVERY_BASELINE
from algorix.journal import Journal
from algorix.metals import metal_snapshot
from algorix.models import Exchange
from algorix.notify import NotConfiguredError, TelegramNotifier, format_digest
from algorix.regime import (
    INDIA_VIX,
    NIFTY_INDEX,
    FlowRepository,
    MarketRegime,
    NseFlowClient,
    assess_regime,
    seed_regime_instruments,
)
from algorix.scoring import ScoredUniverse, score_universe
from algorix.series import load_series
from algorix.storage import DeliveryRepository, Database, InstrumentRepository
from algorix.universe import NIFTY_50, ConstituencyRepository

#: Sessions of history loaded per instrument. The 12-month momentum window
#: with its skip needs 274, so this leaves room for feed gaps.
SCAN_LOOKBACK_SESSIONS = 400


@dataclass(frozen=True)
class ScanResult:
    session_date: date
    universe: ScoredUniverse
    regime: MarketRegime | None
    digest: str
    journal_run_id: int | None = None
    delivered: bool = False
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        ranked = self.universe.top(5)
        lines = [
            f"Session:   {self.session_date}",
            f"Scored:    {len(self.universe.ranked())} of "
            f"{len(self.universe.scores)}",
        ]
        if self.regime:
            lines.append(f"Regime:    {self.regime.verdict}")
        if ranked:
            lines.append(
                "Top:       "
                + ", ".join(f"{s.symbol} ({s.score.value:.0f})" for s in ranked)
            )
        lines.append(f"Journalled: {'yes' if self.journal_run_id else 'no'}")
        lines.append(f"Delivered:  {'yes' if self.delivered else 'no'}")
        for error in self.errors:
            lines.append(f"  - {error}")
        return "\n".join(lines)


def run_scan(
    db: Database,
    now: datetime | None = None,
    calendar: TradingCalendar | None = None,
    *,
    deliver: bool = True,
    top_n: int = 8,
    index_symbol: str = NIFTY_50,
    notifier: TelegramNotifier | None = None,
    flow_client: NseFlowClient | None = None,
) -> ScanResult:
    """Score the universe for the last completed session and deliver it."""
    calendar = calendar or TradingCalendar()
    now = now or datetime.now(IST)
    db.migrate()

    as_of = calendar.last_completed_session(now)
    errors: list[str] = []

    seed_regime_instruments(db)
    instrument_repo = InstrumentRepository(db)
    delivery_repo = DeliveryRepository(db)

    # -- load the universe -------------------------------------------------
    symbols = ConstituencyRepository(db).current_constituents(
        index_symbol, on=as_of
    )
    series_by_symbol = {}
    delivery_by_symbol = {}
    industry_by_symbol = {}

    for symbol in symbols:
        instrument = instrument_repo.get(symbol, Exchange.NSE)
        if instrument is None or instrument.id is None:
            continue
        series_by_symbol[symbol] = load_series(
            db, instrument.id, as_of, SCAN_LOOKBACK_SESSIONS, calendar
        )
        delivery_start = calendar.trading_days_ago(as_of, DELIVERY_BASELINE * 2)
        delivery_by_symbol[symbol] = delivery_repo.get_range(
            instrument.id, delivery_start, as_of
        )
        industry_by_symbol[symbol] = instrument.industry

    if not series_by_symbol:
        errors.append(
            "no universe constituents found -- run `python -m algorix.refresh` first"
        )
        empty = ScoredUniverse(as_of=as_of, scores={})
        return ScanResult(
            session_date=as_of,
            universe=empty,
            regime=None,
            digest=format_digest(empty),
            errors=errors,
        )

    # -- score -------------------------------------------------------------
    momentum = compute_universe_momentum(series_by_symbol, as_of, calendar)
    universe = score_universe(
        series_by_symbol,
        momentum,
        as_of,
        delivery_by_symbol=delivery_by_symbol,
        calendar=calendar,
        universe_label=index_symbol,
        industry_by_symbol=industry_by_symbol,
    )

    # -- regime ------------------------------------------------------------
    regime = _assess(db, as_of, series_by_symbol, calendar, flow_client, errors)

    # -- metals ------------------------------------------------------------
    metals_line = None
    try:
        snapshot = metal_snapshot(db, as_of)
        metals_line = (
            f"gold ${snapshot.gold_usd:,.0f} | silver ${snapshot.silver_usd:,.1f} "
            f"| G/S {snapshot.gold_silver_ratio:.1f} | USDINR {snapshot.usd_inr:.2f}"
        )
    except AlgorixError as exc:
        errors.append(f"metals snapshot unavailable: {exc}")

    digest = format_digest(universe, regime, top_n=top_n, metals=metals_line)

    # -- journal BEFORE delivery ------------------------------------------
    # A delivery failure must never cost the record: the digest can be re-sent
    # from stored data, but an unrecorded scan is irrecoverable.
    journal_run_id = None
    try:
        journal_run_id = Journal(db).record(
            universe, regime, universe_label=index_symbol
        )
    except Exception as exc:  # storage failure must be loud but not fatal
        errors.append(f"journal write failed: {type(exc).__name__}: {exc}")

    # -- deliver -----------------------------------------------------------
    delivered = False
    if deliver:
        try:
            delivered = (notifier or TelegramNotifier()).send(digest)
        except NotConfiguredError as exc:
            errors.append(str(exc))
        except AlgorixError as exc:
            errors.append(f"delivery failed: {type(exc).__name__}: {exc}")

    return ScanResult(
        session_date=as_of,
        universe=universe,
        regime=regime,
        digest=digest,
        journal_run_id=journal_run_id,
        delivered=delivered,
        errors=errors,
    )


def _assess(
    db: Database,
    as_of: date,
    series_by_symbol: dict,
    calendar: TradingCalendar,
    flow_client: NseFlowClient | None,
    errors: list[str],
) -> MarketRegime | None:
    """Build the regime picture, tolerating any individual gate being absent."""
    instrument_repo = InstrumentRepository(db)

    def series_for(symbol: str, exchange: Exchange):
        instrument = instrument_repo.get(symbol, exchange)
        if instrument is None or instrument.id is None:
            return None
        loaded = load_series(db, instrument.id, as_of, 300, calendar)
        return loaded if loaded.bars else None

    index_series = series_for(NIFTY_INDEX.symbol, NIFTY_INDEX.exchange)
    vix_series = series_for(INDIA_VIX.symbol, INDIA_VIX.exchange)

    flow_repo = FlowRepository(db)
    flows = flow_repo.get_for(as_of)
    if not flows:
        try:
            records = (flow_client or NseFlowClient()).fetch()
            flow_repo.upsert_many(records, source="nse-fiidii")
            flows = flow_repo.get_for(as_of)
        except AlgorixError as exc:
            errors.append(f"FII/DII unavailable: {type(exc).__name__}: {exc}")

    return assess_regime(
        as_of,
        series_by_symbol,
        index_series=index_series,
        vix_series=vix_series,
        flows=flows,
        calendar=calendar,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m algorix.scan``."""
    import argparse

    from algorix.config import Config

    parser = argparse.ArgumentParser(description="Run the Algorix morning scan.")
    parser.add_argument("--db", help="database path (overrides config)")
    parser.add_argument(
        "--no-send", action="store_true", help="print the digest, do not deliver"
    )
    parser.add_argument("--top", type=int, default=8, help="entries in the digest")
    parser.add_argument("--index", default=NIFTY_50, help="index to scan")
    parser.add_argument(
        "--print", dest="show", action="store_true", help="also print the digest"
    )
    args = parser.parse_args(argv)

    config = Config.from_env()
    config.ensure_data_dir()
    database = Database(args.db or config.db_path)

    result = run_scan(
        database, deliver=not args.no_send, top_n=args.top,
        index_symbol=args.index.upper(),
    )

    if args.no_send or args.show:
        print(result.digest)
        print()
    print(result.summary())
    return 0 if not result.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
