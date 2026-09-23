"""Data-layer orchestration: one command that brings the database up to date.

Runs the whole pipeline for the latest completed session -- universe sync,
price ingestion, delivery ingestion -- and returns a single report saying what
landed, what was rejected, and what is still missing.

Three behaviours are deliberate:

**Incremental, with overlap.** Instruments with no history get a full initial
load; thereafter only recent sessions are re-fetched. The overlap is not
waste: split- and dividend-adjusted prices are restated retroactively, so the
last few sessions must be re-read to pick up a corporate action. Upserts make
that harmless.

**Failures are collected, never fatal.** One delisted symbol or one missing
bhavcopy must not abandon a 50-instrument refresh -- but every failure appears
in the report. CLAUDE.md forbids a silently partial run.

**Gaps are reported per instrument.** A missing session on an NSE-calendar
instrument is a problem; the same gap on a COMEX or FX series is usually just
a US holiday. The report separates them so a real gap is not lost in noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from algorix.calendar import IST, TradingCalendar
from algorix.delivery import (
    DeliveryIngestReport,
    NseBhavcopyClient,
    ingest_delivery,
)
from algorix.exceptions import AlgorixError, DataUnavailableError
from algorix.ingestion import IngestReport, YFinanceBarSource, ingest_instrument
from algorix.metals import METAL_INSTRUMENTS, seed_metal_instruments
from algorix.models import CalendarPolicy, Exchange, Instrument, InstrumentType
from algorix.regime import REGIME_INSTRUMENTS, seed_regime_instruments
from algorix.storage import BarRepository, Database, InstrumentRepository
from algorix.universe import (
    NIFTY_50,
    ConstituencyRepository,
    NseIndexClient,
    SyncResult,
)

#: Initial history depth. The 200 DMA (A3), 52-week high (A2) and 12-month
#: momentum (A1) all need more than a year, so two years gives headroom.
DEFAULT_INITIAL_HISTORY_DAYS = 730

#: Sessions re-fetched on every incremental run, so retroactive price
#: adjustments after a split or dividend are picked up.
DEFAULT_OVERLAP_SESSIONS = 5

#: Delivery history depth. The A6 trend compares a 5-session mean to a
#: 20-session baseline, so a handful of spare sessions keeps it usable even
#: when a few are unpublished.
DEFAULT_DELIVERY_HISTORY = 30

#: Bhavcopy files fetched per run. Each is a separate ~400KB request against
#: NSE, so backfill is spread over successive runs rather than hammering the
#: archive in one go.
DEFAULT_MAX_DELIVERY_FETCHES = 12


@dataclass(frozen=True)
class RefreshReport:
    """Outcome of one full data refresh."""

    session_date: date
    universe_sync: SyncResult | None = None
    bar_reports: list[IngestReport] = field(default_factory=list)
    delivery: DeliveryIngestReport | None = None
    #: Sessions fetched this run (empty when history was already complete).
    delivery_fetched: list[date] = field(default_factory=list)
    #: Total delivery sessions stored over the history window -- this, not
    #: the fetch count, is what tells you whether A6 can compute.
    delivery_coverage: int = 0
    delivery_required: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def instruments_refreshed(self) -> int:
        return len(self.bar_reports)

    @property
    def bars_stored(self) -> int:
        return sum(r.stored for r in self.bar_reports)

    @property
    def instruments_with_gaps(self) -> list[str]:
        return sorted(r.symbol for r in self.bar_reports if r.missing_sessions)

    @property
    def delivery_ready(self) -> bool:
        """Whether enough delivery history exists for the A6 trend."""
        from algorix.indicators import DELIVERY_BASELINE

        return self.delivery_coverage >= DELIVERY_BASELINE

    @property
    def is_clean(self) -> bool:
        return not self.errors and not self.instruments_with_gaps

    def summary(self) -> str:
        lines = [
            f"Session:      {self.session_date}",
            f"Instruments:  {self.instruments_refreshed}",
            f"Bars stored:  {self.bars_stored}",
        ]
        if self.universe_sync is not None:
            sync = self.universe_sync
            lines.append(
                f"Universe:     +{len(sync.added)} / -{len(sync.removed)} "
                f"({len(sync.unchanged)} unchanged)"
            )
        if self.delivery_required:
            state = "ready" if self.delivery_ready else "backfilling"
            lines.append(
                f"Delivery:     {self.delivery_coverage}/{self.delivery_required} "
                f"sessions ({len(self.delivery_fetched)} fetched this run) "
                f"-- {state}"
            )
        if self.instruments_with_gaps:
            lines.append(f"Gaps:         {', '.join(self.instruments_with_gaps)}")
        if self.errors:
            lines.append("Errors:")
            lines.extend(f"  - {e}" for e in self.errors)
        if self.is_clean:
            lines.append("Status:       clean")
        return "\n".join(lines)


def _start_date_for(
    repository: BarRepository,
    instrument_id: int,
    target: date,
    calendar: TradingCalendar,
    initial_history_days: int,
    overlap_sessions: int,
) -> date:
    """Where to begin fetching: full history, or a short overlapping window."""
    latest = repository.latest_session(instrument_id)
    if latest is None:
        return target - timedelta(days=initial_history_days)

    try:
        return calendar.trading_days_ago(min(latest, target), overlap_sessions)
    except AlgorixError:
        return target - timedelta(days=initial_history_days)


def refresh(
    db: Database,
    now: datetime | None = None,
    calendar: TradingCalendar | None = None,
    *,
    skip_universe: bool = False,
    skip_delivery: bool = False,
    index_symbol: str = NIFTY_50,
    initial_history_days: int = DEFAULT_INITIAL_HISTORY_DAYS,
    overlap_sessions: int = DEFAULT_OVERLAP_SESSIONS,
    delivery_history_sessions: int = DEFAULT_DELIVERY_HISTORY,
    max_delivery_fetches: int = DEFAULT_MAX_DELIVERY_FETCHES,
    bar_source: YFinanceBarSource | None = None,
    index_client: NseIndexClient | None = None,
    bhavcopy_client: NseBhavcopyClient | None = None,
) -> RefreshReport:
    """Bring the database up to date through the last completed session.

    Safe to run repeatedly: every write is an upsert, so a second run on the
    same day changes nothing.
    """
    calendar = calendar or TradingCalendar()
    now = now or datetime.now(IST)
    db.migrate()

    target = calendar.last_completed_session(now)
    errors: list[str] = []

    # -- universe ---------------------------------------------------------
    universe_sync: SyncResult | None = None
    if not skip_universe:
        try:
            client = index_client or NseIndexClient()
            constituency = ConstituencyRepository(db)
            universe_sync = constituency.sync(
                index_symbol, client.fetch_index(index_symbol), target
            )
        except AlgorixError as exc:
            # A failed universe sync is not fatal: previously stored
            # constituents remain valid and can still be refreshed.
            errors.append(f"universe sync failed: {type(exc).__name__}: {exc}")

    seed_metal_instruments(db)
    # The index and India VIX are not index members, so nothing else would
    # pull them in -- yet B2 and B3 are computed from them. Without this the
    # two gates silently report unavailable forever.
    seed_regime_instruments(db)

    # -- assemble the instrument set --------------------------------------
    instruments = _instruments_to_refresh(db, target, index_symbol)
    if not instruments:
        errors.append("no instruments registered; nothing to refresh")
        return RefreshReport(
            session_date=target, universe_sync=universe_sync, errors=errors
        )

    # -- prices -----------------------------------------------------------
    source = bar_source or YFinanceBarSource(calendar)
    bar_repository = BarRepository(db)
    bar_reports: list[IngestReport] = []

    for instrument, instrument_id in instruments:
        start = _start_date_for(
            bar_repository,
            instrument_id,
            target,
            calendar,
            initial_history_days,
            overlap_sessions,
        )
        try:
            bar_reports.append(
                ingest_instrument(
                    db,
                    instrument,
                    instrument_id,
                    start,
                    target,
                    calendar,
                    source=source,
                    max_session=target,
                )
            )
        except AlgorixError as exc:
            errors.append(
                f"{instrument.symbol}: {type(exc).__name__}: {exc}"
            )

    # -- delivery ---------------------------------------------------------
    delivery: DeliveryIngestReport | None = None
    delivery_fetched: list[date] = []
    delivery_coverage = 0
    delivery_required = 0

    if not skip_delivery:
        # Indices trade on NSE but never appear in the bhavcopy delivery
        # file, so including them would report them missing on every run --
        # permanent noise that would bury a real gap.
        equity_ids = {
            instrument.symbol: instrument_id
            for instrument, instrument_id in instruments
            if instrument.exchange is Exchange.NSE
            and instrument.instrument_type is not InstrumentType.INDEX
        }
        if equity_ids:
            client = bhavcopy_client or NseBhavcopyClient()
            # Delivery history matters as much as today's figure: the A6
            # trend compares recent delivery to a 20-session baseline, so a
            # single session leaves the indicator unusable.
            wanted = _delivery_sessions_needed(
                db, target, calendar, delivery_history_sessions, max_delivery_fetches
            )
            for session in wanted:
                try:
                    report = ingest_delivery(db, session, equity_ids, client=client)
                    delivery_fetched.append(session)
                    if session == target:
                        delivery = report
                except AlgorixError as exc:
                    errors.append(
                        f"delivery {session}: {type(exc).__name__}: {exc}"
                    )

            delivery_coverage, delivery_required = _delivery_coverage(
                db, target, calendar, delivery_history_sessions
            )

    return RefreshReport(
        session_date=target,
        universe_sync=universe_sync,
        bar_reports=bar_reports,
        delivery=delivery,
        delivery_fetched=delivery_fetched,
        delivery_coverage=delivery_coverage,
        delivery_required=delivery_required,
        errors=errors,
    )


def _delivery_coverage(
    db: Database,
    target: date,
    calendar: TradingCalendar,
    history_sessions: int,
) -> tuple[int, int]:
    """(sessions stored, sessions expected) across the history window."""
    from algorix.storage import DeliveryRepository

    start = calendar.trading_days_ago(target, history_sessions)
    expected = calendar.sessions_in_range(start, target)
    present = DeliveryRepository(db).sessions_present(start, target)
    return len(present), len(expected)


def _delivery_sessions_needed(
    db: Database,
    target: date,
    calendar: TradingCalendar,
    history_sessions: int,
    max_fetches: int,
) -> list[date]:
    """Sessions still lacking delivery data, newest first, capped.

    Newest first so the most recent figure is always current; the cap spreads
    a cold-start backfill across runs instead of issuing thirty requests to
    NSE at once.
    """
    from algorix.storage import DeliveryRepository

    start = calendar.trading_days_ago(target, history_sessions)
    expected = calendar.sessions_in_range(start, target)
    present = DeliveryRepository(db).sessions_present(start, target)

    missing = [s for s in reversed(expected) if s not in present]
    return missing[:max_fetches]


def _instruments_to_refresh(
    db: Database, target: date, index_symbol: str = NIFTY_50
) -> list[tuple[Instrument, int]]:
    """Current index members plus the metals and regime tracks, de-duplicated."""
    instrument_repository = InstrumentRepository(db)
    constituency = ConstituencyRepository(db)

    wanted: dict[str, Instrument] = {}

    for symbol in constituency.current_constituents(index_symbol, on=target):
        stored = instrument_repository.get(symbol, Exchange.NSE)
        if stored is not None:
            wanted[f"{stored.exchange}:{stored.symbol}"] = stored

    for instrument in (*METAL_INSTRUMENTS, *REGIME_INSTRUMENTS):
        stored = instrument_repository.get(instrument.symbol, instrument.exchange)
        if stored is not None:
            wanted[f"{stored.exchange}:{stored.symbol}"] = stored

    return [
        (instrument, instrument.id)
        for instrument in wanted.values()
        if instrument.id is not None
    ]


def gap_report(
    db: Database,
    start: date,
    end: date,
    calendar: TradingCalendar | None = None,
) -> dict[str, list[date]]:
    """Missing sessions per instrument over a window.

    NSE-calendar instruments are checked against NSE sessions. Instruments on
    GLOBAL policy are checked too, but a gap there is often a US holiday
    rather than a fault -- callers should weight them differently.
    """
    calendar = calendar or TradingCalendar()
    expected = calendar.sessions_in_range(start, end)
    bar_repository = BarRepository(db)

    gaps: dict[str, list[date]] = {}
    for instrument in InstrumentRepository(db).list_active():
        if instrument.id is None:
            continue
        missing = bar_repository.missing_sessions(instrument.id, expected)
        if missing:
            gaps[instrument.symbol] = missing
    return gaps


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m algorix.refresh``."""
    import argparse

    from algorix.config import Config

    parser = argparse.ArgumentParser(description="Refresh the Algorix data layer.")
    parser.add_argument("--db", help="database path (overrides config)")
    parser.add_argument(
        "--skip-universe", action="store_true", help="do not re-sync the index"
    )
    parser.add_argument(
        "--skip-delivery", action="store_true", help="do not fetch delivery data"
    )
    parser.add_argument(
        "--index", default=NIFTY_50,
        help="index to track (NIFTY50, NIFTY200, NIFTY500, NIFTYMIDCAP150)",
    )
    parser.add_argument(
        "--history-days",
        type=int,
        default=DEFAULT_INITIAL_HISTORY_DAYS,
        help="initial history depth for new instruments",
    )
    args = parser.parse_args(argv)

    config = Config.from_env()
    config.ensure_data_dir()
    database = Database(args.db or config.db_path)

    report = refresh(
        database,
        skip_universe=args.skip_universe,
        skip_delivery=args.skip_delivery,
        index_symbol=args.index.upper(),
        initial_history_days=args.history_days,
    )
    print(report.summary())
    return 0 if report.is_clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
