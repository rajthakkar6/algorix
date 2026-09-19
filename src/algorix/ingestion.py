"""OHLCV ingestion.

Price data arrives dirty. Three failure modes are handled here, each observed
in real yfinance responses for NSE equities:

1. **Phantom holiday bars.** yfinance emits a row for 2026-09-14 -- an NSE
   holiday -- with all four prices identical and zero volume. Stored, it
   becomes a session that never happened: it flattens ATR (zero range), drags
   moving averages, and makes gap detection believe history is complete. Every
   bar is therefore checked against the trading calendar.

2. **In-progress sessions.** A fetch during market hours returns today's
   partial bar, whose "close" is just the last trade so far. Stored as a
   close, it is simply wrong. Bars after the last *completed* session are
   dropped.

3. **NaN / malformed rows.** Missing prices arrive as NaN rather than absent
   rows.

Rejected rows are returned, never silently dropped -- CLAUDE.md requires a
data gap to be visible. Callers are expected to report a non-empty
`rejected` list rather than proceed quietly.

Prices are split- and dividend-adjusted (`auto_adjust=True`). Unadjusted
series show a 1:2 split as a 50% single-day crash, which every momentum and
trend signal would read as a genuine collapse. The tradeoff: adjusted history
changes retroactively after a corporate action, so re-ingesting an old range
can legitimately update stored prices. Upserts handle that by design.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from algorix.calendar import CalendarRangeError, TradingCalendar
from algorix.exceptions import (
    DataIntegrityError,
    DataUnavailableError,
    SourceUnreachableError,
)
from algorix.models import Bar, CalendarPolicy, Instrument
from algorix.storage import BarRepository, Database

SOURCE_YFINANCE = "yfinance"

_REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


@dataclass(frozen=True)
class RejectedBar:
    """A row that did not become a stored bar, and why."""

    session_date: date
    reason: str


@dataclass(frozen=True)
class FetchResult:
    """Bars that survived validation, plus everything that did not."""

    bars: list[Bar] = field(default_factory=list)
    rejected: list[RejectedBar] = field(default_factory=list)

    @property
    def has_rejections(self) -> bool:
        return bool(self.rejected)

    def rejection_summary(self) -> str:
        if not self.rejected:
            return "no rejected rows"
        return "; ".join(f"{r.session_date}: {r.reason}" for r in self.rejected)


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return bool(pd.isna(value))


def bars_from_dataframe(
    frame: pd.DataFrame,
    calendar: TradingCalendar | None,
    max_session: date,
) -> FetchResult:
    """Convert a yfinance history frame into validated bars.

    Pure and network-free, so every dirty-data case above is testable offline.

    Args:
        frame: yfinance `Ticker.history()` output.
        calendar: used to reject rows dated on non-sessions. Pass None for
            instruments that do not follow the NSE calendar (COMEX futures,
            FX) -- they genuinely trade on Indian holidays, and filtering them
            against NSE sessions would discard real data.
        max_session: the last completed session; later rows are in-progress.
    """
    if frame is None or frame.empty:
        raise DataUnavailableError("No price rows returned")

    missing_columns = [c for c in _REQUIRED_COLUMNS if c not in frame.columns]
    if missing_columns:
        raise DataIntegrityError(
            f"Price frame is missing column(s): {missing_columns}. "
            f"Got: {list(frame.columns)}"
        )

    bars: list[Bar] = []
    rejected: list[RejectedBar] = []

    for index_value, row in frame.iterrows():
        session_date = _to_session_date(index_value)
        if session_date is None:
            continue

        if session_date > max_session:
            rejected.append(
                RejectedBar(
                    session_date,
                    f"session is not complete as of {max_session}; "
                    "an in-progress close is not a close",
                )
            )
            continue

        if calendar is not None:
            try:
                if not calendar.is_trading_day(session_date):
                    rejected.append(
                        RejectedBar(
                            session_date,
                            "not an NSE trading session (phantom bar from the feed)",
                        )
                    )
                    continue
            except CalendarRangeError as exc:
                rejected.append(RejectedBar(session_date, str(exc)))
                continue

        values = {col: row.get(col) for col in _REQUIRED_COLUMNS}
        missing = [col for col, value in values.items() if _is_missing(value)]
        if missing:
            rejected.append(
                RejectedBar(session_date, f"missing value(s) for {missing}")
            )
            continue

        try:
            bars.append(
                Bar(
                    session_date=session_date,
                    open=float(values["Open"]),
                    high=float(values["High"]),
                    low=float(values["Low"]),
                    close=float(values["Close"]),
                    volume=int(values["Volume"]),
                )
            )
        except (DataIntegrityError, ValueError, TypeError) as exc:
            rejected.append(RejectedBar(session_date, str(exc)))

    bars.sort(key=lambda b: b.session_date)
    return FetchResult(bars=bars, rejected=rejected)


def _to_session_date(index_value: object) -> date | None:
    """Normalise a frame index entry to an IST calendar date."""
    if isinstance(index_value, pd.Timestamp):
        # yfinance returns tz-aware timestamps for NSE (Asia/Kolkata). Taking
        # .date() on a UTC-converted value would shift some sessions back a day.
        return index_value.date()
    if isinstance(index_value, date):
        return index_value
    return None


class YFinanceBarSource:
    """Fetches OHLCV history from yfinance. The only networked piece."""

    def __init__(self, calendar: TradingCalendar | None = None) -> None:
        self.calendar = calendar or TradingCalendar()

    def fetch(
        self,
        instrument: Instrument,
        start: date,
        end: date,
        max_session: date | None = None,
    ) -> FetchResult:
        """Fetch bars for `instrument` between `start` and `end` inclusive.

        Args:
            max_session: last completed session. Defaults to `end`, but pass
                the calendar's value when fetching up to today so an
                in-progress bar is excluded.
        """
        import yfinance as yf

        if start > end:
            raise ValueError(f"start {start} is after end {end}")

        symbol = instrument.yahoo_symbol
        if not symbol:
            raise DataUnavailableError(
                f"{instrument.symbol} has no yahoo_symbol; cannot fetch from yfinance"
            )

        try:
            frame = yf.Ticker(symbol).history(
                start=start.isoformat(),
                # yfinance treats `end` as exclusive.
                end=(end + timedelta(days=1)).isoformat(),
                auto_adjust=True,
            )
        except Exception as exc:  # yfinance raises assorted network errors
            raise SourceUnreachableError(
                f"Could not fetch {symbol} from yfinance: {exc}"
            ) from exc

        if frame is None or frame.empty:
            # yfinance returns an empty frame (not an error) for a delisted or
            # misspelled symbol. Treating that as "no data today" would hide a
            # broken instrument indefinitely.
            raise DataUnavailableError(
                f"yfinance returned no rows for {symbol} between {start} and {end}"
            )

        # Only NSE-calendar instruments get session filtering; see CalendarPolicy.
        calendar = (
            self.calendar
            if instrument.calendar_policy is CalendarPolicy.NSE
            else None
        )
        return bars_from_dataframe(
            frame, calendar, max_session if max_session is not None else end
        )


@dataclass(frozen=True)
class IngestReport:
    """Outcome of ingesting one instrument."""

    symbol: str
    stored: int
    rejected: list[RejectedBar]
    missing_sessions: list[date]

    @property
    def is_clean(self) -> bool:
        return not self.rejected and not self.missing_sessions


def ingest_instrument(
    db: Database,
    instrument: Instrument,
    instrument_id: int,
    start: date,
    end: date,
    calendar: TradingCalendar,
    source: YFinanceBarSource | None = None,
    max_session: date | None = None,
) -> IngestReport:
    """Fetch, validate and store bars for one instrument.

    Reports rejected rows and any expected session that ended up with no bar,
    so an incomplete ingestion is visible rather than silently partial.
    """
    source = source or YFinanceBarSource(calendar)
    result = source.fetch(instrument, start, end, max_session=max_session)

    repository = BarRepository(db)
    stored = repository.upsert_many(instrument_id, result.bars, source=SOURCE_YFINANCE)

    expected = calendar.sessions_in_range(
        start, min(end, max_session) if max_session is not None else end
    )
    missing = repository.missing_sessions(instrument_id, expected)

    return IngestReport(
        symbol=instrument.symbol,
        stored=stored,
        rejected=result.rejected,
        missing_sessions=missing,
    )


def ingest_many(
    db: Database,
    instruments: Iterable[tuple[Instrument, int]],
    start: date,
    end: date,
    calendar: TradingCalendar,
    source: YFinanceBarSource | None = None,
    max_session: date | None = None,
) -> list[IngestReport]:
    """Ingest several instruments, continuing past individual failures.

    One delisted symbol must not abort a 50-stock universe refresh, but the
    failure still has to surface -- it becomes a report with zero stored bars
    and the reason recorded.
    """
    source = source or YFinanceBarSource(calendar)
    reports: list[IngestReport] = []

    for instrument, instrument_id in instruments:
        try:
            reports.append(
                ingest_instrument(
                    db,
                    instrument,
                    instrument_id,
                    start,
                    end,
                    calendar,
                    source=source,
                    max_session=max_session,
                )
            )
        except (DataUnavailableError, SourceUnreachableError, DataIntegrityError) as exc:
            reports.append(
                IngestReport(
                    symbol=instrument.symbol,
                    stored=0,
                    rejected=[RejectedBar(end, f"{type(exc).__name__}: {exc}")],
                    missing_sessions=calendar.sessions_in_range(
                        start,
                        min(end, max_session) if max_session is not None else end,
                    ),
                )
            )

    return reports
