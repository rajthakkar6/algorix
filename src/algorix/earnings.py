"""A8: Post-Earnings Announcement Drift (INDICATORS.md Bucket F, promoted).

**Sourcing question this resolves.** PROJECT_SCOPE deferred PEAD, saying
only "reconsider if an earnings-surprise feed is obtainable; the constraint
is data sourcing in India, not merit." It is obtainable: yfinance (already
a dependency) exposes `Ticker.earnings_dates` -- EPS estimate, reported
actual, and surprise% per quarter -- with real coverage across the Nifty
50: 49 of 50 names have it (only JIOFIN, a thin-coverage 2023 spinoff,
does not), history back to 2020. Verified against a real, independent
source before trusting it: RELIANCE's yfinance earnings date for its
2026-07-17 report matches NSE's own "Outcome of Board Meeting" announcement
for that quarter exactly (see announcements.py's live-ingested data).

**Verified for signal, not just availability.** Before this was built, a
390-event pooled test across the real Nifty 50 price history found IC
+0.10 to +0.12 at 5/10/20 day horizons -- positive at every horizon, in
the direction PEAD predicts. Grouped into ~27 independent-ish reporting
weeks (events cluster by earnings season, so pooling overstates
independence the same way daily-sampled overlapping windows did earlier
in this project): 10d IC +0.127, t=+1.99 -- at the edge of significance,
the strongest result found anywhere in this codebase's validation work so
far, though still not conclusively proven at this sample size.

**One risk this module cannot rule out.** yfinance's "EPS Estimate" field
provenance is not documented publicly. If it is silently back-filled or
adjusted after the actual result is known (rather than being a genuine
point-in-time pre-earnings consensus), the measured IC above is inflated
by hindsight the live scanner would not actually have. No historical
snapshot exists to test this directly. Treat the measured IC as
encouraging, not proven, until enough journalled real-time scores exist to
check it going forward.

**This is a company-idiosyncratic effect, not a shared-sector one** --
unlike A1-A4 (see scoring.py's SECTOR_DEMEANED_CONTRIBUTORS), there is no
structural reason a whole sector should surprise in the same direction at
once, so A8 is not sector-demeaned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from algorix.exceptions import DataUnavailableError, SourceUnreachableError
from algorix.models import EarningsSurpriseRecord, Instrument

SOURCE_YFINANCE_EARNINGS = "yfinance-earnings"


@dataclass(frozen=True)
class EarningsFetchResult:
    records: list[EarningsSurpriseRecord] = field(default_factory=list)


def parse_earnings_dates(symbol: str, frame) -> EarningsFetchResult:
    """Parse yfinance's `Ticker.earnings_dates` frame into records.

    A pure function, separated from the network call, so it is testable
    against a constructed DataFrame shaped like the real one -- mirrors
    `announcements.parse_announcements`.
    """
    if frame is None or frame.empty:
        # A real, legitimate outcome, not a fault -- JIOFIN (thin analyst
        # coverage, listed 2023) has none. Never inferred as zero surprise;
        # simply nothing to score on yet.
        return EarningsFetchResult(records=[])

    reported = frame.dropna(subset=["Reported EPS", "Surprise(%)"])

    records: list[EarningsSurpriseRecord] = []
    for timestamp, row in reported.iterrows():
        records.append(
            EarningsSurpriseRecord(
                symbol=symbol,
                # yfinance's earnings timestamps carry a US-market-hours
                # tzoffset regardless of the underlying exchange -- not
                # meaningful for an Indian stock, and verified (see module
                # docstring) to shift the *time*, not the *calendar date*.
                # Only the date is ever used.
                report_date=timestamp.date(),
                eps_estimate=float(row["EPS Estimate"]),
                eps_actual=float(row["Reported EPS"]),
                surprise_pct=float(row["Surprise(%)"]),
            )
        )
    return EarningsFetchResult(records=records)


class YFinanceEarningsClient:
    """Fetches earnings-surprise history. The only networked piece."""

    def fetch(self, instrument: Instrument) -> EarningsFetchResult:
        import yfinance as yf

        symbol = instrument.yahoo_symbol
        if not symbol:
            raise DataUnavailableError(
                f"{instrument.symbol} has no yahoo_symbol; cannot fetch "
                "earnings from yfinance"
            )

        try:
            frame = yf.Ticker(symbol).earnings_dates
        except Exception as exc:  # yfinance raises assorted network errors
            raise SourceUnreachableError(
                f"Could not fetch {symbol} earnings from yfinance: {exc}"
            ) from exc

        return parse_earnings_dates(instrument.symbol, frame)


@dataclass(frozen=True)
class EarningsIngestReport:
    symbol: str
    stored: int


def ingest_earnings(
    db,
    instrument: Instrument,
    instrument_id: int,
    result: EarningsFetchResult | None = None,
    client: YFinanceEarningsClient | None = None,
) -> EarningsIngestReport:
    """Fetch (or accept an already-fetched `result`) and store.

    No incremental windowing, unlike bars/delivery/announcements: yfinance
    returns the whole small history (~25 rows) in one call regardless, so
    there is nothing to save by narrowing the request, and upserting the
    full set is idempotent and cheap.
    """
    from algorix.storage import EarningsSurpriseRepository

    if result is None:
        client = client or YFinanceEarningsClient()
        result = client.fetch(instrument)

    repository = EarningsSurpriseRepository(db)
    stored = repository.upsert_many(
        instrument_id, result.records, source=SOURCE_YFINANCE_EARNINGS
    )
    return EarningsIngestReport(symbol=instrument.symbol, stored=stored)
