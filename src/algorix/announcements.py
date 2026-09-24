"""NSE corporate announcements (INDICATORS.md G1).

The highest-quality news source available to this tool: official filings,
structured, timestamped, free, and impossible to astroturf. This module is
**data plumbing only** -- fetch, parse, store. It does not judge, score, or
classify anything, and nothing here feeds `score_universe`. G0 is explicit
that sentiment/news must never be a directional score contributor; turning
this raw feed into event flags or risk signals is a separate, later concern
(G4) with real per-item LLM cost that has not been sized or built yet.

Source is `nseindia.com/api/corporate-announcements` -- a live JSON API, not
an archived file like bhavcopy, so it needs the same session-priming
`regime.NseFlowClient` already uses (NSE rejects a cold request with no
cookie from the main site).

**Verified live, 2026-09-24, and worth reading before trusting this feed:**

- **No silent truncation found, unlike bhavcopy and the index feed.**
  Requesting `from_date` 30 through 365 days back (both market-wide and
  per-symbol) each returned data tracking the requested start closely --
  the small gaps between a requested date and the earliest row returned
  matched genuine quiet periods, not a hidden cap. An earlier draft of
  this module claimed a ~2-month cap existed; that was traced to a bug in
  the *test script*, not the API -- it sorted `"01-Aug-2026"`-style date
  labels as strings, which does not match chronological order. Recorded
  here so the same mistake is not repeated: **when comparing NSE's `an_dt`
  strings, always parse them to `date` first.** This does not prove no cap
  exists further back than a year -- only that one was tested for and not
  found, in the same spirit as every other verified quirk in this
  codebase.
- **An unknown or misspelled symbol returns HTTP 200 with an empty list**,
  not an error -- exactly like the delisted-symbol case documented for
  yfinance. Treated as "no announcements found", which is what it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import requests

from algorix.exceptions import DataIntegrityError, SourceUnreachableError
from algorix.models import AnnouncementRecord

NSE_ANNOUNCEMENTS_URL = "https://www.nseindia.com/api/corporate-announcements"

SOURCE_NSE_ANNOUNCEMENTS = "nse-announcements"

_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "application/json,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

#: NSE writes the announcement timestamp as "24-Sep-2026 15:38:34".
_TIMESTAMP_FORMAT = "%d-%b-%Y %H:%M:%S"

#: NSE's date-range query parameters want dd-mm-YYYY.
_QUERY_DATE_FORMAT = "%d-%m-%Y"


@dataclass(frozen=True)
class AnnouncementFetchResult:
    """One fetch's parsed announcements, plus enough to judge coverage.

    `skipped` counts rows that failed to parse (a field NSE changed shape,
    a genuinely malformed entry) -- collected and reported, never silently
    dropped, matching refresh.py's "failures are collected, never fatal"
    rule applied to a single response instead of a whole run.
    """

    records: list[AnnouncementRecord] = field(default_factory=list)
    skipped: int = 0

    @property
    def earliest(self) -> date | None:
        if not self.records:
            return None
        return min(r.announced_at.date() for r in self.records)


def _parse_timestamp(raw: str, seq_id: str) -> datetime:
    cleaned = (raw or "").strip()
    if not cleaned:
        raise DataIntegrityError(f"announcement {seq_id}: missing an_dt")
    try:
        return datetime.strptime(cleaned, _TIMESTAMP_FORMAT)
    except ValueError as exc:
        raise DataIntegrityError(
            f"announcement {seq_id}: unparseable an_dt {raw!r}"
        ) from exc


def parse_announcements(payload: object) -> AnnouncementFetchResult:
    """Parse NSE's corporate-announcements JSON into records.

    `payload` is whatever `response.json()` returned -- validated here
    rather than trusted, because NSE's own contract is "a list of objects",
    not a promise every object has every field this parser wants.
    """
    if not isinstance(payload, list):
        raise DataIntegrityError(
            f"corporate announcements: expected a JSON list, got "
            f"{type(payload).__name__}"
        )

    records: list[AnnouncementRecord] = []
    skipped = 0

    for row in payload:
        if not isinstance(row, dict):
            skipped += 1
            continue
        seq_id = str(row.get("seq_id") or "").strip()
        symbol = str(row.get("symbol") or "").strip().upper()
        text = str(row.get("attchmntText") or "").strip()
        category = str(row.get("desc") or "").strip() or "Unspecified"

        if not seq_id or not symbol or not text:
            # NSE's own contract: these three are always populated in
            # practice (verified live). A row missing one is not a smaller
            # announcement, it's a shape this parser has not seen before --
            # skip and count it rather than store a half-formed record.
            skipped += 1
            continue

        try:
            announced_at = _parse_timestamp(row.get("an_dt", ""), seq_id)
            record = AnnouncementRecord(
                seq_id=seq_id,
                symbol=symbol,
                announced_at=announced_at,
                category=category,
                text=text,
                attachment_url=(row.get("attchmntFile") or None),
                isin=(row.get("sm_isin") or None),
            )
        except DataIntegrityError:
            skipped += 1
            continue

        records.append(record)

    return AnnouncementFetchResult(records=records, skipped=skipped)


class NseAnnouncementsClient:
    """Fetches corporate announcements. The only networked piece."""

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        self.timeout_seconds = timeout_seconds

    def _get(self, params: dict[str, str]) -> object:
        session = requests.Session()
        try:
            # As with FII/DII, a cold request with no cookie from the main
            # site is rejected.
            session.get(
                "https://www.nseindia.com",
                headers=_NSE_HEADERS,
                timeout=self.timeout_seconds,
            )
            response = session.get(
                NSE_ANNOUNCEMENTS_URL,
                headers=_NSE_HEADERS,
                params=params,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise SourceUnreachableError(
                f"Could not reach NSE corporate-announcements endpoint: {exc}"
            ) from exc

        if response.status_code != 200:
            raise SourceUnreachableError(
                f"NSE corporate-announcements returned HTTP "
                f"{response.status_code}"
            )

        try:
            return response.json()
        except ValueError as exc:
            raise SourceUnreachableError(
                "NSE corporate-announcements returned non-JSON (likely a "
                "block page)"
            ) from exc

    def fetch(
        self, symbol: str, start: date, end: date
    ) -> AnnouncementFetchResult:
        """Announcements for one symbol over a date range.

        A per-symbol call, not the market-wide feed: the tracked universe
        is scored per-instrument everywhere else in this codebase, and a
        market-wide pull returns tens of thousands of rows for symbols this
        tool has no instrument row for at all.
        """
        if start > end:
            raise ValueError(f"start {start} is after end {end}")

        payload = self._get(
            {
                "index": "equities",
                "symbol": symbol.upper(),
                "from_date": start.strftime(_QUERY_DATE_FORMAT),
                "to_date": end.strftime(_QUERY_DATE_FORMAT),
            }
        )
        return parse_announcements(payload)


@dataclass(frozen=True)
class AnnouncementIngestReport:
    """Outcome of ingesting one instrument's announcements."""

    symbol: str
    requested_start: date
    requested_end: date
    stored: int
    skipped: int
    #: Earliest announcement actually returned. Reported as information, not
    #: compared against `requested_start` to assert a "gap" -- a symbol
    #: legitimately having no announcements early in the window looks
    #: identical to a truncated response, and this endpoint has not been
    #: found to truncate (see module docstring).
    earliest_returned: date | None


def ingest_announcements(
    db,
    symbol: str,
    instrument_id: int,
    start: date,
    end: date,
    result: AnnouncementFetchResult | None = None,
    client: NseAnnouncementsClient | None = None,
) -> AnnouncementIngestReport:
    """Fetch, parse and store one instrument's announcements for a window.

    Pass `result` (an already-parsed `AnnouncementFetchResult`) to store a
    response obtained another way -- mirrors `ingest_delivery`, and lets
    tests exercise storage and reporting without a network call.
    """
    from algorix.storage import AnnouncementRepository

    if result is None:
        client = client or NseAnnouncementsClient()
        result = client.fetch(symbol, start, end)

    repository = AnnouncementRepository(db)
    stored = repository.upsert_many(
        ((record, instrument_id) for record in result.records),
        source=SOURCE_NSE_ANNOUNCEMENTS,
    )

    return AnnouncementIngestReport(
        symbol=symbol,
        requested_start=start,
        requested_end=end,
        stored=stored,
        skipped=result.skipped,
        earliest_returned=result.earliest,
    )
