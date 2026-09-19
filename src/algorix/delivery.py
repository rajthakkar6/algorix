"""NSE delivery data ingestion (INDICATORS.md A6).

Delivery percentage separates genuine accumulation from intraday churn, and
it is the most differentiated signal available to an Indian-market tool --
most exchanges do not publish it at all.

Source is NSE's full bhavcopy: one CSV per session covering the whole market,
published only after close.

Real quirks of that file, all verified against live data:

- **Header and values are whitespace-padded** (`" SERIES"`, `" 18-Sep-2026"`).
  Parsing without stripping silently yields empty columns.
- **Missing delivery is written as `'-'`**, not as an empty field -- 275 of
  3508 rows on a sample session, mostly the BE series. These must be recorded
  as *unavailable*, never as zero: zero delivery asserts nobody took delivery,
  which is a different and false claim.
- **One file holds every series** (EQ, BE, SM, ST, GS...). Only the requested
  series is kept; the rest are different instrument classes.
- **The file does not exist for non-trading days**, and is absent until some
  time after close on a trading day. A 404 is therefore normal, not a fault.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from datetime import date

import requests

from algorix.exceptions import (
    DataIntegrityError,
    DataUnavailableError,
    SourceUnreachableError,
)
from algorix.models import DeliveryRecord

SOURCE_BHAVCOPY = "nse-bhavcopy"

BHAVCOPY_URL_TEMPLATE = (
    "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv"
)

_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "text/csv,application/csv,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

#: Equity series. EQ is the rolling-settlement equity segment we score.
DEFAULT_SERIES = ("EQ",)

_REQUIRED_COLUMNS = {"SYMBOL", "SERIES", "DATE1", "TTL_TRD_QNTY", "DELIV_QTY"}

#: NSE writes the date as 18-Sep-2026.
_DATE_FORMAT = "%d-%b-%Y"


@dataclass(frozen=True)
class BhavcopyResult:
    """Parsed delivery data for one session."""

    session_date: date
    records: dict[str, DeliveryRecord] = field(default_factory=dict)
    #: Symbols present in the file whose delivery figure was not published.
    unavailable: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)


def build_bhavcopy_url(session_date: date) -> str:
    return BHAVCOPY_URL_TEMPLATE.format(ddmmyyyy=session_date.strftime("%d%m%Y"))


def _guard_against_html(text: str) -> None:
    head = text.lstrip()[:200].lower()
    if head.startswith("<!doctype html") or head.startswith("<html") or "<body" in head:
        raise SourceUnreachableError(
            "NSE returned an HTML page instead of the bhavcopy CSV -- the "
            "request was most likely blocked or rate-limited."
        )


def _parse_quantity(raw: str) -> int | None:
    """Parse a quantity, returning None when NSE published no value.

    NSE writes an unpublished figure as `'-'` (quotes included). Any value
    that is not a number means "not published", which is not zero.
    """
    if raw is None:
        return None
    cleaned = raw.strip().strip("'\"").strip()
    if not cleaned or cleaned == "-":
        return None
    try:
        return int(float(cleaned.replace(",", "")))
    except ValueError:
        return None


def parse_bhavcopy(
    text: str,
    expected_date: date | None = None,
    series: tuple[str, ...] = DEFAULT_SERIES,
) -> BhavcopyResult:
    """Parse an NSE full bhavcopy CSV into delivery records.

    Args:
        text: raw CSV.
        expected_date: if given, the file's own DATE1 must match. Guards
            against silently ingesting the wrong session's file under today's
            date -- which would misattribute an entire market's delivery data.
        series: which series to keep.
    """
    if not text or not text.strip():
        raise DataUnavailableError("Bhavcopy is empty")

    _guard_against_html(text)

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise DataUnavailableError("Bhavcopy has no header row")

    columns = {name.strip().upper() for name in reader.fieldnames}
    missing = _REQUIRED_COLUMNS - columns
    if missing:
        raise DataIntegrityError(
            f"Bhavcopy is missing required column(s): {sorted(missing)}. "
            f"Got: {sorted(columns)}"
        )

    wanted = {s.upper() for s in series}
    records: dict[str, DeliveryRecord] = {}
    unavailable: list[str] = []
    file_date: date | None = None

    for row in reader:
        clean = {
            (k or "").strip().upper(): (v or "").strip() for k, v in row.items()
        }

        if clean.get("SERIES", "").upper() not in wanted:
            continue

        symbol = clean.get("SYMBOL", "").upper()
        if not symbol:
            continue

        row_date = _parse_date(clean.get("DATE1", ""))
        if row_date is None:
            raise DataIntegrityError(
                f"Bhavcopy row for {symbol} has an unparseable date: "
                f"{clean.get('DATE1')!r}"
            )
        if file_date is None:
            file_date = row_date

        traded = _parse_quantity(clean.get("TTL_TRD_QNTY", ""))
        delivered = _parse_quantity(clean.get("DELIV_QTY", ""))

        if traded is None or delivered is None:
            # Not published for this security -- recorded, never inferred.
            unavailable.append(symbol)
            continue

        try:
            records[symbol] = DeliveryRecord(
                session_date=row_date,
                traded_quantity=traded,
                delivered_quantity=delivered,
            )
        except DataIntegrityError:
            unavailable.append(symbol)

    if file_date is None:
        raise DataUnavailableError(
            f"Bhavcopy contained no rows for series {sorted(wanted)}"
        )

    if expected_date is not None and file_date != expected_date:
        raise DataIntegrityError(
            f"Bhavcopy is for {file_date}, expected {expected_date}. Refusing "
            "to attribute one session's delivery data to another."
        )

    return BhavcopyResult(
        session_date=file_date, records=records, unavailable=sorted(set(unavailable))
    )


def _parse_date(raw: str) -> date | None:
    from datetime import datetime

    cleaned = raw.strip()
    if not cleaned:
        return None
    try:
        return datetime.strptime(cleaned, _DATE_FORMAT).date()
    except ValueError:
        return None


class NseBhavcopyClient:
    """Fetches the daily bhavcopy. The only networked piece."""

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        self.timeout_seconds = timeout_seconds

    def fetch(
        self, session_date: date, series: tuple[str, ...] = DEFAULT_SERIES
    ) -> BhavcopyResult:
        url = build_bhavcopy_url(session_date)

        try:
            response = requests.get(
                url, headers=_NSE_HEADERS, timeout=self.timeout_seconds
            )
        except requests.RequestException as exc:
            raise SourceUnreachableError(f"Could not reach {url}: {exc}") from exc

        if response.status_code == 404:
            # Normal, not a fault: no file exists for a non-trading day, and
            # the file appears only some time after close on a trading day.
            raise DataUnavailableError(
                f"No bhavcopy published for {session_date} (HTTP 404). It may "
                "be a non-trading day, or the file may not be out yet."
            )
        if response.status_code != 200:
            raise SourceUnreachableError(
                f"{url} returned HTTP {response.status_code}"
            )

        return parse_bhavcopy(response.text, expected_date=session_date, series=series)


@dataclass(frozen=True)
class DeliveryIngestReport:
    """Outcome of storing one session's delivery data."""

    session_date: date
    stored: int
    #: Requested symbols with no delivery figure in the file.
    unavailable: list[str]
    #: Requested symbols absent from the file entirely.
    not_in_file: list[str]

    @property
    def is_clean(self) -> bool:
        return not self.unavailable and not self.not_in_file


def ingest_delivery(
    db,
    session_date: date,
    symbol_to_instrument_id: dict[str, int],
    result: BhavcopyResult | None = None,
    client: NseBhavcopyClient | None = None,
) -> DeliveryIngestReport:
    """Store delivery data for the given symbols.

    Symbols missing from the file, or present without a delivery figure, are
    reported rather than stored as zero.
    """
    from algorix.storage import DeliveryRepository

    if result is None:
        client = client or NseBhavcopyClient()
        result = client.fetch(session_date)

    repository = DeliveryRepository(db)
    stored = 0
    not_in_file: list[str] = []
    unavailable: list[str] = []

    for symbol, instrument_id in symbol_to_instrument_id.items():
        record = result.records.get(symbol)
        if record is None:
            if symbol in result.unavailable:
                unavailable.append(symbol)
            else:
                not_in_file.append(symbol)
            continue
        stored += repository.upsert_many(
            instrument_id, [record], source=SOURCE_BHAVCOPY
        )

    return DeliveryIngestReport(
        session_date=result.session_date,
        stored=stored,
        unavailable=sorted(unavailable),
        not_in_file=sorted(not_in_file),
    )
