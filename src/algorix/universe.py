"""Index universe: who is in the Nifty 50, and who was in it when.

Constituency is tracked point-in-time. Storing only "the current 50" would
bake survivorship bias into every future backtest -- the index would appear to
have always held whatever is winning today, and any historical score would be
computed against a universe that did not exist at the time.

Fetching is separated from parsing on purpose. Parsing and sync logic are
pure and fully testable offline; only `NseIndexClient` touches the network.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date

import requests

from algorix.exceptions import (
    DataIntegrityError,
    DataUnavailableError,
    SourceUnreachableError,
)
from algorix.models import Exchange, Instrument, InstrumentType
from algorix.storage import Database, InstrumentRepository, _utc_now_iso

NIFTY_50 = "NIFTY50"
NIFTY_200 = "NIFTY200"
NIFTY_500 = "NIFTY500"
NIFTY_MIDCAP_150 = "NIFTYMIDCAP150"

_INDEX_BASE = "https://nsearchives.nseindia.com/content/indices/{slug}.csv"

#: Index code -> NSE constituent-list slug.
#:
#: Universe width is not cosmetic. Cross-sectional momentum is a *dispersion*
#: effect: it ranks winners against losers, and needs a wide spread to have
#: anything to rank. Fifty mega-caps that largely move together provide
#: little, which is why the published momentum research (the Nifty 200
#: Momentum 30 index) draws from 200 names rather than 50. Backtesting on
#: NIFTY50 measures momentum where it is weakest.
INDEX_SLUGS: dict[str, str] = {
    NIFTY_50: "ind_nifty50list",
    NIFTY_200: "ind_nifty200list",
    NIFTY_500: "ind_nifty500list",
    NIFTY_MIDCAP_150: "ind_niftymidcap150list",
}

NIFTY_50_CSV_URL = _INDEX_BASE.format(slug=INDEX_SLUGS[NIFTY_50])


def index_url(index_symbol: str) -> str:
    """Constituent-list URL for a supported index."""
    slug = INDEX_SLUGS.get(index_symbol.upper())
    if slug is None:
        raise ValueError(
            f"Unknown index {index_symbol!r}. Known: {sorted(INDEX_SLUGS)}"
        )
    return _INDEX_BASE.format(slug=slug)

# NSE rejects unadorned clients. Without a browser-like User-Agent it returns
# an HTML block page with HTTP 200 -- see _guard_against_html below.
_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "text/csv,application/csv,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

_REQUIRED_COLUMNS = {"Symbol", "Company Name"}


@dataclass(frozen=True)
class ConstituentRecord:
    """One row of an NSE index constituent list."""

    symbol: str
    name: str
    industry: str | None = None
    isin: str | None = None
    series: str | None = None

    def to_instrument(self) -> Instrument:
        return Instrument(
            symbol=self.symbol,
            exchange=Exchange.NSE,
            instrument_type=InstrumentType.EQUITY,
            name=self.name,
            # yfinance addresses NSE equities with a .NS suffix.
            yahoo_symbol=f"{self.symbol}.NS",
        )


@dataclass(frozen=True)
class SyncResult:
    """What changed when constituency was last synced."""

    added: list[str]
    removed: list[str]
    unchanged: list[str]

    @property
    def changed(self) -> bool:
        return bool(self.added or self.removed)


def _guard_against_html(text: str) -> None:
    """Reject an HTML page served in place of the CSV.

    NSE answers blocked or rate-limited requests with an HTML block page and
    HTTP 200. Feeding that to a CSV parser yields plausible-looking garbage
    symbols, so this must fail loudly rather than parse.
    """
    head = text.lstrip()[:200].lower()
    if head.startswith("<!doctype html") or head.startswith("<html") or "<body" in head:
        raise SourceUnreachableError(
            "NSE returned an HTML page instead of CSV -- the request was most "
            "likely blocked or rate-limited. Not parsing it as data."
        )


def parse_constituents_csv(text: str) -> list[ConstituentRecord]:
    """Parse an NSE index constituent CSV.

    Raises:
        SourceUnreachableError: an HTML block page was served instead of data.
        DataUnavailableError: the CSV is empty or has no data rows.
        DataIntegrityError: expected columns are missing, or a row is unusable.
    """
    if not text or not text.strip():
        raise DataUnavailableError("Constituent CSV is empty")

    _guard_against_html(text)

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise DataUnavailableError("Constituent CSV has no header row")

    columns = {name.strip() for name in reader.fieldnames}
    missing = _REQUIRED_COLUMNS - columns
    if missing:
        raise DataIntegrityError(
            f"Constituent CSV is missing required column(s): {sorted(missing)}. "
            f"Got: {sorted(columns)}"
        )

    records: list[ConstituentRecord] = []
    seen: set[str] = set()

    for line_no, row in enumerate(reader, start=2):
        clean = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
        symbol = clean.get("Symbol", "").upper()
        if not symbol:
            # A trailing blank line is normal; a blank symbol mid-file is not,
            # but either way the row carries no usable instrument.
            continue
        if symbol in seen:
            raise DataIntegrityError(
                f"Duplicate symbol {symbol!r} at line {line_no} of constituent CSV"
            )
        seen.add(symbol)

        records.append(
            ConstituentRecord(
                symbol=symbol,
                name=clean.get("Company Name", "") or symbol,
                industry=clean.get("Industry") or None,
                isin=clean.get("ISIN Code") or None,
                series=clean.get("Series") or None,
            )
        )

    if not records:
        raise DataUnavailableError("Constituent CSV contained no usable rows")

    return records


class NseIndexClient:
    """Fetches index constituent lists from NSE. The only networked piece."""

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        self.timeout_seconds = timeout_seconds

    def fetch_nifty_50(self) -> list[ConstituentRecord]:
        return self.fetch_index(NIFTY_50)

    def fetch_index(self, index_symbol: str) -> list[ConstituentRecord]:
        """Fetch any supported index's constituent list."""
        return self.fetch(index_url(index_symbol))

    def fetch(self, url: str) -> list[ConstituentRecord]:
        try:
            response = requests.get(
                url, headers=_NSE_HEADERS, timeout=self.timeout_seconds
            )
        except requests.RequestException as exc:
            raise SourceUnreachableError(f"Could not reach {url}: {exc}") from exc

        if response.status_code != 200:
            raise SourceUnreachableError(
                f"{url} returned HTTP {response.status_code}"
            )

        return parse_constituents_csv(response.text)


class ConstituencyRepository:
    """Point-in-time index membership."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self.instruments = InstrumentRepository(db)

    def sync(
        self,
        index_symbol: str,
        records: Sequence[ConstituentRecord],
        effective_date: date,
    ) -> SyncResult:
        """Reconcile stored membership against a freshly fetched list.

        New symbols open a membership period at `effective_date`; departed
        symbols have theirs closed at the same date. Unchanged symbols are
        left alone, so a re-run on the same day is a no-op.

        Refuses an empty list: a fetch that silently returned nothing would
        otherwise wipe the entire universe.
        """
        if not records:
            raise DataUnavailableError(
                f"Refusing to sync {index_symbol} with an empty constituent "
                "list -- this would deactivate the whole universe"
            )

        incoming = {r.symbol: r for r in records}
        existing = set(self.current_constituents(index_symbol, on=effective_date))

        added = sorted(set(incoming) - existing)
        removed = sorted(existing - set(incoming))
        unchanged = sorted(existing & set(incoming))

        for symbol in added:
            instrument_id = self.instruments.upsert(incoming[symbol].to_instrument())
            self._open_membership(index_symbol, instrument_id, effective_date)

        for symbol in removed:
            self._close_membership(index_symbol, symbol, effective_date)

        return SyncResult(added=added, removed=removed, unchanged=unchanged)

    def _open_membership(
        self, index_symbol: str, instrument_id: int, effective_from: date
    ) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO index_constituents
                    (index_symbol, instrument_id, effective_from, effective_to,
                     recorded_at)
                VALUES (?, ?, ?, NULL, ?)
                ON CONFLICT (index_symbol, instrument_id, effective_from)
                DO NOTHING
                """,
                (
                    index_symbol,
                    instrument_id,
                    effective_from.isoformat(),
                    _utc_now_iso(),
                ),
            )

    def _close_membership(
        self, index_symbol: str, symbol: str, effective_to: date
    ) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                UPDATE index_constituents
                SET effective_to = ?
                WHERE index_symbol = ?
                  AND effective_to IS NULL
                  AND instrument_id = (
                      SELECT id FROM instruments
                      WHERE symbol = ? AND exchange = ?
                  )
                """,
                (effective_to.isoformat(), index_symbol, symbol, str(Exchange.NSE)),
            )

    def current_constituents(
        self, index_symbol: str, on: date | None = None
    ) -> list[str]:
        """Symbols in the index on `on` (default: the latest known state).

        `effective_to` is exclusive -- a period ending on D does not include D.
        """
        if on is None:
            query = """
                SELECT i.symbol FROM index_constituents c
                JOIN instruments i ON i.id = c.instrument_id
                WHERE c.index_symbol = ? AND c.effective_to IS NULL
                ORDER BY i.symbol
            """
            params: tuple[object, ...] = (index_symbol,)
        else:
            query = """
                SELECT i.symbol FROM index_constituents c
                JOIN instruments i ON i.id = c.instrument_id
                WHERE c.index_symbol = ?
                  AND c.effective_from <= ?
                  AND (c.effective_to IS NULL OR c.effective_to > ?)
                ORDER BY i.symbol
            """
            iso = on.isoformat()
            params = (index_symbol, iso, iso)

        with self.db.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [r["symbol"] for r in rows]

    def instrument_ids(
        self, index_symbol: str, on: date | None = None
    ) -> list[int]:
        """Instrument ids for the index membership on `on`."""
        symbols = self.current_constituents(index_symbol, on=on)
        if not symbols:
            return []

        placeholders = ",".join("?" * len(symbols))
        with self.db.connect() as conn:
            rows = conn.execute(
                f"SELECT id FROM instruments WHERE exchange = ? "
                f"AND symbol IN ({placeholders}) ORDER BY symbol",
                (str(Exchange.NSE), *symbols),
            ).fetchall()
        return [int(r["id"]) for r in rows]


def sync_index(
    db: Database,
    index_symbol: str,
    effective_date: date,
    client: NseIndexClient | None = None,
    records: Iterable[ConstituentRecord] | None = None,
) -> SyncResult:
    """Fetch and store an index's current membership.

    `records` allows an already-fetched list to be supplied, keeping this
    usable offline and in tests.
    """
    if records is None:
        client = client or NseIndexClient()
        records = client.fetch_index(index_symbol)

    repo = ConstituencyRepository(db)
    return repo.sync(index_symbol, list(records), effective_date)


def sync_nifty_50(
    db: Database,
    effective_date: date,
    client: NseIndexClient | None = None,
    records: Iterable[ConstituentRecord] | None = None,
) -> SyncResult:
    """Fetch and store the current Nifty 50 membership."""
    return sync_index(db, NIFTY_50, effective_date, client, records)
