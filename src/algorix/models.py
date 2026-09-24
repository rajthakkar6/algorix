"""Core domain types.

Validation lives here as well as in the database schema. The schema is the
backstop that makes bad data unstorable; these checks exist to fail earlier,
with a message that says which instrument and which session went wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum

from algorix.exceptions import DataIntegrityError


class Exchange(StrEnum):
    NSE = "NSE"
    BSE = "BSE"
    MCX = "MCX"
    COMEX = "COMEX"
    FX = "FX"


class InstrumentType(StrEnum):
    EQUITY = "EQUITY"
    COMMODITY = "COMMODITY"
    INDEX = "INDEX"
    CURRENCY = "CURRENCY"


class CalendarPolicy(StrEnum):
    """Which trading calendar an instrument's sessions must conform to.

    NSE-listed instruments are validated against the NSE calendar, which is
    what rejects the phantom bars yfinance emits on Indian holidays.

    COMEX futures and FX are genuinely open on days NSE is closed -- verified
    live: GC=F and SI=F both traded on 2026-09-14, an NSE holiday. Applying
    the NSE calendar to them would discard real data, so they are checked for
    coherence and completeness but not for session membership.
    """

    NSE = "NSE"
    GLOBAL = "GLOBAL"


@dataclass(frozen=True)
class Instrument:
    """A tradeable or trackable series.

    `symbol` is the canonical exchange symbol (e.g. "RELIANCE"). Vendor-
    specific identifiers are kept separately -- yfinance wants "RELIANCE.NS",
    and mixing vendor syntax into the canonical symbol makes the same
    instrument arrive under two identities.
    """

    symbol: str
    exchange: Exchange
    instrument_type: InstrumentType
    name: str | None = None
    yahoo_symbol: str | None = None
    is_active: bool = True
    calendar_policy: CalendarPolicy = CalendarPolicy.NSE
    #: NSE's own sector classification (e.g. "Information Technology"),
    #: carried through from the index constituent feed. None for instruments
    #: that don't have one (metals, indices, FX) or haven't synced yet.
    industry: str | None = None
    id: int | None = None

    def __post_init__(self) -> None:
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol cannot be empty")
        if self.symbol != self.symbol.strip().upper():
            raise ValueError(
                f"symbol must be upper-case and unpadded, got {self.symbol!r}"
            )


@dataclass(frozen=True)
class Bar:
    """One session's OHLCV for one instrument.

    Volume of zero is permitted: an illiquid name can genuinely trade nothing
    in a session. Prices of zero are not -- that is always bad data.
    """

    session_date: date
    open: float
    high: float
    low: float
    close: float
    volume: int

    def __post_init__(self) -> None:
        prices = {
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
        }
        for field, value in prices.items():
            if value is None:
                raise DataIntegrityError(
                    f"{self.session_date}: {field} is missing"
                )
            if value <= 0:
                raise DataIntegrityError(
                    f"{self.session_date}: {field} must be positive, got {value}"
                )

        if self.volume < 0:
            raise DataIntegrityError(
                f"{self.session_date}: volume cannot be negative, got {self.volume}"
            )

        if self.high < self.low:
            raise DataIntegrityError(
                f"{self.session_date}: high {self.high} is below low {self.low}"
            )
        if self.high < self.open or self.high < self.close:
            raise DataIntegrityError(
                f"{self.session_date}: high {self.high} is below open "
                f"{self.open} or close {self.close}"
            )
        if self.low > self.open or self.low > self.close:
            raise DataIntegrityError(
                f"{self.session_date}: low {self.low} is above open "
                f"{self.open} or close {self.close}"
            )


@dataclass(frozen=True)
class DeliveryRecord:
    """NSE delivery data for one session.

    Delivery percentage separates genuine accumulation from intraday churn --
    see INDICATORS.md A6. It is published only after close, and not for every
    series, so absence is normal and must not be read as zero.
    """

    session_date: date
    traded_quantity: int
    delivered_quantity: int

    def __post_init__(self) -> None:
        if self.traded_quantity < 0:
            raise DataIntegrityError(
                f"{self.session_date}: traded_quantity cannot be negative"
            )
        if self.delivered_quantity < 0:
            raise DataIntegrityError(
                f"{self.session_date}: delivered_quantity cannot be negative"
            )
        if self.delivered_quantity > self.traded_quantity:
            raise DataIntegrityError(
                f"{self.session_date}: delivered {self.delivered_quantity} "
                f"exceeds traded {self.traded_quantity}"
            )

    @property
    def delivery_pct(self) -> float | None:
        """Delivered as a percentage of traded.

        None when nothing traded -- 0/0 is undefined, and returning 0.0 would
        read as "nobody took delivery", which is a different claim.
        """
        if self.traded_quantity == 0:
            return None
        return (self.delivered_quantity / self.traded_quantity) * 100.0


@dataclass(frozen=True)
class AnnouncementRecord:
    """One NSE corporate announcement (INDICATORS.md G1).

    The highest-quality news source available to this tool -- official,
    structured, timestamped, free, and impossible to astroturf. This is raw
    structured data only: G0 forbids using it as a directional score input,
    and no sentiment/materiality judgement is made here. That is a later,
    separate concern (G4) with real per-item LLM cost.

    `seq_id` is NSE's own identifier for the announcement and is globally
    unique, so it is the natural idempotency key for storage -- the same
    announcement fetched twice (an inevitable consequence of overlapping
    date-range refetches) upserts rather than duplicates.
    """

    seq_id: str
    symbol: str
    announced_at: datetime
    category: str
    text: str
    attachment_url: str | None = None
    isin: str | None = None

    def __post_init__(self) -> None:
        if not self.seq_id or not self.seq_id.strip():
            raise DataIntegrityError("announcement seq_id cannot be empty")
        if not self.symbol or not self.symbol.strip():
            raise DataIntegrityError(
                f"announcement {self.seq_id}: symbol cannot be empty"
            )
        if not self.text or not self.text.strip():
            # An announcement with no text is not a smaller announcement --
            # it is a parse failure. NSE's own feed always carries this.
            raise DataIntegrityError(
                f"announcement {self.seq_id}: text cannot be empty"
            )


class EventType(StrEnum):
    """Consolidated event categories for G4 extraction.

    NSE's own `desc` field on an announcement (AnnouncementRecord.category)
    is already a category, but a fine-grained and inconsistent one -- a
    30-day live sample across 15 Nifty 50 names turned up ~28 distinct
    values ("Updates", "General Updates" and "Analysts/Institutional
    Investor Meet/Con. Call Updates" all separately, for instance). This
    enum is what the extraction consolidates *into*: a small, closed,
    consistent set an LLM can classify reliably and a digest can group by.
    Grounded in that same sample, not invented.
    """

    EARNINGS_OR_RESULTS = "earnings_or_results"
    CORPORATE_ACTION = "corporate_action"
    MANAGEMENT_CHANGE = "management_change"
    REGULATORY_OR_LEGAL = "regulatory_or_legal"
    MERGER_ACQUISITION = "merger_acquisition"
    BUSINESS_UPDATE = "business_update"
    CREDIT_RATING = "credit_rating"
    SHAREHOLDER_MEETING = "shareholder_meeting"
    ADMINISTRATIVE = "administrative"
    OTHER = "other"


class Polarity(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class Materiality(StrEnum):
    """How significant this event is to the business -- not how the market
    might react to it. A calm-toned but structurally important disclosure
    (auditor resignation) can be HIGH; an emphatic press release about a
    routine matter can be LOW. Three levels, not five: nothing here has
    been calibrated against outcomes yet, so more granularity would be
    false precision (see the equal-weighting rationale in scoring.py)."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class ExtractedEvent:
    """One announcement's structured judgement (INDICATORS.md G4).

    **Never a score input** -- G0 is explicit that sentiment/news must never
    become a directional score contributor. This exists for event
    detection, risk flags, and contrarian-extreme context only (G0's three
    approved uses), surfaced in the digest/journal as a separate, clearly
    labelled layer alongside the quant score, never blended into it
    (mirrors CLAUDE.md invariant 1's rule for the LLM concluder -- the same
    separation applies here, one step earlier in the pipeline).

    One extraction per announcement (`announcement_seq_id` is the source
    row's own primary key), so storage can upsert on it the same way.
    `model_id` and `prompt_version` are recorded with every row -- CLAUDE.md
    invariant 7: without them, months of extracted events are uninterpretable
    because nothing says what changed underneath them.
    """

    announcement_seq_id: str
    event_type: EventType
    entities: list[str]
    polarity: Polarity
    materiality: Materiality
    #: Fixed "official" for G1 (NSE filings). The field exists now, not
    #: because G1 varies, so G2/G3 (mainstream news, social) slot into the
    #: same schema later without a migration.
    source_credibility: str
    risk_flag: bool
    risk_reason: str | None
    model_id: str
    prompt_version: int
    extracted_at: datetime

    def __post_init__(self) -> None:
        if not self.announcement_seq_id or not self.announcement_seq_id.strip():
            raise DataIntegrityError(
                "extracted event: announcement_seq_id cannot be empty"
            )
        if self.risk_flag and not (self.risk_reason or "").strip():
            raise DataIntegrityError(
                f"{self.announcement_seq_id}: risk_flag is set but "
                "risk_reason is empty -- a risk flag without a reason is "
                "not traceable to anything"
            )


@dataclass(frozen=True)
class EarningsSurpriseRecord:
    """One reported quarter's earnings vs. its estimate (INDICATORS.md A8,
    Post-Earnings Announcement Drift).

    Only completed reports are represented -- a future consensus estimate
    with no actual yet is not a surprise, so it is filtered out before this
    is ever constructed (see `earnings.py`'s parser). `surprise_pct` is the
    value scored on; `eps_estimate`/`eps_actual` are kept only for
    explainability (what a stock's A8 line in the digest is based on).
    """

    symbol: str
    report_date: date
    eps_estimate: float
    eps_actual: float
    surprise_pct: float

    def __post_init__(self) -> None:
        if not self.symbol or not self.symbol.strip():
            raise DataIntegrityError("earnings surprise: symbol cannot be empty")
