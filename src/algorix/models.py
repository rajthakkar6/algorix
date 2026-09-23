"""Core domain types.

Validation lives here as well as in the database schema. The schema is the
backstop that makes bad data unstorable; these checks exist to fail earlier,
with a message that says which instrument and which session went wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
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
