"""Gold and silver instruments, plus the currency leg (INDICATORS.md D1-D3).

Why five instruments rather than two: INDICATORS.md D2 makes USD/INR
decomposition mandatory, because an INR-denominated metal price moves on both
the global commodity price and the rupee. Without separating them, "gold looks
strong" may be nothing more than rupee weakness -- a different trade, and one
that says nothing about gold.

So we carry:

- ``GOLDUSD`` / ``SILVERUSD`` (COMEX futures) -- the global commodity leg.
- ``USDINR`` -- the currency leg.
- ``GOLDBEES`` / ``SILVERBEES`` -- NSE-listed, INR-denominated, and actually
  tradeable from an Indian retail account.

Two data realities shaped this, both verified live:

1. **COMEX and FX trade on Indian holidays.** GC=F and SI=F both printed a
   session on 2026-09-14, an NSE holiday. They therefore carry
   ``CalendarPolicy.GLOBAL`` -- coherence is still checked, but NSE session
   membership is not.

2. **The NSE metal ETFs have unreliable recent bars.** On 2026-09-18 both
   GOLDBEES and SILVERBEES returned NaN prices alongside a real volume of
   ~28M. The NaN filter rejects those rows and gap detection reports the
   session as missing, which is why the USD leg is the primary series and the
   ETFs are treated as the tradeable proxy rather than the source of truth.

FX carries no volume (always 0), which `Bar` permits.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from algorix.exceptions import DataUnavailableError
from algorix.models import (
    CalendarPolicy,
    Exchange,
    Instrument,
    InstrumentType,
)
from algorix.storage import BarRepository, Database, InstrumentRepository

GOLD_USD = Instrument(
    symbol="GOLDUSD",
    exchange=Exchange.COMEX,
    instrument_type=InstrumentType.COMMODITY,
    name="Gold Futures (COMEX, USD)",
    yahoo_symbol="GC=F",
    calendar_policy=CalendarPolicy.GLOBAL,
)

SILVER_USD = Instrument(
    symbol="SILVERUSD",
    exchange=Exchange.COMEX,
    instrument_type=InstrumentType.COMMODITY,
    name="Silver Futures (COMEX, USD)",
    yahoo_symbol="SI=F",
    calendar_policy=CalendarPolicy.GLOBAL,
)

USD_INR = Instrument(
    symbol="USDINR",
    exchange=Exchange.FX,
    instrument_type=InstrumentType.CURRENCY,
    name="US Dollar / Indian Rupee",
    yahoo_symbol="USDINR=X",
    calendar_policy=CalendarPolicy.GLOBAL,
)

GOLD_ETF = Instrument(
    symbol="GOLDBEES",
    exchange=Exchange.NSE,
    instrument_type=InstrumentType.COMMODITY,
    name="Nippon India ETF Gold BeES",
    yahoo_symbol="GOLDBEES.NS",
    calendar_policy=CalendarPolicy.NSE,
)

SILVER_ETF = Instrument(
    symbol="SILVERBEES",
    exchange=Exchange.NSE,
    instrument_type=InstrumentType.COMMODITY,
    name="Nippon India ETF Silver BeES",
    yahoo_symbol="SILVERBEES.NS",
    calendar_policy=CalendarPolicy.NSE,
)

#: Every instrument the metals track needs.
METAL_INSTRUMENTS: tuple[Instrument, ...] = (
    GOLD_USD,
    SILVER_USD,
    USD_INR,
    GOLD_ETF,
    SILVER_ETF,
)


def seed_metal_instruments(db: Database) -> dict[str, int]:
    """Register the metals universe. Returns symbol -> instrument id."""
    repository = InstrumentRepository(db)
    return {
        instrument.symbol: repository.upsert(instrument)
        for instrument in METAL_INSTRUMENTS
    }


@dataclass(frozen=True)
class MetalSnapshot:
    """Gold and silver context for one session.

    `gold_inr_implied` is the USD gold price converted at the day's rate. It
    is not a tradeable price -- it exists so that a move in the INR series can
    be attributed to the metal or to the rupee.
    """

    session_date: date
    gold_usd: float
    silver_usd: float
    usd_inr: float

    @property
    def gold_silver_ratio(self) -> float:
        """Ounces of silver per ounce of gold (INDICATORS.md D3).

        A classic relative-value measure: it answers "if buying a precious
        metal, which one?", which neither series answers alone.
        """
        return self.gold_usd / self.silver_usd

    @property
    def gold_inr_implied(self) -> float:
        return self.gold_usd * self.usd_inr

    @property
    def silver_inr_implied(self) -> float:
        return self.silver_usd * self.usd_inr


def metal_snapshot(
    db: Database,
    session_date: date,
    instrument_ids: dict[str, int] | None = None,
) -> MetalSnapshot:
    """Assemble the metals picture for one session.

    Raises DataUnavailableError if any leg is missing -- a snapshot with a
    stale currency rate and a fresh metal price would misattribute the move,
    which is precisely what D2 exists to prevent.
    """
    ids = instrument_ids or {
        instrument.symbol: _require_id(db, instrument)
        for instrument in (GOLD_USD, SILVER_USD, USD_INR)
    }
    repository = BarRepository(db)

    closes: dict[str, float] = {}
    for symbol in ("GOLDUSD", "SILVERUSD", "USDINR"):
        instrument_id = ids.get(symbol)
        if instrument_id is None:
            raise DataUnavailableError(f"{symbol} is not registered")
        bars = repository.get_range(instrument_id, session_date, session_date)
        if not bars:
            raise DataUnavailableError(
                f"No {symbol} bar for {session_date}; refusing to build a "
                "metals snapshot from mismatched sessions"
            )
        closes[symbol] = bars[0].close

    return MetalSnapshot(
        session_date=session_date,
        gold_usd=closes["GOLDUSD"],
        silver_usd=closes["SILVERUSD"],
        usd_inr=closes["USDINR"],
    )


def _require_id(db: Database, instrument: Instrument) -> int | None:
    stored = InstrumentRepository(db).get(instrument.symbol, instrument.exchange)
    return stored.id if stored else None
