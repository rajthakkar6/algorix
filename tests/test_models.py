"""Tests for domain model validation."""

from datetime import date

import pytest

from algorix.exceptions import DataIntegrityError
from algorix.models import Bar, DeliveryRecord, Exchange, Instrument, InstrumentType

SESSION = date(2026, 9, 18)


# --------------------------------------------------------------------------
# Instrument -- positive
# --------------------------------------------------------------------------


def test_instrument_accepts_canonical_symbol():
    inst = Instrument(
        symbol="RELIANCE",
        exchange=Exchange.NSE,
        instrument_type=InstrumentType.EQUITY,
        yahoo_symbol="RELIANCE.NS",
    )

    assert inst.symbol == "RELIANCE"
    assert inst.is_active is True


def test_instrument_keeps_vendor_symbol_separate():
    """Vendor syntax must not leak into the canonical symbol."""
    inst = Instrument(
        symbol="TCS",
        exchange=Exchange.NSE,
        instrument_type=InstrumentType.EQUITY,
        yahoo_symbol="TCS.NS",
    )

    assert inst.symbol == "TCS"
    assert inst.yahoo_symbol == "TCS.NS"


# -- Instrument -- negative -------------------------------------------------


def test_empty_symbol_is_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        Instrument(
            symbol="",
            exchange=Exchange.NSE,
            instrument_type=InstrumentType.EQUITY,
        )


def test_whitespace_symbol_is_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        Instrument(
            symbol="   ",
            exchange=Exchange.NSE,
            instrument_type=InstrumentType.EQUITY,
        )


def test_lowercase_symbol_is_rejected():
    """Same instrument under two identities is a silent duplication bug."""
    with pytest.raises(ValueError, match="upper-case"):
        Instrument(
            symbol="reliance",
            exchange=Exchange.NSE,
            instrument_type=InstrumentType.EQUITY,
        )


def test_padded_symbol_is_rejected():
    with pytest.raises(ValueError, match="unpadded"):
        Instrument(
            symbol=" RELIANCE ",
            exchange=Exchange.NSE,
            instrument_type=InstrumentType.EQUITY,
        )


# --------------------------------------------------------------------------
# Bar -- positive
# --------------------------------------------------------------------------


def test_valid_bar_is_accepted():
    bar = Bar(
        session_date=SESSION, open=100.0, high=105.0, low=99.0, close=103.0, volume=5000
    )

    assert bar.close == 103.0


def test_flat_bar_is_valid():
    """All four prices equal -- a legitimate circuit-locked or untraded session."""
    bar = Bar(
        session_date=SESSION, open=100.0, high=100.0, low=100.0, close=100.0, volume=0
    )

    assert bar.high == bar.low


def test_zero_volume_is_allowed():
    """An illiquid name can genuinely trade nothing in a session."""
    bar = Bar(
        session_date=SESSION, open=50.0, high=51.0, low=49.0, close=50.5, volume=0
    )

    assert bar.volume == 0


# -- Bar -- negative --------------------------------------------------------


def test_high_below_low_is_rejected():
    with pytest.raises(DataIntegrityError, match="is below low"):
        Bar(
            session_date=SESSION,
            open=100.0,
            high=98.0,
            low=99.0,
            close=98.5,
            volume=100,
        )


def test_high_below_close_is_rejected():
    with pytest.raises(DataIntegrityError, match="high .* is below"):
        Bar(
            session_date=SESSION,
            open=100.0,
            high=102.0,
            low=99.0,
            close=103.0,
            volume=100,
        )


def test_low_above_open_is_rejected():
    with pytest.raises(DataIntegrityError, match="low .* is above"):
        Bar(
            session_date=SESSION,
            open=100.0,
            high=105.0,
            low=101.0,
            close=103.0,
            volume=100,
        )


def test_zero_price_is_rejected():
    with pytest.raises(DataIntegrityError, match="must be positive"):
        Bar(
            session_date=SESSION, open=0.0, high=105.0, low=99.0, close=103.0, volume=1
        )


def test_negative_price_is_rejected():
    with pytest.raises(DataIntegrityError, match="must be positive"):
        Bar(
            session_date=SESSION,
            open=-10.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1,
        )


def test_none_price_is_rejected():
    """yfinance returns NaN/None for gaps -- must not reach storage."""
    with pytest.raises(DataIntegrityError, match="is missing"):
        Bar(
            session_date=SESSION,
            open=None,  # type: ignore[arg-type]
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1,
        )


def test_negative_volume_is_rejected():
    with pytest.raises(DataIntegrityError, match="volume cannot be negative"):
        Bar(
            session_date=SESSION,
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=-5,
        )


# --------------------------------------------------------------------------
# DeliveryRecord
# --------------------------------------------------------------------------


def test_delivery_pct_is_computed():
    rec = DeliveryRecord(
        session_date=SESSION, traded_quantity=1000, delivered_quantity=450
    )

    assert rec.delivery_pct == pytest.approx(45.0)


def test_full_delivery_is_valid():
    rec = DeliveryRecord(
        session_date=SESSION, traded_quantity=1000, delivered_quantity=1000
    )

    assert rec.delivery_pct == pytest.approx(100.0)


def test_delivery_pct_is_none_when_nothing_traded():
    """0/0 is undefined. Returning 0.0 would assert "nobody took delivery"."""
    rec = DeliveryRecord(
        session_date=SESSION, traded_quantity=0, delivered_quantity=0
    )

    assert rec.delivery_pct is None


def test_delivered_exceeding_traded_is_rejected():
    with pytest.raises(DataIntegrityError, match="exceeds traded"):
        DeliveryRecord(
            session_date=SESSION, traded_quantity=100, delivered_quantity=150
        )


def test_negative_traded_quantity_is_rejected():
    with pytest.raises(DataIntegrityError, match="traded_quantity cannot be negative"):
        DeliveryRecord(
            session_date=SESSION, traded_quantity=-1, delivered_quantity=0
        )


def test_negative_delivered_quantity_is_rejected():
    with pytest.raises(
        DataIntegrityError, match="delivered_quantity cannot be negative"
    ):
        DeliveryRecord(
            session_date=SESSION, traded_quantity=100, delivered_quantity=-1
        )
