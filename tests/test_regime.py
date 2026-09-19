"""Tests for market regime gates."""

from datetime import date, timedelta

import pytest

from algorix.calendar import TradingCalendar
from algorix.exceptions import DataIntegrityError, DataUnavailableError
from algorix.models import Bar
from algorix.regime import (
    BREADTH_WEAK,
    EFFICIENCY_CHOPPY,
    VIX_PERCENTILE_WINDOW,
    FlowRecord,
    FlowRepository,
    MarketRegime,
    RegimeVerdict,
    assess_regime,
    efficiency_ratio,
    market_breadth,
    parse_flows,
    percentile_of_latest,
    seed_regime_instruments,
)
from algorix.series import IndicatorValue, PriceSeries
from algorix.storage import Database

AS_OF = date(2026, 9, 18)


def series_from(closes: list[float], instrument_id: int = 1) -> PriceSeries:
    start = date(2020, 1, 1)
    bars = tuple(
        Bar(
            session_date=start + timedelta(days=i),
            open=c,
            high=c + 0.5,
            low=c - 0.5,
            close=c,
            volume=1000,
        )
        for i, c in enumerate(closes)
    )
    return PriceSeries(instrument_id=instrument_id, as_of=bars[-1].session_date, bars=bars)


def universe(above: int, below: int, length: int = 80) -> dict[str, PriceSeries]:
    """Build a universe with a known number above/below their 50 DMA."""
    out = {}
    for i in range(above):
        out[f"UP{i}"] = series_from([100.0] * (length - 1) + [200.0], i + 1)
    for i in range(below):
        out[f"DOWN{i}"] = series_from([100.0] * (length - 1) + [50.0], 1000 + i)
    return out


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "test.db")
    database.migrate()
    return database


# ---------------------------------------------------------------------------
# B1 -- breadth
# ---------------------------------------------------------------------------


def test_all_above_is_full_breadth():
    assert market_breadth(universe(above=20, below=0)).value == pytest.approx(100.0)


def test_none_above_is_zero_breadth():
    assert market_breadth(universe(above=0, below=20)).value == pytest.approx(0.0)


def test_half_above_is_half_breadth():
    assert market_breadth(universe(above=10, below=10)).value == pytest.approx(50.0)


def test_short_history_excluded_from_both_sides():
    """Counting a new listing as 'below' would fake a breadth collapse."""
    series = universe(above=10, below=0)
    series["NEWLY_LISTED"] = series_from([100.0] * 5, 999)

    assert market_breadth(series).value == pytest.approx(100.0)


def test_breadth_needs_a_cohort():
    result = market_breadth(universe(above=2, below=1))

    assert result.available is False
    assert "needs 10" in result.reason


def test_breadth_of_empty_universe():
    assert market_breadth({}).available is False


# ---------------------------------------------------------------------------
# B3 -- efficiency ratio
# ---------------------------------------------------------------------------


def test_straight_line_is_perfectly_efficient():
    closes = [100.0 + i for i in range(25)]

    assert efficiency_ratio(series_from(closes)).value == pytest.approx(1.0)


def test_zigzag_is_inefficient():
    closes = [100.0 + (5 if i % 2 else 0) for i in range(25)]

    result = efficiency_ratio(series_from(closes))

    assert result.value < EFFICIENCY_CHOPPY


def test_round_trip_has_near_zero_efficiency():
    """Up then back down covers ground but goes nowhere."""
    closes = [100.0 + i for i in range(11)] + [110.0 - i for i in range(1, 11)]

    result = efficiency_ratio(series_from(closes))

    assert result.value == pytest.approx(0.0, abs=0.01)


def test_flat_series_has_no_efficiency():
    result = efficiency_ratio(series_from([100.0] * 25))

    assert result.available is False
    assert "no movement" in result.reason


def test_efficiency_needs_history():
    assert efficiency_ratio(series_from([100.0] * 5)).available is False


# ---------------------------------------------------------------------------
# B2 -- VIX percentile
# ---------------------------------------------------------------------------


def test_highest_reading_is_top_percentile():
    closes = [10.0] * (VIX_PERCENTILE_WINDOW - 1) + [40.0]

    assert percentile_of_latest(series_from(closes)).value == pytest.approx(100.0)


def test_lowest_reading_is_bottom_percentile():
    closes = [40.0] * (VIX_PERCENTILE_WINDOW - 1) + [5.0]

    assert percentile_of_latest(series_from(closes)).value == pytest.approx(0.0)


def test_percentile_needs_history():
    assert percentile_of_latest(series_from([10.0] * 50)).available is False


# ---------------------------------------------------------------------------
# B4 -- flows
# ---------------------------------------------------------------------------


def test_parses_live_shaped_payload():
    payload = [
        {
            "buyValue": "17310.04",
            "category": "DII",
            "date": "18-Sep-2026",
            "netValue": "1019.69",
            "sellValue": "16290.35",
        },
        {
            "buyValue": "38461.63",
            "category": "FII/FPI",
            "date": "18-Sep-2026",
            "netValue": "599.54",
            "sellValue": "37862.09",
        },
    ]

    records = parse_flows(payload)

    assert len(records) == 2
    by_cat = {r.category: r for r in records}
    assert by_cat["DII"].net_value == pytest.approx(1019.69)
    assert by_cat["FII/FPI"].session_date == AS_OF


def test_negative_net_flow_is_parsed():
    payload = [
        {
            "buyValue": "100.0",
            "category": "FII/FPI",
            "date": "18-Sep-2026",
            "netValue": "-500.25",
            "sellValue": "600.25",
        }
    ]

    assert parse_flows(payload)[0].net_value == pytest.approx(-500.25)


def test_empty_payload_is_rejected():
    with pytest.raises(DataUnavailableError, match="empty"):
        parse_flows([])


def test_non_list_payload_is_rejected():
    with pytest.raises(DataUnavailableError):
        parse_flows({"error": "blocked"})


def test_unparseable_date_is_rejected():
    payload = [
        {
            "buyValue": "1",
            "category": "DII",
            "date": "yesterday",
            "netValue": "1",
            "sellValue": "0",
        }
    ]

    with pytest.raises(DataIntegrityError, match="unparseable date"):
        parse_flows(payload)


def test_non_numeric_values_are_skipped_not_zeroed():
    payload = [
        {"buyValue": "-", "category": "DII", "date": "18-Sep-2026",
         "netValue": "-", "sellValue": "-"},
        {"buyValue": "100", "category": "FII/FPI", "date": "18-Sep-2026",
         "netValue": "50", "sellValue": "50"},
    ]

    records = parse_flows(payload)

    assert [r.category for r in records] == ["FII/FPI"]


def test_flow_round_trip(db):
    repo = FlowRepository(db)
    repo.upsert_many(
        [FlowRecord(AS_OF, "FII/FPI", 100.0, 60.0, 40.0)], source="test"
    )

    stored = repo.get_for(AS_OF)

    assert stored["FII/FPI"].net_value == pytest.approx(40.0)


def test_flow_upsert_is_idempotent(db):
    repo = FlowRepository(db)
    repo.upsert_many([FlowRecord(AS_OF, "DII", 1.0, 1.0, 0.0)], source="a")
    repo.upsert_many([FlowRecord(AS_OF, "DII", 2.0, 1.0, 1.0)], source="b")

    stored = repo.get_for(AS_OF)

    assert len(stored) == 1
    assert stored["DII"].net_value == pytest.approx(1.0)


def test_flows_absent_for_unknown_session(db):
    assert FlowRepository(db).get_for(AS_OF) == {}


def test_seeding_registers_index_and_vix(db):
    ids = seed_regime_instruments(db)

    assert set(ids) == {"NIFTY50", "INDIAVIX"}


# ---------------------------------------------------------------------------
# Composite verdict
# ---------------------------------------------------------------------------


def trending_index():
    return series_from([100.0 + i for i in range(30)])


def choppy_index():
    return series_from([100.0 + (5 if i % 2 else 0) for i in range(30)])


def calm_vix():
    return series_from([20.0] * (VIX_PERCENTILE_WINDOW - 1) + [10.0])


def panicked_vix():
    return series_from([10.0] * (VIX_PERCENTILE_WINDOW - 1) + [45.0])


def test_healthy_market_is_favourable():
    regime = assess_regime(
        AS_OF,
        universe(above=18, below=2),
        index_series=trending_index(),
        vix_series=calm_vix(),
        flows={"FII/FPI": FlowRecord(AS_OF, "FII/FPI", 1.0, 0.0, 500.0)},
    )

    assert regime.verdict is RegimeVerdict.FAVOURABLE
    assert regime.is_favourable is True
    assert regime.warnings == []
    assert regime.positives


def test_one_warning_is_mixed():
    regime = assess_regime(
        AS_OF,
        universe(above=2, below=18),  # narrow breadth only
        index_series=trending_index(),
        vix_series=calm_vix(),
    )

    assert regime.verdict is RegimeVerdict.MIXED
    assert len(regime.warnings) == 1
    assert "narrow breadth" in regime.warnings[0]


def test_two_warnings_are_hostile():
    """Narrow breadth plus a choppy tape -- exactly where momentum unwinds."""
    regime = assess_regime(
        AS_OF,
        universe(above=1, below=19),
        index_series=choppy_index(),
        vix_series=calm_vix(),
    )

    assert regime.verdict is RegimeVerdict.HOSTILE
    assert regime.is_favourable is False
    assert len(regime.warnings) >= 2


def test_elevated_vix_warns():
    regime = assess_regime(
        AS_OF,
        universe(above=18, below=2),
        index_series=trending_index(),
        vix_series=panicked_vix(),
    )

    assert any("elevated volatility" in w for w in regime.warnings)


def test_both_institutions_selling_warns():
    regime = assess_regime(
        AS_OF,
        universe(above=18, below=2),
        index_series=trending_index(),
        vix_series=calm_vix(),
        flows={
            "FII/FPI": FlowRecord(AS_OF, "FII/FPI", 1.0, 2.0, -3000.0),
            "DII": FlowRecord(AS_OF, "DII", 1.0, 2.0, -1500.0),
        },
    )

    assert any("net sellers" in w for w in regime.warnings)


def test_one_institution_buying_does_not_warn():
    regime = assess_regime(
        AS_OF,
        universe(above=18, below=2),
        index_series=trending_index(),
        vix_series=calm_vix(),
        flows={
            "FII/FPI": FlowRecord(AS_OF, "FII/FPI", 1.0, 2.0, -3000.0),
            "DII": FlowRecord(AS_OF, "DII", 2.0, 1.0, 2000.0),
        },
    )

    assert not any("net sellers" in w for w in regime.warnings)


def test_no_data_yields_unknown_not_favourable():
    """Absence of evidence must not read as an all-clear."""
    regime = assess_regime(AS_OF, {})

    assert regime.verdict is RegimeVerdict.UNKNOWN
    assert regime.is_favourable is False


def test_summary_mentions_available_gates():
    regime = assess_regime(
        AS_OF,
        universe(above=18, below=2),
        index_series=trending_index(),
        vix_series=calm_vix(),
        flows={"DII": FlowRecord(AS_OF, "DII", 2.0, 1.0, 1019.69)},
    )

    summary = regime.summary()

    assert "breadth" in summary
    assert "VIX" in summary
    assert "DII" in summary


def test_missing_flows_do_not_warn():
    regime = assess_regime(
        AS_OF,
        universe(above=18, below=2),
        index_series=trending_index(),
        vix_series=calm_vix(),
        flows={},
    )

    assert regime.fii_net.available is False
    assert not any("net sellers" in w for w in regime.warnings)


# ---------------------------------------------------------------------------
# Live network check
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_live_flow_fetch():
    from algorix.regime import NseFlowClient

    records = NseFlowClient().fetch()

    categories = {r.category for r in records}
    assert any("FII" in c for c in categories)
    assert "DII" in categories
