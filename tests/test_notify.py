"""Tests for digest formatting and delivery."""

from datetime import date, timedelta

import pytest

from algorix.cross_sectional import compute_universe_momentum
from algorix.exceptions import SourceUnreachableError
from algorix.models import Bar, DeliveryRecord
from algorix.notify import (
    MAX_MESSAGE_CHARS,
    NotConfiguredError,
    TelegramConfig,
    TelegramNotifier,
    format_digest,
)
from algorix.regime import FlowRecord, RegimeVerdict, assess_regime
from algorix.scoring import score_universe
from algorix.series import PriceSeries
from algorix.scoring import ScoredUniverse

AS_OF = date(2026, 9, 18)


def series_from(closes, instrument_id=1, volume=100_000):
    start = date(2020, 1, 1)
    bars = tuple(
        Bar(session_date=start + timedelta(days=i), open=c, high=c + 1,
            low=c - 1, close=c, volume=volume)
        for i, c in enumerate(closes)
    )
    return PriceSeries(instrument_id=instrument_id, as_of=bars[-1].session_date, bars=bars)


def build_universe(volume=100_000):
    series = {
        f"S{i}": series_from(
            [100.0 * ((1 + i / 10000.0) ** j) for j in range(400)],
            instrument_id=i + 1, volume=volume,
        )
        for i in range(12)
    }
    delivery = {
        s: [DeliveryRecord(session_date=date(2026, 8, 1) + timedelta(days=d),
                           traded_quantity=1000, delivered_quantity=500)
            for d in range(25)]
        for s in series
    }
    return score_universe(series, compute_universe_momentum(series, AS_OF),
                          AS_OF, delivery_by_symbol=delivery), series


def regime_with(verdict_series, **kwargs):
    return assess_regime(AS_OF, verdict_series, **kwargs)


# --- formatting -----------------------------------------------------------


def test_digest_lists_ranked_stocks():
    universe, _ = build_universe()

    digest = format_digest(universe, top_n=5)

    assert "Algorix" in digest
    assert "18 Sep 2026" in digest
    assert "S11" in digest


def test_digest_respects_top_n():
    universe, _ = build_universe()

    digest = format_digest(universe, top_n=3)

    assert digest.count("/100") == 3


def test_digest_explains_strengths():
    universe, _ = build_universe()

    digest = format_digest(universe, top_n=3)

    assert "↑" in digest


def test_digest_includes_stop_levels():
    universe, _ = build_universe()

    assert "stop" in format_digest(universe, top_n=3)


def test_digest_states_the_scoring_method():
    """Principle 0.2a -- method stated, never implied."""
    universe, _ = build_universe()

    assert "cross-sectional" in format_digest(universe)


def test_digest_carries_a_disclaimer():
    universe, _ = build_universe()

    assert "Not investment advice" in format_digest(universe)


def test_hostile_regime_warns_but_digest_still_sent():
    """Open decision 4: warn rather than suppress."""
    universe, series = build_universe()
    choppy = {s: series_from([100.0 + (5 if i % 2 else 0) for i in range(30)], i)
              for i, s in enumerate(series)}
    down = {f"D{i}": series_from([100.0] * 79 + [50.0], 500 + i) for i in range(20)}
    regime = assess_regime(
        AS_OF, down,
        index_series=series_from([100.0 + (5 if i % 2 else 0) for i in range(30)]),
    )

    digest = format_digest(universe, regime, top_n=3)

    assert regime.verdict is RegimeVerdict.HOSTILE
    assert "HOSTILE" in digest
    # The ranked list survives the warning.
    assert "S11" in digest


def test_favourable_regime_is_marked():
    universe, _ = build_universe()
    up = {f"U{i}": series_from([100.0] * 79 + [200.0], 600 + i) for i in range(20)}
    regime = assess_regime(
        AS_OF, up, index_series=series_from([100.0 + i for i in range(30)])
    )

    digest = format_digest(universe, regime)

    assert "Favourable" in digest


def test_digest_reports_ineligible_count():
    universe, _ = build_universe(volume=1)

    digest = format_digest(universe)

    assert "ineligible" in digest
    assert "No instruments scored" in digest


def test_empty_universe_still_produces_a_digest():
    digest = format_digest(ScoredUniverse(as_of=AS_OF, scores={}))

    assert "No instruments scored" in digest


def test_metals_line_is_included():
    universe, _ = build_universe()

    digest = format_digest(universe, metals="gold $4,424 | USDINR 95.80")

    assert "Metals" in digest
    assert "4,424" in digest


def test_long_digest_is_truncated_to_telegram_limit():
    universe, _ = build_universe()

    digest = format_digest(universe, top_n=200)

    assert len(digest) <= MAX_MESSAGE_CHARS


# --- config ---------------------------------------------------------------


def test_config_reads_environment():
    config = TelegramConfig.from_env(
        {"ALGORIX_TELEGRAM_TOKEN": "t", "ALGORIX_TELEGRAM_CHAT_ID": "c"}
    )

    assert config.token == "t"


def test_missing_token_is_reported_clearly():
    with pytest.raises(NotConfiguredError, match="ALGORIX_TELEGRAM_TOKEN"):
        TelegramConfig.from_env({"ALGORIX_TELEGRAM_CHAT_ID": "c"})


def test_missing_chat_id_is_reported():
    with pytest.raises(NotConfiguredError):
        TelegramConfig.from_env({"ALGORIX_TELEGRAM_TOKEN": "t"})


def test_blank_credentials_are_rejected():
    with pytest.raises(NotConfiguredError):
        TelegramConfig.from_env(
            {"ALGORIX_TELEGRAM_TOKEN": "  ", "ALGORIX_TELEGRAM_CHAT_ID": "c"}
        )


# --- delivery -------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code=200, text="ok"):
        self.status_code = status_code
        self.text = text


def test_send_posts_the_message(monkeypatch):
    sent = {}

    def fake_post(url, json, timeout):
        sent["url"] = url
        sent["payload"] = json
        return FakeResponse()

    monkeypatch.setattr("algorix.notify.requests.post", fake_post)
    notifier = TelegramNotifier(TelegramConfig(token="T", chat_id="C"))

    assert notifier.send("hello") is True
    assert sent["payload"]["chat_id"] == "C"
    assert "T" in sent["url"]


def test_empty_message_is_refused():
    notifier = TelegramNotifier(TelegramConfig(token="T", chat_id="C"))

    with pytest.raises(ValueError, match="empty digest"):
        notifier.send("   ")


def test_http_error_raises_rather_than_returning_false(monkeypatch):
    """A failed delivery must not look like an empty digest."""
    monkeypatch.setattr(
        "algorix.notify.requests.post",
        lambda url, json, timeout: FakeResponse(400, "bad request"),
    )
    notifier = TelegramNotifier(TelegramConfig(token="T", chat_id="C"))

    with pytest.raises(SourceUnreachableError, match="HTTP 400"):
        notifier.send("hello")


def test_network_failure_raises(monkeypatch):
    import requests as real_requests

    def boom(url, json, timeout):
        raise real_requests.ConnectionError("down")

    monkeypatch.setattr("algorix.notify.requests.post", boom)
    notifier = TelegramNotifier(TelegramConfig(token="T", chat_id="C"))

    with pytest.raises(SourceUnreachableError, match="Could not reach Telegram"):
        notifier.send("hello")
