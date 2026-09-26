"""Tests for AI Q&A (qa.py)."""

from datetime import date

import pytest

from algorix.exceptions import ConfigError, DataIntegrityError, SourceUnreachableError
from algorix.sentiment import NotConfiguredError

TARGET = date(2026, 9, 18)


class FakeAnswerer:
    """Stands in for AnthropicAnswerer/OpenAIAnswerer -- same shape as
    test_sentiment.py's FakeExtractor."""

    def __init__(self, answer: str = "a plain-text answer", model_id: str = "fake-model-1"):
        self.model_id = model_id
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    def ask(self, system_prompt: str, user_message: str) -> str:
        self.calls.append((system_prompt, user_message))
        return self.answer


# ---------------------------------------------------------------------------
# build_answerer
# ---------------------------------------------------------------------------


def test_build_answerer_defaults_to_anthropic(monkeypatch):
    from algorix.qa import DEFAULT_ANTHROPIC_MODEL_ID, AnthropicAnswerer, build_answerer

    monkeypatch.delenv("ALGORIX_QA_PROVIDER", raising=False)

    answerer = build_answerer()

    assert isinstance(answerer, AnthropicAnswerer)
    assert answerer.model_id == DEFAULT_ANTHROPIC_MODEL_ID


def test_build_answerer_explicit_openai():
    from algorix.qa import DEFAULT_OPENAI_MODEL_ID, OpenAIAnswerer, build_answerer

    answerer = build_answerer(provider="openai")

    assert isinstance(answerer, OpenAIAnswerer)
    assert answerer.model_id == DEFAULT_OPENAI_MODEL_ID


def test_build_answerer_reads_provider_from_env(monkeypatch):
    from algorix.qa import OpenAIAnswerer, build_answerer

    monkeypatch.setenv("ALGORIX_QA_PROVIDER", "openai")

    assert isinstance(build_answerer(), OpenAIAnswerer)


def test_build_answerer_is_independent_of_g4_env_vars(monkeypatch):
    """ALGORIX_LLM_PROVIDER (G4's own env var) must not affect Q&A -- the
    two are deliberately decoupled so a G4 cost-driven model change can't
    silently change Q&A quality too."""
    from algorix.qa import AnthropicAnswerer, build_answerer

    monkeypatch.setenv("ALGORIX_LLM_PROVIDER", "openai")
    monkeypatch.delenv("ALGORIX_QA_PROVIDER", raising=False)

    assert isinstance(build_answerer(), AnthropicAnswerer)


def test_build_answerer_model_override():
    from algorix.qa import build_answerer

    answerer = build_answerer(provider="openai", model_id="gpt-4o")

    assert answerer.model_id == "gpt-4o"


def test_build_answerer_model_from_env(monkeypatch):
    from algorix.qa import build_answerer

    monkeypatch.setenv("ALGORIX_QA_MODEL", "gpt-4o")

    answerer = build_answerer(provider="openai")

    assert answerer.model_id == "gpt-4o"


def test_build_answerer_unknown_provider_raises_config_error():
    from algorix.qa import build_answerer

    with pytest.raises(ConfigError, match="mistral"):
        build_answerer(provider="mistral")


def test_build_answerer_is_case_insensitive():
    from algorix.qa import OpenAIAnswerer, build_answerer

    assert isinstance(build_answerer(provider="OpenAI"), OpenAIAnswerer)


# ---------------------------------------------------------------------------
# AnthropicAnswerer / OpenAIAnswerer -- credential resolution
# ---------------------------------------------------------------------------


def test_anthropic_answerer_raises_not_configured_without_api_key(monkeypatch):
    from algorix.qa import AnthropicAnswerer

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    answerer = AnthropicAnswerer()

    with pytest.raises(NotConfiguredError, match="ANTHROPIC_API_KEY"):
        answerer.ask("system", "question")


def test_openai_answerer_raises_not_configured_without_api_key(monkeypatch):
    from algorix.qa import OpenAIAnswerer

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    answerer = OpenAIAnswerer()

    with pytest.raises(NotConfiguredError, match="OPENAI_API_KEY"):
        answerer.ask("system", "question")


def test_anthropic_answerer_wraps_sdk_failure(monkeypatch):
    """An SDK-level failure (network, auth, rate limit) must surface as
    SourceUnreachableError, not vanish or crash with a raw SDK exception."""
    from algorix.qa import AnthropicAnswerer

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    answerer = AnthropicAnswerer()

    class _BoomClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                raise RuntimeError("connection refused")

    answerer._client = _BoomClient()

    with pytest.raises(SourceUnreachableError, match="connection refused"):
        answerer.ask("system", "question")


def test_anthropic_answerer_rejects_empty_response_text(monkeypatch):
    from algorix.qa import AnthropicAnswerer

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    answerer = AnthropicAnswerer()

    class _Block:
        type = "image"  # no text block in the response at all

    class _Response:
        content = [_Block()]

    class _EmptyClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                return _Response()

    answerer._client = _EmptyClient()

    with pytest.raises(DataIntegrityError, match="no text content"):
        answerer.ask("system", "question")


# ---------------------------------------------------------------------------
# gather_stock_context / StockContext / prompt assembly
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    from algorix.storage import Database

    database = Database(tmp_path / "qa.db")
    database.migrate()
    return database


@pytest.fixture
def calendar():
    from algorix.calendar import TradingCalendar

    return TradingCalendar()


@pytest.fixture
def sparse_instrument(db):
    """An instrument with almost no history -- every indicator should
    render as an explicit "unavailable" reason, never omitted or zero."""
    from algorix.models import Bar, Exchange, Instrument, InstrumentType
    from algorix.storage import BarRepository, InstrumentRepository

    instrument_id = InstrumentRepository(db).upsert(
        Instrument(symbol="SPARSE", exchange=Exchange.NSE,
                   instrument_type=InstrumentType.EQUITY)
    )
    BarRepository(db).upsert_many(
        instrument_id,
        [Bar(session_date=TARGET, open=99, high=101, low=98, close=100, volume=1000)],
        source="test",
    )
    return InstrumentRepository(db).get_by_id(instrument_id)


def test_gather_stock_context_with_sparse_data_shows_unavailable_not_omitted(
    db, calendar, sparse_instrument
):
    from algorix.qa import gather_stock_context

    ctx = gather_stock_context(db, sparse_instrument, TARGET, calendar)

    assert ctx.symbol == "SPARSE"
    assert ctx.latest_score is None
    assert ctx.score_contributions == {}
    assert ctx.delivery_summary == "no delivery data available"
    assert ctx.earnings_summary == "no recent reported quarter in the lookback window"
    assert ctx.recent_events == []
    assert len(ctx.indicators) == 7
    # Every indicator value is a formatted string -- unavailable indicators
    # must still say so explicitly, never be blank or silently dropped.
    for _code, _label, text in ctx.indicators:
        assert text  # non-empty
    assert any("unavailable" in text for _c, _l, text in ctx.indicators)


def test_build_user_message_includes_every_section_and_the_question():
    from algorix.qa import StockContext, _build_user_message

    ctx = StockContext(
        symbol="SPARSE", as_of=TARGET, trend_summary="insufficient history",
        indicators=[("A2", "52-week high proximity", "unavailable")],
        latest_score=None, score_contributions={}, delivery_summary="no delivery data available",
        earnings_summary="no recent reported quarter in the lookback window",
        recent_events=[], recent_ohlcv=[],
    )

    message = _build_user_message("Why did it drop today?", ctx)

    for heading in ("Quant score", "Trend", "Indicators", "Delivery",
                    "Earnings (PEAD)", "Recent official-filing events", "Recent OHLCV"):
        assert heading in message
    assert "Why did it drop today?" in message
    assert "Score: unavailable" in message
    assert "none in the lookback window" in message


def test_build_user_message_lists_score_contributions():
    from algorix.qa import StockContext, _build_user_message

    ctx = StockContext(
        symbol="SPARSE", as_of=TARGET, trend_summary="Uptrend, +2.0% vs 50DMA, +4.0% vs 200DMA",
        indicators=[], latest_score=77.6, score_contributions={"A4": 91.0},
        delivery_summary="no delivery data available",
        earnings_summary="no recent reported quarter in the lookback window",
        recent_events=[], recent_ohlcv=[],
    )

    message = _build_user_message("How strong is the breakout?", ctx)

    assert "Score: 78/100" in message
    assert "A4 (breakout): 91th percentile" in message


# ---------------------------------------------------------------------------
# ask_about_stock
# ---------------------------------------------------------------------------


def test_ask_about_stock_round_trips_into_a_qa_query(db, calendar, sparse_instrument):
    from algorix.qa import PROMPT_VERSION, ask_about_stock
    from algorix.storage import QaQueryRepository

    fake = FakeAnswerer(answer="Delivery data isn't available for this name.")

    query = ask_about_stock(
        db, sparse_instrument, "Why no delivery data?",
        as_of=TARGET, calendar=calendar, answerer=fake,
    )

    assert query.id is not None
    assert query.answer == "Delivery data isn't available for this name."
    assert query.model_id == "fake-model-1"
    assert query.prompt_version == PROMPT_VERSION
    assert len(fake.calls) == 1
    stored = QaQueryRepository(db).history_for(sparse_instrument.id)
    assert len(stored) == 1
    assert stored[0].question == "Why no delivery data?"


def test_ask_about_stock_propagates_not_configured(db, calendar, sparse_instrument):
    from algorix.qa import ask_about_stock

    class UnconfiguredAnswerer:
        model_id = "fake"

        def ask(self, system_prompt, user_message):
            raise NotConfiguredError("ANTHROPIC_API_KEY is not set.")

    with pytest.raises(NotConfiguredError):
        ask_about_stock(
            db, sparse_instrument, "A question", as_of=TARGET, calendar=calendar,
            answerer=UnconfiguredAnswerer(),
        )


def test_ask_about_stock_propagates_source_unreachable(db, calendar, sparse_instrument):
    from algorix.qa import ask_about_stock

    class BrokenAnswerer:
        model_id = "fake"

        def ask(self, system_prompt, user_message):
            raise SourceUnreachableError("timeout")

    with pytest.raises(SourceUnreachableError):
        ask_about_stock(
            db, sparse_instrument, "A question", as_of=TARGET, calendar=calendar,
            answerer=BrokenAnswerer(),
        )
