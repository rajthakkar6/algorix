"""AI Q&A: one free-text question about one stock, answered from data this
tool has already gathered. Sits beside sentiment.py the same way
sentiment.py sits beside scoring.py -- read by nothing in the scoring path.

**Synchronous, not batched.** Unlike G4 (Batch API, ~50% discount), every
question here is one real-time API call at full price. Flagged per
CLAUDE.md's cost-consciousness rule: at personal usage volume this is
unlikely to be material, but it is a materially different cost shape than
the rest of this codebase (everything else is either free or batched), and
that must be stated, not silently absorbed.

**One-shot, not a conversation.** No chat history, no multi-turn memory --
each question is independent context assembled fresh from the store.

Invariant 1 (the LLM never overrides the quant score) is enforced in the
system prompt: the model is told the quant score is this tool's single
authoritative number and to label any disagreement rather than substitute
its own verdict. Invariant 2 (sentiment/LLM output is never a score input)
is structural here too: nothing in scoring.py imports this module, and
nothing here writes anywhere scoring.py reads.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from typing import Protocol

from algorix.calendar import IST, TradingCalendar
from algorix.exceptions import ConfigError, DataIntegrityError, SourceUnreachableError
from algorix.indicators import DELIVERY_BASELINE, PEAD_WINDOW_SESSIONS, stock_indicator_rows
from algorix.journal import Journal
from algorix.models import Instrument, QaQuery
from algorix.scan import SCAN_LOOKBACK_SESSIONS
from algorix.scoring import SIGNAL_LABELS
from algorix.sentiment import NotConfiguredError
from algorix.series import load_series
from algorix.storage import (
    AnnouncementRepository,
    Database,
    DeliveryRepository,
    EarningsSurpriseRepository,
    EventExtractionRepository,
    QaQueryRepository,
)

DEFAULT_ANTHROPIC_MODEL_ID = "claude-sonnet-5"
DEFAULT_OPENAI_MODEL_ID = "gpt-4o-mini"
DEFAULT_PROVIDER = "anthropic"

#: Deliberately separate from sentiment.py's ALGORIX_LLM_PROVIDER/_MODEL --
#: G4 wants the cheapest capable batch model at high volume; Q&A is
#: low-volume and interactive, where a stronger model is more likely worth
#: it. Coupling the two would mean a G4 cost-driven model change silently
#: changes Q&A answer quality too.
_ENV_PROVIDER = "ALGORIX_QA_PROVIDER"
_ENV_MODEL = "ALGORIX_QA_MODEL"

#: Bumped whenever the prompt or schema changes in a way that makes new
#: answers incomparable to old ones -- same convention as
#: sentiment.PROMPT_VERSION.
PROMPT_VERSION = 1

#: Sessions of OHLCV included verbatim in the prompt -- enough for "what's
#: it done lately," not the whole scoring window (already summarized via
#: the indicators/score above).
RECENT_BARS = 20

#: How far back to pull G4-extracted announcement events for context.
ANNOUNCEMENT_LOOKBACK_DAYS = 60


# ---------------------------------------------------------------------------
# Provider layer -- synchronous, structurally new (everything else in this
# codebase is Batch-API-only). Mirrors sentiment.py's Extractor shape where
# it can, but the actual call is a normal request/response, not submit/poll.
# ---------------------------------------------------------------------------


class Answerer(Protocol):
    model_id: str

    def ask(self, system_prompt: str, user_message: str) -> str: ...


class AnthropicAnswerer:
    def __init__(self, model_id: str = DEFAULT_ANTHROPIC_MODEL_ID) -> None:
        self.model_id = model_id
        self._client = None  # constructed lazily -- see _require_client

    def _require_client(self):
        if self._client is not None:
            return self._client
        if not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
            raise NotConfiguredError(
                "ANTHROPIC_API_KEY is not set. AI Q&A is opt-in -- set it "
                "(or otherwise configure Anthropic credentials) to enable it."
            )
        import anthropic

        self._client = anthropic.Anthropic()
        return self._client

    def ask(self, system_prompt: str, user_message: str) -> str:
        client = self._require_client()
        try:
            response = client.messages.create(
                model=self.model_id,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
        except NotConfiguredError:
            raise
        except Exception as exc:
            raise SourceUnreachableError(
                f"Anthropic call failed: {type(exc).__name__}: {exc}"
            ) from exc
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise DataIntegrityError("LLM response contained no text content")
        return text


class OpenAIAnswerer:
    def __init__(self, model_id: str = DEFAULT_OPENAI_MODEL_ID) -> None:
        self.model_id = model_id
        self._client = None  # constructed lazily -- see _require_client

    def _require_client(self):
        if self._client is not None:
            return self._client
        if not (os.environ.get("OPENAI_API_KEY") or "").strip():
            raise NotConfiguredError(
                "OPENAI_API_KEY is not set. AI Q&A is opt-in -- set it to "
                "enable it with ALGORIX_QA_PROVIDER=openai."
            )
        import openai

        self._client = openai.OpenAI()
        return self._client

    def ask(self, system_prompt: str, user_message: str) -> str:
        client = self._require_client()
        try:
            response = client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
        except NotConfiguredError:
            raise
        except Exception as exc:
            raise SourceUnreachableError(
                f"OpenAI call failed: {type(exc).__name__}: {exc}"
            ) from exc
        text = response.choices[0].message.content if response.choices else None
        if not text:
            raise DataIntegrityError("LLM response contained no text content")
        return text


_PROVIDER_DEFAULT_MODEL = {
    "anthropic": DEFAULT_ANTHROPIC_MODEL_ID,
    "openai": DEFAULT_OPENAI_MODEL_ID,
}


def build_answerer(provider: str | None = None, model_id: str | None = None) -> Answerer:
    """Construct the active provider's Answerer.

    `provider` defaults to `ALGORIX_QA_PROVIDER`, then `DEFAULT_PROVIDER`.
    `model_id` defaults to that provider's own default -- mirrors
    sentiment.build_extractor's reasoning exactly, applied to a different
    env var pair (see module docstring for why they're separate).
    """
    provider = (provider or os.environ.get(_ENV_PROVIDER) or DEFAULT_PROVIDER).lower()
    if provider not in _PROVIDER_DEFAULT_MODEL:
        raise ConfigError(
            f"Unknown {_ENV_PROVIDER} {provider!r}. "
            f"Supported: {sorted(_PROVIDER_DEFAULT_MODEL)}."
        )
    model_id = model_id or os.environ.get(_ENV_MODEL) or _PROVIDER_DEFAULT_MODEL[provider]

    if provider == "openai":
        return OpenAIAnswerer(model_id=model_id)
    return AnthropicAnswerer(model_id=model_id)


# ---------------------------------------------------------------------------
# Context assembly -- pure, no network, independently testable
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StockContext:
    symbol: str
    as_of: date
    trend_summary: str
    indicators: list[tuple[str, str, str]]  # (code, label, formatted text)
    latest_score: float | None
    score_contributions: dict[str, float]
    delivery_summary: str
    earnings_summary: str
    recent_events: list[dict]
    recent_ohlcv: list[dict]

    def to_storage_dict(self) -> dict:
        d = asdict(self)
        d["as_of"] = self.as_of.isoformat()
        return d


def gather_stock_context(
    db: Database, instrument: Instrument, as_of: date, calendar: TradingCalendar
) -> StockContext:
    """Everything this page already knows about one stock, assembled into
    plain text for the prompt. Every missing/empty piece renders as
    explicit "unavailable"/"none" text -- never omitted, never shown as
    zero, the same rule this project's UI already follows applied to
    prompt content.
    """
    series = load_series(db, instrument.id, as_of, SCAN_LOOKBACK_SESSIONS, calendar)
    records = DeliveryRepository(db).get_range(
        instrument.id, calendar.trading_days_ago(as_of, DELIVERY_BASELINE * 2), as_of
    )
    trend, _donchian, _short_return, rows = stock_indicator_rows(series, records, calendar)
    trend_summary = (
        f"{'Uptrend' if trend.is_uptrend else 'Not an uptrend'}, "
        f"{trend.pct_from_fast:+.1f}% vs 50DMA, {trend.pct_from_slow:+.1f}% vs 200DMA"
    ) if trend else "insufficient history for a trend reading"

    history = Journal(db).history_for(instrument.symbol, limit=1)
    latest = history[0] if history else None

    earnings = EarningsSurpriseRepository(db).get_range(
        instrument.id, calendar.trading_days_ago(as_of, PEAD_WINDOW_SESSIONS), as_of
    )
    earnings_summary = (
        "; ".join(f"{e.report_date}: surprise {e.surprise_pct:+.1f}%" for e in earnings)
        or "no recent reported quarter in the lookback window"
    )

    announcements = AnnouncementRepository(db).get_range(
        instrument.id, as_of - timedelta(days=ANNOUNCEMENT_LOOKBACK_DAYS), as_of
    )
    events_repo = EventExtractionRepository(db)
    recent_events = []
    for a in announcements:
        event = events_repo.get(a.seq_id)
        if event is not None:
            recent_events.append({
                "announced_at": a.announced_at.date().isoformat(),
                "event_type": str(event.event_type),
                "polarity": str(event.polarity),
                "materiality": str(event.materiality),
                "risk_flag": event.risk_flag,
                "risk_reason": event.risk_reason,
            })

    recent_bars = series.bars[-RECENT_BARS:]

    def _format(value, spec) -> str:
        if value is None or not getattr(value, "available", False):
            reason = getattr(value, "reason", None) if value is not None else None
            return f"unavailable ({reason})" if reason else "unavailable"
        return format(value.value, spec)

    return StockContext(
        symbol=instrument.symbol,
        as_of=as_of,
        trend_summary=trend_summary,
        indicators=[(code, label, _format(value, spec) + unit)
                    for code, label, value, unit, spec in rows],
        latest_score=latest.score if latest else None,
        score_contributions=latest.contributions if latest else {},
        delivery_summary=(
            "; ".join(
                f"{r.session_date}: {r.delivery_pct:.1f}%"
                for r in records[-5:] if r.delivery_pct is not None
            )
            or "no delivery data available"
        ),
        earnings_summary=earnings_summary,
        recent_events=recent_events,
        recent_ohlcv=[
            {"date": b.session_date.isoformat(), "open": b.open, "high": b.high,
             "low": b.low, "close": b.close, "volume": b.volume}
            for b in recent_bars
        ],
    )


_SYSTEM_PROMPT = """You are an analysis assistant inside Algorix, a personal \
swing-trading tool for Indian equities. Answer the trader's question about \
ONE stock using only the structured data given below.

The data includes the tool's own quantitative score and its per-factor \
percentile contributions -- this score is computed by a separate \
deterministic pipeline and is the tool's single authoritative number. You \
must NOT invent a competing score, rating, or buy/sell recommendation. If \
your reading of the data would point a different way than the quant score, \
say so explicitly as a labelled disagreement ("the quant score weights X \
highly, but the Y data here suggests...") -- never silently substitute your \
own judgement for it or present your view as equally authoritative.

If the data given does not answer the question, say so plainly rather than \
guessing or drawing on outside knowledge about the company."""


def _build_user_message(question: str, ctx: StockContext) -> str:
    lines = [
        f"Stock: {ctx.symbol}", f"Session: {ctx.as_of}", "",
        "== Quant score ==",
        f"Score: {ctx.latest_score:.0f}/100" if ctx.latest_score is not None else "Score: unavailable",
        "Per-factor contributions (percentile within universe):",
    ]
    if ctx.score_contributions:
        for code, pct in ctx.score_contributions.items():
            lines.append(f"  {code} ({SIGNAL_LABELS.get(code, code)}): {pct:.0f}th percentile")
    else:
        lines.append("  none available")

    lines += ["", "== Trend ==", ctx.trend_summary, "", "== Indicators =="]
    lines += [f"  {c} {l}: {v}" for c, l, v in ctx.indicators]

    lines += ["", "== Delivery ==", ctx.delivery_summary,
              "", "== Earnings (PEAD) ==", ctx.earnings_summary,
              "", "== Recent official-filing events (G4, never a score input) =="]
    if ctx.recent_events:
        for e in ctx.recent_events:
            risk = f" [RISK: {e['risk_reason']}]" if e["risk_flag"] else ""
            lines.append(
                f"  {e['announced_at']}: {e['event_type']} / {e['polarity']} / "
                f"materiality={e['materiality']}{risk}"
            )
    else:
        lines.append("  none in the lookback window")

    lines += ["", f"== Recent OHLCV (last {len(ctx.recent_ohlcv)} sessions) =="]
    for b in ctx.recent_ohlcv:
        lines.append(
            f"  {b['date']}: O{b['open']:.2f} H{b['high']:.2f} L{b['low']:.2f} "
            f"C{b['close']:.2f} V{b['volume']}"
        )

    lines += ["", f"== Question ==", question]
    return "\n".join(lines)


def ask_about_stock(
    db: Database,
    instrument: Instrument,
    question: str,
    *,
    as_of: date | None = None,
    calendar: TradingCalendar | None = None,
    answerer: Answerer | None = None,
) -> QaQuery:
    """Assemble context, ask the LLM, log the audit row, return it.

    `NotConfiguredError`/`SourceUnreachableError`/`DataIntegrityError`
    propagate uncaught -- the caller (the web route) decides how each is
    presented; nothing here swallows a failure.
    """
    calendar = calendar or TradingCalendar()
    as_of = as_of or calendar.last_completed_session(datetime.now(IST))
    ctx = gather_stock_context(db, instrument, as_of, calendar)
    answerer = answerer or build_answerer()
    answer_text = answerer.ask(_SYSTEM_PROMPT, _build_user_message(question, ctx))
    query = QaQuery(
        instrument_id=instrument.id,
        question=question,
        context=ctx.to_storage_dict(),
        answer=answer_text,
        model_id=answerer.model_id,
        prompt_version=PROMPT_VERSION,
    )
    return QaQueryRepository(db).create(query)
