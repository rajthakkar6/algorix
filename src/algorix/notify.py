"""Digest formatting and delivery.

The digest is the product: a ranked, explained list in your hand before the
09:15 open. Formatting lives apart from delivery so the message can be
rendered and inspected without sending anything.

Delivery is Telegram. Credentials come from the environment
(``ALGORIX_TELEGRAM_TOKEN``, ``ALGORIX_TELEGRAM_CHAT_ID``) and are never
written to disk or logged.

A hostile regime **warns, it does not suppress** (PROJECT_SCOPE open decision
4). Silence is ambiguous -- an unsent digest is indistinguishable from a
broken cron job -- whereas a warning at the top of a digest is information
you can act on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import requests

from algorix.exceptions import AlgorixError, SourceUnreachableError
from algorix.regime import MarketRegime, RegimeVerdict
from algorix.scoring import ScoredUniverse, StockScore

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"

#: Telegram rejects messages beyond this; digests are trimmed to fit.
MAX_MESSAGE_CHARS = 4000


class NotConfiguredError(AlgorixError):
    """Delivery credentials are absent."""


@dataclass(frozen=True)
class TelegramConfig:
    token: str
    chat_id: str

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "TelegramConfig":
        env = env if env is not None else dict(os.environ)
        token = env.get("ALGORIX_TELEGRAM_TOKEN", "").strip()
        chat_id = env.get("ALGORIX_TELEGRAM_CHAT_ID", "").strip()

        if not token or not chat_id:
            raise NotConfiguredError(
                "Telegram is not configured. Set ALGORIX_TELEGRAM_TOKEN and "
                "ALGORIX_TELEGRAM_CHAT_ID."
            )
        return cls(token=token, chat_id=chat_id)


def _fmt(value, spec: str = ".0f", fallback: str = "--") -> str:
    return f"{value.value:{spec}}" if value.available else fallback


def format_stock_line(rank: int, score: StockScore) -> str:
    """One ranked entry: the number, then why."""
    by_code = {c.code: c for c in score.contributions}
    strengths = [
        c.label for c in score.contributions if c.percentile >= 70
    ]

    line = f"{rank}. *{score.symbol}* — {score.score.value:.0f}/100"
    if strengths:
        line += f"\n    ↑ {', '.join(strengths)}"

    if score.risk and score.risk.stop_price.available:
        line += (
            f"\n    stop {_fmt(score.risk.stop_price, '.1f')}"
            f" ({_fmt(score.risk.stop_distance_pct, '.1f')}%)"
        )
    if score.missing:
        line += f"\n    ⚠ no data for {', '.join(score.missing)}"
    return line


def format_digest(
    universe: ScoredUniverse,
    regime: MarketRegime | None = None,
    top_n: int = 8,
    metals: str | None = None,
) -> str:
    """Render the morning digest."""
    lines = [f"*Algorix* — {universe.as_of:%d %b %Y}"]

    if regime is not None:
        lines.append("")
        if regime.verdict is RegimeVerdict.HOSTILE:
            lines.append("🔴 *HOSTILE REGIME — scores are less reliable today*")
        elif regime.verdict is RegimeVerdict.MIXED:
            lines.append("🟡 *Mixed regime — treat scores with caution*")
        elif regime.verdict is RegimeVerdict.FAVOURABLE:
            lines.append("🟢 Favourable regime")
        else:
            lines.append("⚪ Regime unknown — insufficient data")

        lines.append(f"_{regime.summary()}_")
        for warning in regime.warnings:
            lines.append(f"• {warning}")

    ranked = universe.top(top_n)
    lines.append("")
    if not ranked:
        lines.append("_No instruments scored today._")
    else:
        lines.append(f"*Top {len(ranked)} of {len(universe.scores)}*")
        lines.append("")
        for i, score in enumerate(ranked, start=1):
            lines.append(format_stock_line(i, score))

    if metals:
        lines.append("")
        lines.append(f"*Metals* — {metals}")

    notes = []
    if universe.ineligible:
        notes.append(f"{len(universe.ineligible)} ineligible")
    if universe.unscored:
        notes.append(f"{len(universe.unscored)} unscored")
    if notes:
        lines.append("")
        lines.append("_" + ", ".join(notes) + "_")

    # Method is stated, never implied -- principle 0.2a.
    if ranked:
        lines.append(
            f"_Scored {ranked[0].method}, cohort {ranked[0].cohort_size}._"
        )
    lines.append("_Not investment advice._")

    message = "\n".join(lines)
    if len(message) > MAX_MESSAGE_CHARS:
        message = message[: MAX_MESSAGE_CHARS - 20].rstrip() + "\n_…truncated_"
    return message


class TelegramNotifier:
    """Sends digests to Telegram."""

    def __init__(self, config: TelegramConfig | None = None) -> None:
        self.config = config or TelegramConfig.from_env()

    def send(self, message: str, timeout_seconds: float = 30.0) -> bool:
        """Send a message. Returns True on success.

        Raises SourceUnreachableError rather than returning False on a
        network fault, so a failed delivery cannot be mistaken for an empty
        digest.
        """
        if not message.strip():
            raise ValueError("refusing to send an empty digest")

        try:
            response = requests.post(
                TELEGRAM_API.format(token=self.config.token),
                json={
                    "chat_id": self.config.chat_id,
                    "text": message,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True,
                },
                timeout=timeout_seconds,
            )
        except requests.RequestException as exc:
            raise SourceUnreachableError(
                f"Could not reach Telegram: {exc}"
            ) from exc

        if response.status_code != 200:
            raise SourceUnreachableError(
                f"Telegram returned HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )
        return True
