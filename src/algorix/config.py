"""Runtime configuration.

Loaded from environment variables with sane defaults so the tool runs with no
setup, while still allowing overrides for testing and alternate data
locations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from algorix.exceptions import ConfigError

# NSE and MCX both operate in IST. Every date boundary in this system -- what
# "today" means, when a session closed, which bar is the latest -- is an IST
# question, so the timezone is fixed rather than taken from the host clock.
MARKET_TIMEZONE = "Asia/Kolkata"

DEFAULT_DATA_DIR = Path.home() / ".algorix"
DEFAULT_DB_FILENAME = "algorix.db"


def load_dotenv_if_present(dotenv_path: str | Path | None = None) -> None:
    """Load a `.env` file into the process environment, if one exists. A
    no-op otherwise -- this must never be the reason the tool fails to run
    with no setup.

    Called once, explicitly, at the top of each CLI `main()` -- not at
    import time. Importing this module happens during test collection too,
    and a real `.env` sitting on a developer's machine must never leak
    secrets into `os.environ` for a test that reads it directly without
    overriding (see sentiment.py's ANTHROPIC_API_KEY check). A CLI
    invocation is a deliberate run of the tool; an import is not.

    `dotenv_path=None` (every real call site) hands resolution to
    `python-dotenv` itself, which walks up from *this file's own location
    on disk* -- not `os.getcwd()` -- so the project's `.env` is found
    however the CLI is invoked. That also means `monkeypatch.chdir` cannot
    isolate a test from a real `.env`: pass an explicit `dotenv_path`
    instead (tests do).

    Values already set in the real environment win -- `load_dotenv`'s
    default is to not override an existing variable, which is the right
    precedence for a machine where a value was deliberately exported.
    """
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=dotenv_path)


@dataclass(frozen=True)
class Config:
    """Immutable runtime configuration."""

    data_dir: Path
    db_path: Path
    request_timeout_seconds: float
    max_retries: int
    retry_backoff_seconds: float

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Config":
        """Build config from environment variables, falling back to defaults.

        Passing `env` explicitly keeps tests independent of the real
        environment.
        """
        env = os.environ if env is None else env

        data_dir = Path(env.get("ALGORIX_DATA_DIR", str(DEFAULT_DATA_DIR))).expanduser()
        db_path = Path(
            env.get("ALGORIX_DB_PATH", str(data_dir / DEFAULT_DB_FILENAME))
        ).expanduser()

        config = cls(
            data_dir=data_dir,
            db_path=db_path,
            request_timeout_seconds=_read_float(
                env, "ALGORIX_REQUEST_TIMEOUT_SECONDS", 30.0
            ),
            max_retries=_read_int(env, "ALGORIX_MAX_RETRIES", 3),
            retry_backoff_seconds=_read_float(
                env, "ALGORIX_RETRY_BACKOFF_SECONDS", 1.0
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Reject values that would silently misbehave at runtime."""
        if self.request_timeout_seconds <= 0:
            raise ConfigError(
                "request_timeout_seconds must be positive, got "
                f"{self.request_timeout_seconds}"
            )
        # A negative retry count would read as "retry forever" in a naive loop.
        # Zero is legitimate: try once, do not retry.
        if self.max_retries < 0:
            raise ConfigError(f"max_retries cannot be negative, got {self.max_retries}")
        if self.retry_backoff_seconds < 0:
            raise ConfigError(
                "retry_backoff_seconds cannot be negative, got "
                f"{self.retry_backoff_seconds}"
            )

    def ensure_data_dir(self) -> Path:
        """Create the data directory if absent. Returns the directory."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ConfigError(f"Cannot create data directory {self.data_dir}: {exc}") from exc
        return self.data_dir


def _read_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from exc


def _read_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a number, got {raw!r}") from exc
