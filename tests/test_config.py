"""Tests for runtime configuration loading and validation."""

import os
from pathlib import Path

import pytest

from algorix.config import DEFAULT_DB_FILENAME, Config
from algorix.exceptions import ConfigError


# --------------------------------------------------------------------------
# Positive cases
# --------------------------------------------------------------------------


def test_defaults_apply_when_env_is_empty():
    config = Config.from_env({})

    assert config.request_timeout_seconds == 30.0
    assert config.max_retries == 3
    assert config.retry_backoff_seconds == 1.0
    assert config.db_path.name == DEFAULT_DB_FILENAME


def test_env_overrides_are_applied():
    config = Config.from_env(
        {
            "ALGORIX_DATA_DIR": "/tmp/algorix-test",
            "ALGORIX_REQUEST_TIMEOUT_SECONDS": "5.5",
            "ALGORIX_MAX_RETRIES": "7",
            "ALGORIX_RETRY_BACKOFF_SECONDS": "0.25",
        }
    )

    assert config.data_dir == Path("/tmp/algorix-test")
    assert config.request_timeout_seconds == 5.5
    assert config.max_retries == 7
    assert config.retry_backoff_seconds == 0.25


def test_db_path_defaults_inside_data_dir():
    config = Config.from_env({"ALGORIX_DATA_DIR": "/tmp/algorix-test"})

    assert config.db_path == Path("/tmp/algorix-test") / DEFAULT_DB_FILENAME


def test_db_path_can_be_overridden_independently_of_data_dir():
    config = Config.from_env(
        {
            "ALGORIX_DATA_DIR": "/tmp/algorix-test",
            "ALGORIX_DB_PATH": "/tmp/elsewhere/custom.db",
        }
    )

    assert config.db_path == Path("/tmp/elsewhere/custom.db")
    assert config.data_dir == Path("/tmp/algorix-test")


def test_tilde_in_paths_is_expanded():
    config = Config.from_env({"ALGORIX_DATA_DIR": "~/algorix-test"})

    assert "~" not in str(config.data_dir)
    assert config.data_dir.is_absolute()


def test_zero_retries_is_valid():
    """Zero means "try once, do not retry" -- a legitimate choice."""
    config = Config.from_env({"ALGORIX_MAX_RETRIES": "0"})

    assert config.max_retries == 0


def test_ensure_data_dir_creates_directory(tmp_path):
    target = tmp_path / "nested" / "algorix"
    config = Config.from_env({"ALGORIX_DATA_DIR": str(target)})

    assert not target.exists()
    returned = config.ensure_data_dir()

    assert target.is_dir()
    assert returned == target


def test_ensure_data_dir_is_idempotent(tmp_path):
    config = Config.from_env({"ALGORIX_DATA_DIR": str(tmp_path / "algorix")})

    config.ensure_data_dir()
    config.ensure_data_dir()  # must not raise on an existing directory

    assert (tmp_path / "algorix").is_dir()


def test_config_is_immutable():
    """Config is frozen so a stray write cannot change behaviour mid-run."""
    config = Config.from_env({})

    with pytest.raises(Exception):
        config.max_retries = 99  # type: ignore[misc]


# --------------------------------------------------------------------------
# Negative cases
# --------------------------------------------------------------------------


def test_non_numeric_timeout_is_rejected():
    with pytest.raises(ConfigError, match="must be a number"):
        Config.from_env({"ALGORIX_REQUEST_TIMEOUT_SECONDS": "soon"})


def test_non_integer_retries_is_rejected():
    with pytest.raises(ConfigError, match="must be an integer"):
        Config.from_env({"ALGORIX_MAX_RETRIES": "3.5"})


def test_zero_timeout_is_rejected():
    """A zero timeout would fail every request instantly."""
    with pytest.raises(ConfigError, match="must be positive"):
        Config.from_env({"ALGORIX_REQUEST_TIMEOUT_SECONDS": "0"})


def test_negative_timeout_is_rejected():
    with pytest.raises(ConfigError, match="must be positive"):
        Config.from_env({"ALGORIX_REQUEST_TIMEOUT_SECONDS": "-5"})


def test_negative_retries_is_rejected():
    """Negative retries would read as "retry forever" in a naive loop."""
    with pytest.raises(ConfigError, match="cannot be negative"):
        Config.from_env({"ALGORIX_MAX_RETRIES": "-1"})


def test_negative_backoff_is_rejected():
    with pytest.raises(ConfigError, match="cannot be negative"):
        Config.from_env({"ALGORIX_RETRY_BACKOFF_SECONDS": "-0.5"})


def test_empty_string_env_var_falls_back_to_default():
    """An unset-but-exported shell variable arrives as "" -- treat as absent."""
    config = Config.from_env({"ALGORIX_MAX_RETRIES": ""})

    assert config.max_retries == 3


def test_ensure_data_dir_reports_unwritable_location(tmp_path):
    """A blocked path must raise, not silently continue without storage."""
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory")
    config = Config.from_env({"ALGORIX_DATA_DIR": str(blocker / "algorix")})

    with pytest.raises(ConfigError, match="Cannot create data directory"):
        config.ensure_data_dir()


# --------------------------------------------------------------------------
# load_dotenv_if_present
# --------------------------------------------------------------------------


def test_load_dotenv_reads_a_real_env_file(tmp_path, monkeypatch):
    """Passes an explicit `dotenv_path` deliberately -- `load_dotenv`'s
    default search walks up from config.py's own location on disk, not
    from the process cwd, so `monkeypatch.chdir` alone cannot isolate this
    test from a real `.env` that happens to sit above the repo."""
    from algorix.config import load_dotenv_if_present

    monkeypatch.delenv("ALGORIX_TEST_DOTENV_VALUE", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("ALGORIX_TEST_DOTENV_VALUE=hello\n")

    load_dotenv_if_present(dotenv_path=env_file)

    try:
        assert os.environ.get("ALGORIX_TEST_DOTENV_VALUE") == "hello"
    finally:
        monkeypatch.delenv("ALGORIX_TEST_DOTENV_VALUE", raising=False)


def test_load_dotenv_does_not_override_an_already_set_value(tmp_path, monkeypatch):
    """A value deliberately exported in the real environment must win over
    whatever a stray .env file also happens to set."""
    from algorix.config import load_dotenv_if_present

    monkeypatch.setenv("ALGORIX_TEST_DOTENV_VALUE", "from-real-env")
    env_file = tmp_path / ".env"
    env_file.write_text("ALGORIX_TEST_DOTENV_VALUE=from-dotenv\n")

    load_dotenv_if_present(dotenv_path=env_file)

    assert os.environ.get("ALGORIX_TEST_DOTENV_VALUE") == "from-real-env"


def test_load_dotenv_with_no_file_is_a_noop(tmp_path):
    from algorix.config import load_dotenv_if_present

    # A path that does not exist -- must not raise, and must not touch
    # unrelated environment state.
    load_dotenv_if_present(dotenv_path=tmp_path / "does-not-exist.env")
