"""Tests for cron scheduling."""

import pytest

from algorix.schedule import (
    CRON_MARKER,
    ScheduleSpec,
    build_crontab,
    is_installed,
    remove_from_crontab,
)


def test_cron_line_carries_the_marker():
    assert CRON_MARKER in ScheduleSpec().cron_line


def test_cron_line_runs_weekdays_only():
    """NSE does not trade at weekends."""
    assert "* * 1-5" in ScheduleSpec().cron_line


def test_schedule_time_is_reflected():
    spec = ScheduleSpec(hour=6, minute=45)

    assert spec.cron_line.startswith("45 6 ")


def test_refresh_runs_before_scan():
    """Scanning before refreshing would score yesterday's data."""
    command = ScheduleSpec().command

    assert command.index("algorix.refresh") < command.index("algorix.scan")


def test_db_path_is_passed_to_both_commands():
    command = ScheduleSpec(db_path="/tmp/a.db").command

    assert command.count("--db /tmp/a.db") == 2


def test_log_path_redirects_output():
    command = ScheduleSpec(log_path="/tmp/a.log").command

    assert ">> /tmp/a.log 2>&1" in command


def test_invalid_hour_is_rejected():
    with pytest.raises(ValueError, match="hour must be 0-23"):
        ScheduleSpec(hour=25)


def test_invalid_minute_is_rejected():
    with pytest.raises(ValueError, match="minute must be 0-59"):
        ScheduleSpec(minute=60)


def test_describe_warns_about_market_open():
    assert "09:15" in ScheduleSpec().describe()


# --- crontab manipulation -------------------------------------------------


def test_install_into_empty_crontab():
    result = build_crontab(ScheduleSpec(), "")

    assert CRON_MARKER in result
    assert result.endswith("\n")


def test_install_preserves_other_entries():
    existing = "0 9 * * * /usr/bin/backup.sh\n"

    result = build_crontab(ScheduleSpec(), existing)

    assert "/usr/bin/backup.sh" in result
    assert CRON_MARKER in result


def test_reinstall_replaces_rather_than_duplicates():
    first = build_crontab(ScheduleSpec(hour=7), "")
    second = build_crontab(ScheduleSpec(hour=6), first)

    assert second.count(CRON_MARKER) == 1
    assert "0 6 " in second or " 6 " in second


def test_remove_deletes_only_our_entry():
    existing = build_crontab(ScheduleSpec(), "0 9 * * * /usr/bin/backup.sh\n")

    result = remove_from_crontab(existing)

    assert CRON_MARKER not in result
    assert "/usr/bin/backup.sh" in result


def test_remove_from_crontab_without_entry_is_safe():
    existing = "0 9 * * * /usr/bin/backup.sh\n"

    assert remove_from_crontab(existing).strip() == existing.strip()


def test_is_installed_detects_the_marker():
    assert is_installed(build_crontab(ScheduleSpec(), "")) is True
    assert is_installed("0 9 * * * other\n") is False
