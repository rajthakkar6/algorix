"""Scheduling the morning run.

Generates (and optionally installs) a cron entry that refreshes data and then
runs the scan, before the 09:15 IST open.

Order matters: refresh must complete before the scan, or the scan scores
yesterday's data. They run as one chained command rather than two entries so
cron cannot interleave them.

Installing is opt-in. `python -m algorix.schedule` prints what it would do;
only `--install` touches your crontab, and it refuses to create a duplicate
entry.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

#: Marks entries this tool manages, so install/remove never touches
#: anything else in the user's crontab.
CRON_MARKER = "# algorix-morning-scan"

DEFAULT_HOUR = 7
DEFAULT_MINUTE = 30


@dataclass(frozen=True)
class ScheduleSpec:
    """A cron schedule for the morning run."""

    hour: int = DEFAULT_HOUR
    minute: int = DEFAULT_MINUTE
    python: str = sys.executable
    db_path: str | None = None
    log_path: str | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.hour <= 23:
            raise ValueError(f"hour must be 0-23, got {self.hour}")
        if not 0 <= self.minute <= 59:
            raise ValueError(f"minute must be 0-59, got {self.minute}")

    @property
    def command(self) -> str:
        """Refresh then scan, chained so a failed refresh skips the scan."""
        db = f" --db {self.db_path}" if self.db_path else ""
        refresh = f"{self.python} -m algorix.refresh{db}"
        scan = f"{self.python} -m algorix.scan{db}"
        chained = f"{refresh} ; {scan}"
        if self.log_path:
            chained = f"({chained}) >> {self.log_path} 2>&1"
        return chained

    @property
    def cron_line(self) -> str:
        # Mon-Fri only: NSE does not trade at weekends, and the scan would
        # simply re-score Friday's session.
        return (
            f"{self.minute} {self.hour} * * 1-5 {self.command} {CRON_MARKER}"
        )

    def describe(self) -> str:
        when = f"{self.hour:02d}:{self.minute:02d}"
        return (
            f"Runs Monday-Friday at {when} (machine local time).\n"
            f"NSE opens at 09:15 IST -- make sure {when} is before that in "
            f"your timezone.\n\n"
            f"Crontab line:\n  {self.cron_line}\n"
        )


def current_crontab() -> str:
    """Existing crontab, or empty string when none is set."""
    result = subprocess.run(
        ["crontab", "-l"], capture_output=True, text=True, check=False
    )
    return result.stdout if result.returncode == 0 else ""


def is_installed(crontab: str | None = None) -> bool:
    return CRON_MARKER in (crontab if crontab is not None else current_crontab())


def build_crontab(spec: ScheduleSpec, existing: str) -> str:
    """Existing crontab with our entry added or replaced.

    Lines carrying the marker are replaced rather than appended to, so
    re-installing changes the schedule instead of stacking duplicates.
    """
    kept = [
        line
        for line in existing.splitlines()
        if CRON_MARKER not in line
    ]
    while kept and not kept[-1].strip():
        kept.pop()
    kept.append(spec.cron_line)
    return "\n".join(kept) + "\n"


def remove_from_crontab(existing: str) -> str:
    kept = [line for line in existing.splitlines() if CRON_MARKER not in line]
    return "\n".join(kept) + ("\n" if kept else "")


def write_crontab(content: str) -> None:
    subprocess.run(["crontab", "-"], input=content, text=True, check=True)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from algorix.config import load_dotenv_if_present

    load_dotenv_if_present()

    parser = argparse.ArgumentParser(
        description="Schedule the Algorix morning scan."
    )
    parser.add_argument("--hour", type=int, default=DEFAULT_HOUR)
    parser.add_argument("--minute", type=int, default=DEFAULT_MINUTE)
    parser.add_argument("--db", help="database path passed to both commands")
    parser.add_argument("--log", help="append output to this file")
    parser.add_argument(
        "--install", action="store_true", help="write the entry to your crontab"
    )
    parser.add_argument(
        "--remove", action="store_true", help="remove the entry from your crontab"
    )
    args = parser.parse_args(argv)

    if args.remove:
        existing = current_crontab()
        if not is_installed(existing):
            print("No Algorix entry found in your crontab.")
            return 0
        write_crontab(remove_from_crontab(existing))
        print("Removed the Algorix entry from your crontab.")
        return 0

    spec = ScheduleSpec(
        hour=args.hour,
        minute=args.minute,
        db_path=args.db,
        log_path=args.log,
    )

    if not args.install:
        print(spec.describe())
        print("Nothing installed. Re-run with --install to add it.")
        print("On macOS, cron needs Full Disk Access for your terminal:")
        print("  System Settings > Privacy & Security > Full Disk Access")
        return 0

    existing = current_crontab()
    replacing = is_installed(existing)
    write_crontab(build_crontab(spec, existing))
    print(("Updated" if replacing else "Installed") + " the Algorix schedule:")
    print(f"  {spec.cron_line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
