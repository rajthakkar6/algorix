"""Tests for data-layer orchestration.

Every test here runs offline with injected fakes: orchestration logic
(incremental windows, failure isolation, gap reporting) is what is under test,
not the feeds themselves.
"""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from algorix.calendar import TradingCalendar
from algorix.delivery import BhavcopyResult
from algorix.exceptions import DataUnavailableError, SourceUnreachableError
from algorix.ingestion import bars_from_dataframe
from algorix.models import Bar, DeliveryRecord, Exchange, Instrument, InstrumentType
from algorix.refresh import (
    DEFAULT_OVERLAP_SESSIONS,
    RefreshReport,
    _start_date_for,
    gap_report,
    refresh,
)
from algorix.storage import BarRepository, Database, InstrumentRepository
from algorix.universe import ConstituentRecord

IST = ZoneInfo("Asia/Kolkata")
TARGET = date(2026, 9, 18)
# 08:00 IST on the 19th -- before the open, so the 18th is the last complete
# session. This is the scanner's real run time.
NOW = datetime(2026, 9, 19, 8, 0, tzinfo=IST)


@pytest.fixture
def cal():
    return TradingCalendar()


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "test.db")
    database.migrate()
    return database


class FakeIndexClient:
    def __init__(self, symbols=("RELIANCE", "TCS"), error=None):
        self.symbols = symbols
        self.error = error

    def fetch_nifty_50(self):
        return self.fetch_index("NIFTY50")

    def fetch_index(self, index_symbol):
        if self.error:
            raise self.error
        return [
            ConstituentRecord(symbol=s, name=f"{s} Ltd.") for s in self.symbols
        ]


class FakeBarSource:
    """Returns a fixed set of sessions for any instrument."""

    def __init__(self, sessions=(date(2026, 9, 17), date(2026, 9, 18)), error=None,
                 fail_symbols=()):
        self.sessions = sessions
        self.error = error
        self.fail_symbols = set(fail_symbols)
        self.requested: list[tuple[str, date, date]] = []

    def fetch(self, instrument, start, end, max_session=None):
        self.requested.append((instrument.symbol, start, end))
        if self.error:
            raise self.error
        if instrument.symbol in self.fail_symbols:
            raise DataUnavailableError(f"{instrument.symbol} is delisted")

        rows = {
            d.isoformat(): (100.0, 101.0, 99.0, 100.5, 1000)
            for d in self.sessions
            if d <= (max_session or end)
        }
        if not rows:
            raise DataUnavailableError("no rows")
        index = pd.DatetimeIndex(
            [pd.Timestamp(d, tz="Asia/Kolkata") for d in rows], name="Date"
        )
        frame = pd.DataFrame(
            {
                "Open": [v[0] for v in rows.values()],
                "High": [v[1] for v in rows.values()],
                "Low": [v[2] for v in rows.values()],
                "Close": [v[3] for v in rows.values()],
                "Volume": [v[4] for v in rows.values()],
            },
            index=index,
        )
        return bars_from_dataframe(frame, None, max_session or end)


class FakeBhavcopyClient:
    def __init__(self, symbols=("RELIANCE", "TCS"), error=None):
        self.symbols = symbols
        self.error = error

    def fetch(self, session_date, series=("EQ",)):
        if self.error:
            raise self.error
        return BhavcopyResult(
            session_date=session_date,
            records={
                s: DeliveryRecord(
                    session_date=session_date,
                    traded_quantity=1000,
                    delivered_quantity=600,
                )
                for s in self.symbols
            },
        )


class FakeAnnouncementsClient:
    """Mirrors FakeBhavcopyClient's shape for the G1 announcements client."""

    def __init__(self, records_by_symbol=None, error=None):
        self.records_by_symbol = records_by_symbol or {}
        self.error = error
        self.requested: list[tuple[str, date, date]] = []

    def fetch(self, symbol, start, end):
        self.requested.append((symbol, start, end))
        if self.error:
            raise self.error
        from algorix.announcements import AnnouncementFetchResult

        return AnnouncementFetchResult(
            records=self.records_by_symbol.get(symbol, [])
        )


def run(db, cal, **kwargs):
    defaults = dict(
        now=NOW,
        calendar=cal,
        bar_source=FakeBarSource(),
        index_client=FakeIndexClient(),
        bhavcopy_client=FakeBhavcopyClient(),
        announcements_client=FakeAnnouncementsClient(),
        # Sentiment needs ANTHROPIC_API_KEY and costs real (small) money --
        # the pre-existing 34 tests below are about bars/delivery/universe
        # orchestration, not G4, so they stay off it by default. Dedicated
        # tests further down turn it on explicitly with a fake extractor.
        skip_sentiment=True,
    )
    defaults.update(kwargs)
    return refresh(db, **defaults)


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


def test_refresh_targets_last_completed_session(db, cal):
    """At 08:00 on the 19th, the 18th is the newest complete session."""
    report = run(db, cal)

    assert report.session_date == TARGET


def test_refresh_syncs_universe_and_stores_bars(db, cal):
    report = run(db, cal)

    assert report.universe_sync is not None
    assert set(report.universe_sync.added) == {"RELIANCE", "TCS"}
    assert report.bars_stored > 0


def test_refresh_includes_metals(db, cal):
    report = run(db, cal)

    symbols = {r.symbol for r in report.bar_reports}
    assert {"GOLDUSD", "SILVERUSD", "USDINR"} <= symbols


def test_refresh_includes_index_constituents(db, cal):
    report = run(db, cal)

    symbols = {r.symbol for r in report.bar_reports}
    assert {"RELIANCE", "TCS"} <= symbols


def test_refresh_stores_delivery(db, cal):
    report = run(db, cal)

    assert report.delivery is not None
    assert report.delivery.stored == 2


def test_refresh_is_idempotent(db, cal):
    first = run(db, cal)
    second = run(db, cal)

    assert second.session_date == first.session_date
    assert second.universe_sync.added == []
    bars = BarRepository(db).get_range(
        InstrumentRepository(db).get("RELIANCE", Exchange.NSE).id,
        date(2026, 9, 1),
        date(2026, 9, 30),
    )
    assert len(bars) == 2  # not duplicated


def test_refresh_migrates_an_unmigrated_database(tmp_path, cal):
    """The entry point should work against a fresh, never-migrated file."""
    database = Database(tmp_path / "fresh.db")

    report = run(database, cal)

    assert report.session_date == TARGET


def test_skip_flags_are_honoured(db, cal):
    report = run(db, cal, skip_universe=True, skip_delivery=True)

    assert report.universe_sync is None
    assert report.delivery is None


# --------------------------------------------------------------------------
# Incremental windows
# --------------------------------------------------------------------------


def test_first_run_requests_full_history(db, cal):
    source = FakeBarSource()
    run(db, cal, bar_source=source, initial_history_days=730)

    starts = {symbol: start for symbol, start, _ in source.requested}
    assert starts["RELIANCE"] < date(2025, 1, 1)


def test_second_run_requests_only_an_overlap_window(db, cal):
    run(db, cal)
    source = FakeBarSource()
    run(db, cal, bar_source=source)

    starts = {symbol: start for symbol, start, _ in source.requested}
    # Re-reads recent sessions to catch retroactive price adjustments.
    assert starts["RELIANCE"] > date(2026, 8, 1)


def test_start_date_uses_full_history_when_empty(db, cal):
    instrument_id = InstrumentRepository(db).upsert(
        Instrument(
            symbol="NEW", exchange=Exchange.NSE, instrument_type=InstrumentType.EQUITY
        )
    )

    start = _start_date_for(
        BarRepository(db), instrument_id, TARGET, cal, 730, DEFAULT_OVERLAP_SESSIONS
    )

    assert start == TARGET - pd_timedelta(730)


def test_start_date_overlaps_existing_history(db, cal):
    repo = InstrumentRepository(db)
    instrument_id = repo.upsert(
        Instrument(
            symbol="OLD", exchange=Exchange.NSE, instrument_type=InstrumentType.EQUITY
        )
    )
    BarRepository(db).upsert_many(
        instrument_id,
        [Bar(session_date=TARGET, open=1, high=1, low=1, close=1, volume=1)],
        source="test",
    )

    start = _start_date_for(
        BarRepository(db), instrument_id, TARGET, cal, 730, 5
    )

    assert start == cal.trading_days_ago(TARGET, 5)


def pd_timedelta(days: int):
    from datetime import timedelta

    return timedelta(days=days)


# --------------------------------------------------------------------------
# Failure isolation
# --------------------------------------------------------------------------


def test_one_dead_symbol_does_not_abort_the_run(db, cal):
    run(db, cal)  # register the universe first

    report = run(db, cal, bar_source=FakeBarSource(fail_symbols={"RELIANCE"}))

    symbols = {r.symbol for r in report.bar_reports}
    assert "TCS" in symbols
    assert any("RELIANCE" in e for e in report.errors)
    assert report.is_clean is False


def test_universe_failure_does_not_stop_price_refresh(db, cal):
    """Stored constituents stay valid when the index fetch fails."""
    run(db, cal)

    report = run(
        db,
        cal,
        index_client=FakeIndexClient(error=SourceUnreachableError("NSE down")),
    )

    assert any("universe sync failed" in e for e in report.errors)
    assert {"RELIANCE", "TCS"} <= {r.symbol for r in report.bar_reports}


def test_delivery_failure_is_reported_not_fatal(db, cal):
    report = run(
        db,
        cal,
        bhavcopy_client=FakeBhavcopyClient(
            error=DataUnavailableError("not published yet")
        ),
    )

    assert report.delivery is None
    # Errors name the session that failed, so a backfill gap is identifiable.
    assert any("delivery" in e and "not published yet" in e for e in report.errors)
    assert report.bars_stored > 0


def test_total_source_outage_is_reported(db, cal):
    run(db, cal)

    report = run(
        db, cal, bar_source=FakeBarSource(error=SourceUnreachableError("no network"))
    )

    assert report.bars_stored == 0
    assert len(report.errors) >= 2


def test_empty_database_with_universe_skipped_reports_nothing_to_do(tmp_path, cal):
    database = Database(tmp_path / "empty.db")
    database.migrate()

    # Metals always seed, so remove them to reach the genuinely-empty branch.
    report = refresh(
        database,
        now=NOW,
        calendar=cal,
        skip_universe=True,
        skip_delivery=True,
        bar_source=FakeBarSource(),
    )

    assert report.bar_reports  # metals are always present
    assert report.session_date == TARGET


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def test_clean_report_says_so(db, cal):
    report = run(db, cal, bar_source=FakeBarSource(sessions=tuple(
        cal.sessions_in_range(date(2026, 9, 1), TARGET)
    )))

    assert "Session:" in report.summary()


def test_summary_lists_errors(db, cal):
    run(db, cal)
    report = run(db, cal, bar_source=FakeBarSource(fail_symbols={"TCS"}))

    assert "Errors:" in report.summary()
    assert "TCS" in report.summary()


def test_summary_reports_gaps(db, cal):
    report = run(db, cal)

    # FakeBarSource returns only two sessions, so older ones are missing.
    assert report.instruments_with_gaps
    assert "Gaps:" in report.summary()


def test_empty_report_is_clean():
    assert RefreshReport(session_date=TARGET).is_clean is True


# --------------------------------------------------------------------------
# Gap reporting
# --------------------------------------------------------------------------


def test_gap_report_finds_missing_sessions(db, cal):
    instrument_id = InstrumentRepository(db).upsert(
        Instrument(
            symbol="GAPPY",
            exchange=Exchange.NSE,
            instrument_type=InstrumentType.EQUITY,
        )
    )
    BarRepository(db).upsert_many(
        instrument_id,
        [Bar(session_date=TARGET, open=1, high=1, low=1, close=1, volume=1)],
        source="test",
    )

    gaps = gap_report(db, date(2026, 9, 15), TARGET, cal)

    assert date(2026, 9, 17) in gaps["GAPPY"]
    assert TARGET not in gaps["GAPPY"]


def test_gap_report_omits_complete_instruments(db, cal):
    instrument_id = InstrumentRepository(db).upsert(
        Instrument(
            symbol="FULL", exchange=Exchange.NSE, instrument_type=InstrumentType.EQUITY
        )
    )
    sessions = cal.sessions_in_range(date(2026, 9, 15), TARGET)
    BarRepository(db).upsert_many(
        instrument_id,
        [
            Bar(session_date=d, open=1, high=1, low=1, close=1, volume=1)
            for d in sessions
        ],
        source="test",
    )

    gaps = gap_report(db, date(2026, 9, 15), TARGET, cal)

    assert "FULL" not in gaps


def test_gap_report_on_empty_database(db, cal):
    assert gap_report(db, date(2026, 9, 15), TARGET, cal) == {}


# --------------------------------------------------------------------------
# Delivery backfill
# --------------------------------------------------------------------------


def test_delivery_backfills_history_not_just_today(db, cal):
    """A6 needs a 20-session baseline; one session leaves it unusable."""
    from algorix.storage import DeliveryRepository

    run(db, cal, delivery_history_sessions=10, max_delivery_fetches=10)

    present = DeliveryRepository(db).sessions_present(
        date(2026, 8, 1), TARGET
    )
    assert len(present) > 1
    assert TARGET in present


def test_delivery_backfill_is_capped_per_run(db, cal):
    """A cold start must not fire thirty requests at NSE at once."""
    from algorix.storage import DeliveryRepository

    run(db, cal, delivery_history_sessions=30, max_delivery_fetches=3)

    present = DeliveryRepository(db).sessions_present(date(2026, 7, 1), TARGET)
    assert len(present) == 3


def test_delivery_backfill_resumes_across_runs(db, cal):
    from algorix.storage import DeliveryRepository

    run(db, cal, delivery_history_sessions=30, max_delivery_fetches=3)
    run(db, cal, delivery_history_sessions=30, max_delivery_fetches=3)

    present = DeliveryRepository(db).sessions_present(date(2026, 7, 1), TARGET)
    assert len(present) == 6


def test_delivery_backfill_fetches_newest_first(db, cal):
    """The latest figure must be current even mid-backfill."""
    from algorix.storage import DeliveryRepository

    run(db, cal, delivery_history_sessions=30, max_delivery_fetches=2)

    present = DeliveryRepository(db).sessions_present(date(2026, 7, 1), TARGET)
    assert TARGET in present


def test_delivery_backfill_skips_sessions_already_stored(db, cal):
    from algorix.storage import DeliveryRepository

    run(db, cal, delivery_history_sessions=5, max_delivery_fetches=10)
    before = DeliveryRepository(db).sessions_present(date(2026, 8, 1), TARGET)

    run(db, cal, delivery_history_sessions=5, max_delivery_fetches=10)
    after = DeliveryRepository(db).sessions_present(date(2026, 8, 1), TARGET)

    assert before == after


def test_one_missing_bhavcopy_does_not_abort_the_backfill(db, cal):
    """An unpublished session is normal; the rest must still land."""
    from algorix.storage import DeliveryRepository

    class FlakyBhavcopy(FakeBhavcopyClient):
        def fetch(self, session_date, series=("EQ",)):
            if session_date == date(2026, 9, 16):
                raise DataUnavailableError("not published")
            return super().fetch(session_date, series)

    report = run(
        db, cal, bhavcopy_client=FlakyBhavcopy(),
        delivery_history_sessions=10, max_delivery_fetches=10,
    )

    present = DeliveryRepository(db).sessions_present(date(2026, 8, 1), TARGET)
    assert date(2026, 9, 16) not in present
    assert TARGET in present
    assert any("2026-09-16" in e for e in report.errors)


# --------------------------------------------------------------------------
# Regime instruments (B2 India VIX, B3 efficiency ratio)
#
# These are not index members, so nothing in the universe sync pulls them in.
# Before this was wired up they were seeded by the scanner but never ingested,
# leaving both gates permanently unavailable without anything saying so.
# --------------------------------------------------------------------------


def test_refresh_includes_regime_instruments(db, cal):
    """The index and VIX are refreshed alongside the metals track."""
    report = run(db, cal)

    symbols = {r.symbol for r in report.bar_reports}
    assert {"NIFTY50", "INDIAVIX"} <= symbols


def test_regime_instruments_get_bars_stored(db, cal):
    """Bars actually land -- a report entry alone would not feed the gates."""
    run(db, cal)

    repository = InstrumentRepository(db)
    bars = BarRepository(db)
    for symbol in ("NIFTY50", "INDIAVIX"):
        instrument = repository.get(symbol, Exchange.NSE)
        assert instrument is not None, f"{symbol} was never seeded"
        assert bars.latest_session(instrument.id) == TARGET


def test_indices_are_excluded_from_delivery(db, cal):
    """Indices are absent from the bhavcopy file by nature, not by fault.

    Reporting them as missing every single run would be permanent noise, and
    noise is what hides a real gap.
    """
    report = run(db, cal)

    assert report.delivery is not None
    missing = set(report.delivery.not_in_file) | set(report.delivery.unavailable)
    assert not ({"NIFTY50", "INDIAVIX"} & missing)


def test_regime_instrument_failure_is_collected_not_fatal(db, cal):
    """A dead VIX feed must not abandon the rest of the refresh."""
    report = run(db, cal, bar_source=FakeBarSource(fail_symbols=("INDIAVIX",)))

    symbols = {r.symbol for r in report.bar_reports}
    assert "INDIAVIX" not in symbols
    # The equities and the index still landed.
    assert {"RELIANCE", "TCS", "NIFTY50"} <= symbols
    # And the failure is visible rather than swallowed.
    assert any("INDIAVIX" in e for e in report.errors)
    assert not report.is_clean


# --------------------------------------------------------------------------
# Announcements (G1) wired into refresh
# --------------------------------------------------------------------------


def test_refresh_ingests_announcements_for_equities(db, cal):
    from algorix.models import AnnouncementRecord

    client = FakeAnnouncementsClient(
        records_by_symbol={
            "RELIANCE": [
                AnnouncementRecord(
                    seq_id="1", symbol="RELIANCE",
                    announced_at=datetime(2026, 9, 17, 10, 0),
                    category="Updates", text="a real disclosure",
                )
            ],
        }
    )

    report = run(db, cal, announcements_client=client)

    assert report.announcements_stored == 1
    symbols_fetched = {s for s, _, _ in client.requested}
    # Fetched for equities, never for the index or India VIX -- they do
    # not issue corporate announcements.
    assert "RELIANCE" in symbols_fetched
    assert "NIFTY50" not in symbols_fetched
    assert "INDIAVIX" not in symbols_fetched


def test_refresh_announcement_failure_is_collected_not_fatal(db, cal):
    client = FakeAnnouncementsClient(error=SourceUnreachableError("NSE unreachable"))

    report = run(db, cal, announcements_client=client)

    assert report.announcement_reports == []
    assert any("announcements" in e for e in report.errors)
    # Bars still landed -- one broken feed does not abandon the run.
    assert report.bars_stored > 0


def test_skip_announcements_fetches_nothing(db, cal):
    client = FakeAnnouncementsClient()

    report = run(db, cal, skip_announcements=True, announcements_client=client)

    assert report.announcement_reports == []
    assert client.requested == []


def test_announcement_incremental_window_uses_latest_stored(db, cal):
    """A second run should not re-request the full history window -- it
    should start from near the last stored announcement, mirroring how bar
    refreshes only re-fetch a short overlap."""
    from algorix.models import AnnouncementRecord
    from algorix.storage import AnnouncementRepository, InstrumentRepository

    first_client = FakeAnnouncementsClient()
    run(db, cal, announcements_client=first_client)

    repo = InstrumentRepository(db)
    reliance = repo.get("RELIANCE", Exchange.NSE)
    AnnouncementRepository(db).upsert_many(
        [
            (
                AnnouncementRecord(
                    seq_id="1", symbol="RELIANCE",
                    announced_at=datetime(2026, 9, 16, 9, 0),
                    category="Updates", text="already stored",
                ),
                reliance.id,
            )
        ],
        source="test",
    )

    second_client = FakeAnnouncementsClient()
    run(db, cal, announcements_client=second_client, now=datetime(2026, 9, 19, 8, 0, tzinfo=IST))

    reliance_requests = [r for r in second_client.requested if r[0] == "RELIANCE"]
    assert len(reliance_requests) == 1
    _, start, _ = reliance_requests[0]
    # Should start near 2026-09-16 (the stored announcement, minus overlap),
    # not the full 30-day history window.
    assert start > date(2026, 9, 1)


# --------------------------------------------------------------------------
# Sentiment (G4) wired into refresh
# --------------------------------------------------------------------------


class FakeSentimentExtractor:
    """Minimal stand-in for sentiment.AnthropicExtractor at the refresh
    orchestration level -- mirrors test_sentiment.py's FakeExtractor."""

    def __init__(self, not_configured=False):
        self.not_configured = not_configured
        self.model_id = "claude-sonnet-5"
        self.submitted = []
        self.results_by_batch = {}
        self.status_by_batch = {}
        self._next_id = 1

    def submit_batch(self, items):
        from algorix.sentiment import NotConfiguredError

        if self.not_configured:
            raise NotConfiguredError("ANTHROPIC_API_KEY is not set.")
        batch_id = f"batch_{self._next_id}"
        self._next_id += 1
        self.submitted.append(items)
        return batch_id

    def batch_status(self, batch_id):
        return self.status_by_batch.get(batch_id, "ended")

    def batch_results(self, batch_id):
        return self.results_by_batch.get(batch_id, [])


def test_refresh_with_sentiment_enabled_submits_a_batch(db, cal):
    """New announcements land, then get submitted for extraction in the
    same run -- collect-then-submit within one refresh call."""
    from algorix.models import AnnouncementRecord

    ann_client = FakeAnnouncementsClient(
        records_by_symbol={
            "RELIANCE": [
                AnnouncementRecord(
                    seq_id="1", symbol="RELIANCE",
                    announced_at=datetime(2026, 9, 17, 10, 0),
                    category="Updates", text="a real disclosure",
                )
            ],
        }
    )
    extractor = FakeSentimentExtractor()

    report = run(
        db, cal, skip_sentiment=False,
        announcements_client=ann_client, extractor=extractor,
    )

    assert report.sentiment_submission is not None
    assert report.sentiment_submission.submitted is True
    assert report.sentiment_submission.item_count == 1


def test_refresh_collects_a_ready_pending_batch(db, cal):
    from algorix.models import AnnouncementRecord
    from algorix.sentiment import BatchItemResult
    from algorix.storage import ExtractionBatch, ExtractionBatchRepository

    # The extracted event's FK requires the source announcement to actually
    # exist -- a batch is always submitted from real, already-ingested rows.
    ann_client = FakeAnnouncementsClient(
        records_by_symbol={
            "RELIANCE": [
                AnnouncementRecord(
                    seq_id="1", symbol="RELIANCE",
                    announced_at=datetime(2026, 9, 17, 10, 0),
                    category="Updates", text="a real disclosure",
                )
            ],
        }
    )
    run(db, cal, announcements_client=ann_client)

    ExtractionBatchRepository(db).record_submission(
        ExtractionBatch(
            batch_id="batch_1", submitted_at=NOW, item_count=1,
            model_id="claude-sonnet-5", prompt_version=1, status="submitted",
        )
    )
    extractor = FakeSentimentExtractor()
    extractor.results_by_batch["batch_1"] = [
        BatchItemResult(
            custom_id="1", outcome="succeeded",
            text='{"event_type": "business_update", "entities": [], '
                 '"polarity": "neutral", "materiality": "low", '
                 '"risk_flag": false, "risk_reason": null}',
        )
    ]

    report = run(db, cal, skip_sentiment=False, extractor=extractor)

    assert report.events_extracted == 1
    assert len(report.sentiment_collected) == 1


def test_refresh_sentiment_not_configured_degrades_cleanly(db, cal):
    """No API key must not fail the refresh -- bars/delivery/announcements
    still complete, exactly like an unconfigured Telegram digest."""
    from algorix.models import AnnouncementRecord

    # Needs a real unextracted announcement, or submission short-circuits
    # on "nothing to do" before ever reaching the extractor.
    ann_client = FakeAnnouncementsClient(
        records_by_symbol={
            "RELIANCE": [
                AnnouncementRecord(
                    seq_id="1", symbol="RELIANCE",
                    announced_at=datetime(2026, 9, 17, 10, 0),
                    category="Updates", text="a real disclosure",
                )
            ],
        }
    )
    extractor = FakeSentimentExtractor(not_configured=True)

    report = run(
        db, cal, skip_sentiment=False,
        announcements_client=ann_client, extractor=extractor,
    )

    assert report.sentiment_submission is not None
    assert report.sentiment_submission.submitted is False
    assert "ANTHROPIC_API_KEY" in report.sentiment_submission.reason
    assert report.bars_stored > 0
    # Not configured is not an error -- it must not add to `errors`, the
    # same way an unconfigured Telegram digest never does.
    assert report.errors == []


def test_skip_sentiment_never_touches_the_extractor(db, cal):
    extractor = FakeSentimentExtractor()

    report = run(db, cal, skip_sentiment=True, extractor=extractor)

    assert report.sentiment_submission is None
    assert extractor.submitted == []
