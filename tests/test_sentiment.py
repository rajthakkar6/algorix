"""Tests for G4 LLM structured extraction (INDICATORS.md G4).

Every test here is offline: `FakeExtractor` stands in for `AnthropicExtractor`
(the injection seam), so nothing makes a real Anthropic API call or spends
real money. This mirrors how `ingest_delivery`/`ingest_announcements` accept
an already-parsed `result` to bypass their network client in tests.
"""

import json
from datetime import date, datetime, timezone

import pytest

from algorix.exceptions import DataIntegrityError
from algorix.models import (
    AnnouncementRecord,
    EventType,
    Exchange,
    ExtractedEvent,
    Instrument,
    InstrumentType,
    Materiality,
    Polarity,
)
from algorix.sentiment import (
    DEFAULT_MODEL_ID,
    PROMPT_VERSION,
    BatchItemResult,
    EventExtraction,
    NotConfiguredError,
    build_user_message,
    collect_extraction_results,
    submit_extraction_batch,
)
from algorix.storage import (
    AnnouncementRepository,
    Database,
    EventExtractionRepository,
    ExtractionBatch,
    ExtractionBatchRepository,
    InstrumentRepository,
)

NOW = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "test.db")
    database.migrate()
    return database


@pytest.fixture
def reliance_id(db):
    return InstrumentRepository(db).upsert(
        Instrument(
            symbol="RELIANCE",
            exchange=Exchange.NSE,
            instrument_type=InstrumentType.EQUITY,
        )
    )


def seed_announcements(db, instrument_id, seq_ids):
    records = [
        AnnouncementRecord(
            seq_id=seq_id,
            symbol="RELIANCE",
            announced_at=datetime(2026, 9, 20 + i),
            category="Updates",
            text=f"disclosure text {seq_id}",
        )
        for i, seq_id in enumerate(seq_ids)
    ]
    AnnouncementRepository(db).upsert_many(
        ((r, instrument_id) for r in records), source="test"
    )
    return records


def payload(
    event_type="business_update",
    entities=None,
    polarity="neutral",
    materiality="low",
    risk_flag=False,
    risk_reason=None,
):
    return json.dumps(
        {
            "event_type": event_type,
            "entities": entities or [],
            "polarity": polarity,
            "materiality": materiality,
            "risk_flag": risk_flag,
            "risk_reason": risk_reason,
        }
    )


class FakeExtractor:
    """Stands in for AnthropicExtractor. `results_by_batch` maps a batch id
    to the list[BatchItemResult] `collect` should return for it; `status`
    maps a batch id to its processing_status (defaults to "ended" once
    results are registered)."""

    def __init__(self, not_configured: bool = False):
        self.not_configured = not_configured
        self.model_id = DEFAULT_MODEL_ID
        self.submitted: list[list[tuple[str, str]]] = []
        self.results_by_batch: dict[str, list[BatchItemResult]] = {}
        self.status_by_batch: dict[str, str] = {}
        self._next_id = 1

    def submit_batch(self, items):
        if self.not_configured:
            raise NotConfiguredError("ANTHROPIC_API_KEY is not set.")
        batch_id = f"batch_{self._next_id}"
        self._next_id += 1
        self.submitted.append(items)
        return batch_id

    def batch_status(self, batch_id):
        if self.not_configured:
            raise NotConfiguredError("ANTHROPIC_API_KEY is not set.")
        return self.status_by_batch.get(batch_id, "ended")

    def batch_results(self, batch_id):
        return self.results_by_batch.get(batch_id, [])


# ---------------------------------------------------------------------------
# build_user_message
# ---------------------------------------------------------------------------


def test_build_user_message_includes_symbol_category_and_text():
    ann = AnnouncementRecord(
        seq_id="1", symbol="RELIANCE", announced_at=datetime(2026, 9, 20),
        category="Updates", text="something happened",
    )
    msg = build_user_message(ann)

    assert "RELIANCE" in msg
    assert "Updates" in msg
    assert "something happened" in msg


# ---------------------------------------------------------------------------
# submit_extraction_batch -- positive
# ---------------------------------------------------------------------------


def test_submit_batches_all_unextracted_announcements(db, reliance_id):
    seed_announcements(db, reliance_id, ["1", "2", "3"])
    extractor = FakeExtractor()

    report = submit_extraction_batch(db, now=NOW, extractor=extractor)

    assert report.submitted is True
    assert report.item_count == 3
    assert len(extractor.submitted[0]) == 3
    submitted_ids = {custom_id for custom_id, _ in extractor.submitted[0]}
    assert submitted_ids == {"1", "2", "3"}


def test_submit_records_the_batch_for_later_collection(db, reliance_id):
    seed_announcements(db, reliance_id, ["1"])
    extractor = FakeExtractor()

    report = submit_extraction_batch(db, now=NOW, extractor=extractor)

    pending = ExtractionBatchRepository(db).pending()
    assert len(pending) == 1
    assert pending[0].batch_id == report.batch_id
    assert pending[0].item_count == 1
    assert pending[0].prompt_version == PROMPT_VERSION


def test_submit_respects_limit(db, reliance_id):
    seed_announcements(db, reliance_id, ["1", "2", "3", "4", "5"])
    extractor = FakeExtractor()

    report = submit_extraction_batch(db, limit=2, now=NOW, extractor=extractor)

    assert report.item_count == 2


def test_submit_excludes_already_extracted(db, reliance_id):
    seed_announcements(db, reliance_id, ["1", "2"])
    EventExtractionRepository(db).upsert_many(
        [
            ExtractedEvent(
                announcement_seq_id="1", event_type=EventType.OTHER, entities=[],
                polarity=Polarity.NEUTRAL, materiality=Materiality.LOW,
                source_credibility="official", risk_flag=False, risk_reason=None,
                model_id=DEFAULT_MODEL_ID, prompt_version=PROMPT_VERSION,
                extracted_at=NOW,
            )
        ]
    )
    extractor = FakeExtractor()

    report = submit_extraction_batch(db, now=NOW, extractor=extractor)

    assert report.item_count == 1
    assert extractor.submitted[0][0][0] == "2"


# ---------------------------------------------------------------------------
# submit_extraction_batch -- negative
# ---------------------------------------------------------------------------


def test_submit_with_nothing_to_do_does_not_call_the_extractor(db):
    extractor = FakeExtractor()

    report = submit_extraction_batch(db, now=NOW, extractor=extractor)

    assert report.submitted is False
    assert "no unextracted" in report.reason
    assert extractor.submitted == []


def test_submit_when_not_configured_reports_not_raises(db, reliance_id):
    """An opt-in feature with no credentials must degrade to a clear report,
    not crash the caller -- mirrors notify.py's Telegram-unconfigured shape
    exactly."""
    seed_announcements(db, reliance_id, ["1"])
    extractor = FakeExtractor(not_configured=True)

    report = submit_extraction_batch(db, now=NOW, extractor=extractor)

    assert report.submitted is False
    assert "ANTHROPIC_API_KEY" in report.reason
    # And nothing was recorded as submitted -- a failed submission must not
    # look like a real, pollable batch.
    assert ExtractionBatchRepository(db).pending() == []


# ---------------------------------------------------------------------------
# collect_extraction_results -- positive
# ---------------------------------------------------------------------------


def test_collect_stores_successful_extractions(db, reliance_id):
    seed_announcements(db, reliance_id, ["1"])
    extractor = FakeExtractor()
    extractor.results_by_batch["batch_1"] = [
        BatchItemResult(custom_id="1", outcome="succeeded", text=payload())
    ]

    report = collect_extraction_results(db, "batch_1", now=NOW, extractor=extractor)

    assert report.ready is True
    assert report.stored == 1
    assert report.failed == []
    stored = EventExtractionRepository(db).get("1")
    assert stored is not None
    assert stored.event_type == EventType.BUSINESS_UPDATE
    assert stored.model_id == DEFAULT_MODEL_ID
    assert stored.prompt_version == PROMPT_VERSION


def test_collect_marks_the_batch_collected(db, reliance_id):
    seed_announcements(db, reliance_id, ["1"])
    ExtractionBatchRepository(db).record_submission(
        ExtractionBatch(
            batch_id="batch_1", submitted_at=NOW, item_count=1,
            model_id=DEFAULT_MODEL_ID, prompt_version=PROMPT_VERSION,
            status="submitted",
        )
    )
    extractor = FakeExtractor()
    extractor.results_by_batch["batch_1"] = [
        BatchItemResult(custom_id="1", outcome="succeeded", text=payload())
    ]

    collect_extraction_results(db, "batch_1", now=NOW, extractor=extractor)

    assert ExtractionBatchRepository(db).pending() == []


def test_collect_stores_a_risk_flag_with_its_reason(db, reliance_id):
    seed_announcements(db, reliance_id, ["1"])
    extractor = FakeExtractor()
    extractor.results_by_batch["batch_1"] = [
        BatchItemResult(
            custom_id="1", outcome="succeeded",
            text=payload(
                event_type="regulatory_or_legal", polarity="negative",
                materiality="high", risk_flag=True,
                risk_reason="SEBI order restricting trading",
            ),
        )
    ]

    collect_extraction_results(db, "batch_1", now=NOW, extractor=extractor)

    stored = EventExtractionRepository(db).get("1")
    assert stored.risk_flag is True
    assert stored.risk_reason == "SEBI order restricting trading"


def test_collect_stores_multiple_succeeded_items(db, reliance_id):
    seed_announcements(db, reliance_id, ["1", "2"])
    extractor = FakeExtractor()
    extractor.results_by_batch["batch_1"] = [
        BatchItemResult(custom_id="1", outcome="succeeded", text=payload()),
        BatchItemResult(custom_id="2", outcome="succeeded", text=payload()),
    ]

    report = collect_extraction_results(db, "batch_1", now=NOW, extractor=extractor)

    assert report.stored == 2


# ---------------------------------------------------------------------------
# collect_extraction_results -- negative
# ---------------------------------------------------------------------------


def test_collect_when_batch_still_processing_stores_nothing(db, reliance_id):
    seed_announcements(db, reliance_id, ["1"])
    extractor = FakeExtractor()
    extractor.status_by_batch["batch_1"] = "in_progress"

    report = collect_extraction_results(db, "batch_1", now=NOW, extractor=extractor)

    assert report.ready is False
    assert report.stored == 0
    assert EventExtractionRepository(db).get("1") is None


def test_collect_when_not_configured_reports_not_raises(db, reliance_id):
    """A batch submitted while configured must not crash collection if
    credentials are absent by the time collection runs -- mirrors
    submit_extraction_batch's handling of the same error exactly."""
    seed_announcements(db, reliance_id, ["1"])
    extractor = FakeExtractor(not_configured=True)

    report = collect_extraction_results(db, "batch_1", now=NOW, extractor=extractor)

    assert report.ready is False
    assert "ANTHROPIC_API_KEY" in report.reason
    assert EventExtractionRepository(db).get("1") is None


def test_collect_a_still_processing_batch_stays_pending(db):
    ExtractionBatchRepository(db).record_submission(
        ExtractionBatch(
            batch_id="batch_1", submitted_at=NOW, item_count=1,
            model_id=DEFAULT_MODEL_ID, prompt_version=PROMPT_VERSION,
            status="submitted",
        )
    )
    extractor = FakeExtractor()
    extractor.status_by_batch["batch_1"] = "in_progress"

    collect_extraction_results(db, "batch_1", now=NOW, extractor=extractor)

    assert len(ExtractionBatchRepository(db).pending()) == 1


def test_collect_reports_errored_items_without_crashing(db, reliance_id):
    seed_announcements(db, reliance_id, ["1", "2"])
    extractor = FakeExtractor()
    extractor.results_by_batch["batch_1"] = [
        BatchItemResult(custom_id="1", outcome="errored", error="internal error"),
        BatchItemResult(custom_id="2", outcome="succeeded", text=payload()),
    ]

    report = collect_extraction_results(db, "batch_1", now=NOW, extractor=extractor)

    assert report.stored == 1
    assert report.failed == [("1", "internal error")]
    assert EventExtractionRepository(db).get("1") is None
    assert EventExtractionRepository(db).get("2") is not None


def test_collect_reports_canceled_and_expired_items(db, reliance_id):
    seed_announcements(db, reliance_id, ["1", "2"])
    extractor = FakeExtractor()
    extractor.results_by_batch["batch_1"] = [
        BatchItemResult(custom_id="1", outcome="canceled"),
        BatchItemResult(custom_id="2", outcome="expired"),
    ]

    report = collect_extraction_results(db, "batch_1", now=NOW, extractor=extractor)

    assert report.stored == 0
    assert set(report.failed) == {("1", "canceled"), ("2", "expired")}


def test_collect_reports_malformed_json_without_crashing(db, reliance_id):
    seed_announcements(db, reliance_id, ["1"])
    extractor = FakeExtractor()
    extractor.results_by_batch["batch_1"] = [
        BatchItemResult(custom_id="1", outcome="succeeded", text="not json at all")
    ]

    report = collect_extraction_results(db, "batch_1", now=NOW, extractor=extractor)

    assert report.stored == 0
    assert report.failed[0][0] == "1"
    assert "schema validation failed" in report.failed[0][1]


def test_collect_reports_invalid_enum_value_without_crashing(db, reliance_id):
    """A model response that violates its own requested schema (should be
    rare given output_config.format, but must not be trusted blindly)."""
    seed_announcements(db, reliance_id, ["1"])
    extractor = FakeExtractor()
    extractor.results_by_batch["batch_1"] = [
        BatchItemResult(
            custom_id="1", outcome="succeeded",
            text=payload(event_type="not_a_real_event_type"),
        )
    ]

    report = collect_extraction_results(db, "batch_1", now=NOW, extractor=extractor)

    assert report.stored == 0
    assert report.failed[0][0] == "1"


def test_collect_reports_risk_flag_without_reason_as_failed(db, reliance_id):
    """ExtractedEvent's own validation (risk_flag=True needs a reason) is a
    second line of defence beneath the JSON schema -- this must surface as
    a collected failure, not an unhandled DataIntegrityError."""
    seed_announcements(db, reliance_id, ["1"])
    extractor = FakeExtractor()
    extractor.results_by_batch["batch_1"] = [
        BatchItemResult(
            custom_id="1", outcome="succeeded",
            text=payload(risk_flag=True, risk_reason=None),
        )
    ]

    report = collect_extraction_results(db, "batch_1", now=NOW, extractor=extractor)

    assert report.stored == 0
    assert report.failed[0][0] == "1"


def test_collect_with_empty_results_reports_nothing_stored(db):
    extractor = FakeExtractor()
    extractor.results_by_batch["batch_1"] = []

    report = collect_extraction_results(db, "batch_1", now=NOW, extractor=extractor)

    assert report.ready is True
    assert report.stored == 0
    assert report.failed == []


# ---------------------------------------------------------------------------
# EventExtraction schema validation -- negative
# ---------------------------------------------------------------------------


def test_event_extraction_rejects_missing_required_field():
    with pytest.raises(Exception):
        EventExtraction.model_validate_json(
            json.dumps({"event_type": "other", "entities": []})
        )


def test_event_extraction_rejects_unknown_field():
    with pytest.raises(Exception):
        EventExtraction.model_validate_json(
            payload().replace("}", ', "extra_field": "nope"}')
        )


# ---------------------------------------------------------------------------
# AnthropicExtractor -- negative (credential resolution)
# ---------------------------------------------------------------------------


def test_extractor_raises_not_configured_without_api_key(monkeypatch):
    from algorix.sentiment import AnthropicExtractor

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    extractor = AnthropicExtractor()

    with pytest.raises(NotConfiguredError, match="ANTHROPIC_API_KEY"):
        extractor.submit_batch([("1", "some text")])
