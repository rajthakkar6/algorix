"""G4: LLM structured extraction over G1 announcements (INDICATORS.md Bucket G).

Turns a raw `AnnouncementRecord` into a fixed-schema judgement --
`ExtractedEvent`: event type, entities, polarity, materiality, risk flag.
**Never a score input.** G0 is explicit and CLAUDE.md invariant 2 repeats it:
sentiment/news must never become a directional score contributor. This feeds
event detection, risk flags, and (later) contrarian-extreme context only --
the three uses G0 names, and nothing else. Nothing in `scoring.py` imports
from this module, and nothing here should ever be added to it.

**Cost, sized before building (2026-09-24):** live Nifty 50 volume over the
prior 30 days was 485 announcements / 50 stocks -- about 16/day
universe-wide. At Claude Sonnet 5 via the Batch API (50% off, ~500 input +
~200 output tokens/item), that is roughly $0.70-$1/month at observed
volume, plausibly a few dollars/month during an earnings-season spike (not
directly measured). Not the "possibly largest running cost" PROJECT_SCOPE
flags for the deferred X/social source (G3) -- this is cheap at this
universe size.

**Two-phase, not synchronous, because the Batch API is asynchronous.**
Results can take up to 24 hours, so `submit_extraction_batch` and
`collect_extraction_results` are necessarily separate operations that may
run on different days -- unlike everything else in this codebase, which
fetches and stores in one call. `ExtractionBatchRepository` is what lets a
later run find a batch this one submitted and ask whether it is ready.

**Not wired into refresh.py or scan.py.** Deliberately: the daily scan is
synchronous and does not wait on a job that can take a day to finish. How
this gets scheduled (submit nightly, collect the next run?) is an
orchestration decision left for a follow-up task, once this pipeline itself
is verified to work.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from algorix.exceptions import AlgorixError, DataIntegrityError
from algorix.models import AnnouncementRecord, EventType, ExtractedEvent, Materiality, Polarity

#: Sized and chosen 2026-09-24 -- see the module docstring for why Sonnet 5
#: over Haiku 4.5: the user's call, given the cost difference is small
#: relative to what the extra judgement is worth on short disclosure text.
DEFAULT_MODEL_ID = "claude-sonnet-5"

#: Bumped whenever the prompt or schema changes in a way that makes new
#: extractions incomparable to old ones -- same reasoning as
#: journal.SCORING_VERSION, applied one layer over.
PROMPT_VERSION = 1

#: G1 is the only source wired up so far; every row gets this. The field
#: exists on ExtractedEvent so G2 (news) and G3 (social, lower credibility)
#: slot into the same schema later without a migration.
SOURCE_CREDIBILITY_OFFICIAL = "official"

#: A nightly job should need nowhere near Anthropic's 100,000-item batch
#: cap. This bounds one run's cost and blast radius regardless of how large
#: a backlog accumulates.
DEFAULT_BATCH_LIMIT = 500


class NotConfiguredError(AlgorixError):
    """No Anthropic credentials are available in this environment.

    Mirrors notify.NotConfiguredError's role for Telegram: G4 is opt-in.
    Without a key, submission is skipped and reported as unconfigured, the
    same shape scan.py already uses for undelivered digests -- this is not
    a new failure mode for the codebase, it is the same one applied here.
    """


# ---------------------------------------------------------------------------
# Extraction schema -- the LLM-facing contract
# ---------------------------------------------------------------------------


class EventExtraction(BaseModel):
    """Structured output schema (G4). Validated against the model's response
    text; the JSON schema sent to the API is built from this by hand (see
    `_response_schema`) rather than auto-derived, so its shape is exactly
    what was tested rather than whatever Pydantic's schema generator emits
    for `Literal` fields in a given version.
    """

    model_config = ConfigDict(extra="forbid")

    event_type: EventType
    entities: list[str] = Field(default_factory=list)
    polarity: Polarity
    materiality: Materiality
    risk_flag: bool
    risk_reason: str | None = None


def _response_schema() -> dict:
    """The JSON schema sent as `output_config.format` on every request.

    Hand-written, not `EventExtraction.model_json_schema()`: Pydantic's
    generated schema for `Literal`/enum fields can nest behind `$ref`/`allOf`
    depending on version, and Anthropic's structured-output contract wants a
    flat object with `additionalProperties: false`. Writing it directly
    means there is one fewer moving part between "what was tested" and
    "what gets sent".
    """
    return {
        "type": "object",
        "properties": {
            "event_type": {"type": "string", "enum": [e.value for e in EventType]},
            "entities": {"type": "array", "items": {"type": "string"}},
            "polarity": {"type": "string", "enum": [p.value for p in Polarity]},
            "materiality": {"type": "string", "enum": [m.value for m in Materiality]},
            "risk_flag": {"type": "boolean"},
            "risk_reason": {"type": ["string", "null"]},
        },
        "required": [
            "event_type", "entities", "polarity", "materiality",
            "risk_flag", "risk_reason",
        ],
        "additionalProperties": False,
    }


_SYSTEM_PROMPT = """You classify Indian stock market corporate announcements \
(NSE/BSE official filings) into a fixed structured schema. You are not an \
investment advisor and must not suggest whether to buy or sell.

event_type -- pick exactly one, the closest fit:
  earnings_or_results   quarterly/annual results, guidance
  corporate_action      dividend, buyback, split, ESOP, allotment, restructuring
  management_change     appointment, resignation, change of director/auditor
  regulatory_or_legal   regulatory orders, litigation, SEBI/exchange action,
                        trading suspension, penalties
  merger_acquisition    M&A, acquisitions, strategic/technical tie-ups
  business_update       operational updates, investor meets, press releases,
                        orders/contracts won, production milestones
  credit_rating         rating agency actions
  shareholder_meeting   AGM/EGM notices and outcomes
  administrative        address changes, newspaper publication copies,
                        procedural filings with no business content
  other                 anything that genuinely fits nothing above

entities: company/people/regulator names explicitly mentioned. Empty list \
if none beyond the filing company itself.

polarity: positive / negative / neutral -- the tone of the disclosed fact \
itself, not a market-reaction prediction.

materiality: how significant this is to the business (low/medium/high) -- \
not how emphatically it is worded. A calm auditor-resignation notice is \
HIGH; an enthusiastic routine press release is LOW.

risk_flag: true only for signals CLAUDE.md and INDICATORS.md call out as \
high-value risk detection -- auditor resignation/change, regulatory action \
or penalty, litigation, trading suspension, credit downgrade, fraud or \
governance concerns. If true, risk_reason must say why in one sentence. If \
false, risk_reason must be null.

Base every field only on the text given. Do not infer beyond it."""


def build_user_message(announcement: AnnouncementRecord) -> str:
    return (
        f"Symbol: {announcement.symbol}\n"
        f"NSE category: {announcement.category}\n"
        f"Announcement:\n{announcement.text}"
    )


# ---------------------------------------------------------------------------
# Thin client wrapper -- the only networked piece, and the injection seam
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchItemResult:
    """One item back from a completed batch."""

    custom_id: str
    outcome: str  # "succeeded" | "errored" | "canceled" | "expired"
    text: str | None = None
    error: str | None = None


class AnthropicExtractor:
    """Wraps the Anthropic Batches API calls this module needs.

    Kept deliberately thin and narrow -- three methods, not a general
    client passthrough -- so tests can inject a fake standing in for this
    class instead of mocking the whole `anthropic.Anthropic` SDK object.
    Mirrors how `NseAnnouncementsClient`/`NseBhavcopyClient` are the
    injection seam for their modules rather than a raw `requests.Session`.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self.model_id = model_id
        self._client = None  # constructed lazily -- see _require_client

    def _require_client(self):
        if self._client is not None:
            return self._client
        if not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
            # The SDK also resolves ANTHROPIC_AUTH_TOKEN and an `ant auth
            # login` profile, which this check cannot see from here -- so a
            # missing env var does not necessarily mean no credentials
            # exist. It is, however, the common case for this tool, and
            # failing fast with a clear message beats a confusing 401 deep
            # inside a batch submission.
            raise NotConfiguredError(
                "ANTHROPIC_API_KEY is not set. G4 extraction is opt-in -- "
                "set it (or otherwise configure Anthropic credentials) to "
                "enable it."
            )
        import anthropic

        self._client = anthropic.Anthropic()
        return self._client

    def submit_batch(self, items: list[tuple[str, str]]) -> str:
        """Submit one batch. `items` is (custom_id, announcement_text)
        pairs. Returns the batch id."""
        from anthropic.types.message_create_params import (
            MessageCreateParamsNonStreaming,
        )
        from anthropic.types.messages.batch_create_params import Request

        client = self._require_client()
        requests = [
            Request(
                custom_id=custom_id,
                params=MessageCreateParamsNonStreaming(
                    model=self.model_id,
                    max_tokens=1024,
                    system=_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}],
                    output_config={
                        "format": {"type": "json_schema", "schema": _response_schema()}
                    },
                ),
            )
            for custom_id, user_message in items
        ]
        batch = client.messages.batches.create(requests=requests)
        return batch.id

    def batch_status(self, batch_id: str) -> str:
        client = self._require_client()
        return client.messages.batches.retrieve(batch_id).processing_status

    def batch_results(self, batch_id: str) -> list[BatchItemResult]:
        client = self._require_client()
        results = []
        for r in client.messages.batches.results(batch_id):
            if r.result.type == "succeeded":
                text = next(
                    (b.text for b in r.result.message.content if b.type == "text"),
                    None,
                )
                results.append(
                    BatchItemResult(custom_id=r.custom_id, outcome="succeeded", text=text)
                )
            else:
                error = getattr(getattr(r.result, "error", None), "message", None)
                results.append(
                    BatchItemResult(
                        custom_id=r.custom_id, outcome=r.result.type, error=error
                    )
                )
        return results


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubmissionReport:
    submitted: bool
    batch_id: str | None = None
    item_count: int = 0
    reason: str | None = None  # set when submitted is False


def submit_extraction_batch(
    db,
    limit: int = DEFAULT_BATCH_LIMIT,
    model_id: str = DEFAULT_MODEL_ID,
    now: datetime | None = None,
    extractor: AnthropicExtractor | None = None,
) -> SubmissionReport:
    """Find unextracted announcements and submit them as one batch.

    Returns a report rather than raising when there is nothing to do (an
    empty candidate set) or when credentials are absent -- both are
    legitimate, expected outcomes for an opt-in feature, not failures.
    """
    from algorix.storage import AnnouncementRepository, EventExtractionRepository

    now = now or datetime.now(timezone.utc)
    seq_ids = EventExtractionRepository(db).unextracted_seq_ids(limit=limit)
    if not seq_ids:
        return SubmissionReport(submitted=False, reason="no unextracted announcements")

    announcements = AnnouncementRepository(db).get_by_seq_ids(seq_ids)
    if not announcements:
        return SubmissionReport(submitted=False, reason="no unextracted announcements")

    items = [(a.seq_id, build_user_message(a)) for a in announcements]

    extractor = extractor or AnthropicExtractor(model_id=model_id)
    try:
        batch_id = extractor.submit_batch(items)
    except NotConfiguredError as exc:
        return SubmissionReport(submitted=False, reason=str(exc))

    from algorix.storage import ExtractionBatch, ExtractionBatchRepository

    ExtractionBatchRepository(db).record_submission(
        ExtractionBatch(
            batch_id=batch_id,
            submitted_at=now,
            item_count=len(items),
            model_id=model_id,
            prompt_version=PROMPT_VERSION,
            status="submitted",
        )
    )
    return SubmissionReport(submitted=True, batch_id=batch_id, item_count=len(items))


# ---------------------------------------------------------------------------
# Collect
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CollectionReport:
    batch_id: str
    #: True only once the batch has actually finished processing.
    ready: bool
    stored: int = 0
    #: (custom_id, outcome) for every item that did not store cleanly --
    #: a non-"succeeded" batch outcome, or a response that failed schema
    #: validation. Collected, never silently dropped.
    failed: list[tuple[str, str]] = field(default_factory=list)
    #: Set only when `ready` is False for a reason other than "still
    #: processing" -- e.g. credentials went missing between submission and
    #: collection. None for the ordinary still-processing case, which is
    #: not exceptional and needs no explanation.
    reason: str | None = None


def collect_extraction_results(
    db,
    batch_id: str,
    now: datetime | None = None,
    extractor: AnthropicExtractor | None = None,
) -> CollectionReport:
    """Poll one batch; store what is ready. Safe to call repeatedly --
    a batch not yet finished simply reports `ready=False` and stores
    nothing, so a scheduled job can call this every run until it is."""
    from algorix.storage import EventExtractionRepository, ExtractionBatchRepository

    now = now or datetime.now(timezone.utc)
    extractor = extractor or AnthropicExtractor()

    try:
        status = extractor.batch_status(batch_id)
    except NotConfiguredError as exc:
        # Mirrors submit_extraction_batch's handling of the same error --
        # a batch submitted while configured must not crash collection just
        # because credentials are absent by the time this runs.
        return CollectionReport(batch_id=batch_id, ready=False, reason=str(exc))

    if status != "ended":
        return CollectionReport(batch_id=batch_id, ready=False)

    events: list[ExtractedEvent] = []
    failed: list[tuple[str, str]] = []

    for item in extractor.batch_results(batch_id):
        if item.outcome != "succeeded" or item.text is None:
            failed.append((item.custom_id, item.error or item.outcome))
            continue
        try:
            parsed = EventExtraction.model_validate_json(item.text)
        except ValidationError as exc:
            failed.append((item.custom_id, f"schema validation failed: {exc}"))
            continue

        try:
            events.append(
                ExtractedEvent(
                    announcement_seq_id=item.custom_id,
                    event_type=parsed.event_type,
                    entities=parsed.entities,
                    polarity=parsed.polarity,
                    materiality=parsed.materiality,
                    source_credibility=SOURCE_CREDIBILITY_OFFICIAL,
                    risk_flag=parsed.risk_flag,
                    risk_reason=parsed.risk_reason,
                    model_id=extractor.model_id,
                    prompt_version=PROMPT_VERSION,
                    extracted_at=now,
                )
            )
        except DataIntegrityError as exc:
            # The JSON schema hints risk_reason should accompany risk_flag,
            # but a model can still violate its own requested schema (see
            # EventExtraction's docstring) -- this is the second line of
            # defence, and it must degrade to a collected failure the same
            # way a schema-validation failure does, not crash the batch.
            failed.append((item.custom_id, str(exc)))

    stored = EventExtractionRepository(db).upsert_many(events)
    ExtractionBatchRepository(db).mark_collected(batch_id, now)

    return CollectionReport(batch_id=batch_id, ready=True, stored=stored, failed=failed)
