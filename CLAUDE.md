# CLAUDE.md — Working Guide for Algorix

Read this before doing anything in this repo.

## Running it

```bash
.venv/bin/python -m algorix.refresh          # bring data up to date
.venv/bin/python -m algorix.scan --no-send   # score + print the digest
.venv/bin/python -m algorix.backtest         # IC + quintile spread, with caveats
.venv/bin/python -m algorix.web              # local dashboard on :8000
.venv/bin/python -m algorix.schedule         # preview the cron entry
.venv/bin/python -m pytest -m "not network"  # offline test suite
```

Telegram delivery needs `ALGORIX_TELEGRAM_TOKEN` and
`ALGORIX_TELEGRAM_CHAT_ID` in the environment. Without them the scan still
runs, journals, and reports delivery as unconfigured.

**A `.env` file at the repo root is picked up automatically** (every CLI
`main()` calls `config.load_dotenv_if_present()` first) -- credentials do
not need to be exported by hand every session. `.env` is gitignored; a
value already exported in the real environment always wins over one in the
file.

`refresh` now also ingests corporate announcements (G1) and drives G4
sentiment extraction every run: it collects any batch submitted by an
earlier run that has finished, then submits whatever is newly unextracted.
`--skip-announcements` and `--skip-sentiment` disable each independently.

**G4 supports multiple LLM providers, one active at a time** (Sep 2026,
to avoid lock-in -- not a fallback chain, not a compare-both harness).
`ALGORIX_LLM_PROVIDER` picks `anthropic` (default, needs
`ANTHROPIC_API_KEY`, the SDK's own standard env var) or `openai` (needs
`OPENAI_API_KEY`). `ALGORIX_LLM_MODEL` overrides that provider's default
model. Without credentials for whichever provider is active, `refresh`
still completes normally and reports the sentiment step as unconfigured,
same shape as an unconfigured Telegram digest; an unrecognised
`ALGORIX_LLM_PROVIDER` value is reported as an error but does not abort
the rest of the run (bars/delivery/announcements/earnings already
succeeded by that point). See `sentiment.Extractor` for the protocol a
third provider would need to implement, and `sentiment.build_extractor`
for the selection logic.

Module map: `calendar` → `storage` → `universe`/`ingestion`/`delivery`/
`metals`/`announcements`/`earnings` → `refresh` (data layer); `series` →
`indicators`/`cross_sectional`/`regime` → `scoring` → `journal`/`notify` →
`scan` (analysis layer); `backtest` and `web` both sit on top and read the
same store. `sentiment` sits beside `scoring`, not inside it, and is driven
by `refresh`, not by `scan` -- the daily scan needs a score right now; G4's
Batch API can take up to 24h, so it runs on its own submit-then-collect
cadence across successive `refresh` calls instead.

`earnings` (INDICATORS.md A8, PEAD) fetches real earnings-surprise data via
`yfinance` and feeds `indicators.pead_signal`, which **is** wired into
`scoring.score_universe` (`SCORING_VERSION` 3) -- unlike `announcements`/
`sentiment`, this one directly affects the score, because it is a
quantitative price/earnings signal, not news/sentiment (invariant 2 is
about sentiment specifically, not every non-price data source).

`announcements` (INDICATORS.md G1) is data plumbing only -- structured NSE
corporate filings, fetched and stored, never scored. `sentiment`
(INDICATORS.md G4) turns those into a structured judgement (event type,
polarity, materiality, risk flag) via the Anthropic Batch API -- sized at
~$1/month at observed Nifty 50 volume before it was built (see the module
docstring). **Nothing in `sentiment.py` is imported by `scoring.py`, and
nothing should be** -- sentiment/news must never become a directional score
input, see invariant 2.

`refresh` is the only module that ingests bars. Anything the analysis layer
reads a price series for -- index, VIX, metals -- must be registered there,
or its signal goes silently unavailable forever.

## What this project is

A personal swing-trading (days–weeks) analysis and scoring tool for Indian
equities (NSE/BSE) + gold/silver. It scores opportunities across a universe
and explains *why*, so a trader sees what they'd otherwise miss.

Design documents — read before implementing anything they cover:
- [PROJECT_SCOPE.md](PROJECT_SCOPE.md) — features, architecture, open decisions
- [INDICATORS.md](INDICATORS.md) — the indicator set and the reasoning behind
  every inclusion and rejection

---

## How we work — non-negotiable

### 1. Plan before building. Never build blindly.
Before writing code for anything: state what is being built, how it fits the
existing design, and what could go wrong. If the plan can't be stated
clearly, it isn't ready to build.

If implementation reveals the plan was wrong — **stop and say so**. Do not
quietly build something different from what was agreed.

### 2. Split features into small tasks. One task at a time.
Break every feature into tasks small enough to finish and verify
individually. Complete and test one before starting the next. No
half-finished parallel work.

### 3. Every task gets positive *and* negative test cases.
Both, always:
- **Positive** — correct inputs produce correct outputs.
- **Negative** — bad inputs, missing data, API failures, empty responses,
  malformed feeds, holidays/non-trading days, rate limits.

Negative cases matter more than usual here: this system depends on external
feeds that fail, lag, and return garbage. A signal computed from bad data is
worse than no signal, because it looks valid.

### 4. Direction check between tasks.
Before picking up the next task, ask: *is this still going the right way?*
Check the new work against the design docs and the goal. Catching drift one
task in is cheap; catching it five tasks in is not. Raise it immediately if
something feels off-course.

### 5. Test at the end of every feature.
Not just the unit tests per task — verify the feature works end to end
before calling it done. Report results honestly: if tests fail, say so and
show the output. Never describe something as working that hasn't been run.

---

## Design invariants — do not violate silently

These came out of deliberate analysis (rationale in the design docs). If one
needs to change, raise it and discuss — never just code around it.

1. **The LLM never overrides the quantitative score.** The quant score is the
   primary, backtestable number. LLM output is a separate labelled layer. On
   disagreement, surface the conflict — never average them.
   (PROJECT_SCOPE §3.7.1)

2. **Sentiment is never a directional score contributor.** It feeds event
   detection, risk flags, and contrarian extremes only. "Positive mentions →
   higher score" is explicitly rejected — it's the most manipulable input in
   the system. (INDICATORS.md §G0)

3. **The score is daily-cadence. Real-time is for monitoring and triggers,
   not re-scoring.** Delivery % and FII/DII are only published post-close, so
   an intraday score would be stale while looking precise.
   (PROJECT_SCOPE §3.5)

4. **Eligibility gates are not score contributors.** A stock failing
   liquidity / circuit-lock / F&O-ban / pledge checks is reported
   **ineligible**, not scored low. (INDICATORS.md §0.2b)

5. **Scores computed by different methods must be labelled.** A cross-
   sectional Nifty 50 score and a self-history fallback score are not
   comparable and must never be silently shown as the same thing.
   (INDICATORS.md §0.2a)

6. **The indicator set is deliberate — both what's in and what's out.**
   RSI, MACD, Stochastics, Bollinger Bands, ADX, Ichimoku and others were
   rejected for specific reasons (redundancy, or conflict with momentum). Do
   not reintroduce them because they're familiar. (INDICATORS.md Bucket E)

7. **Log everything the journal needs to stay interpretable.** Every scan
   result, score breakdown, and LLM verdict — with prompt version and model
   ID. Without this, months of journal data can't be evaluated because you
   can't tell what changed underneath it.

8. **Rank cross-sectionally, don't threshold absolutely.** Percentile within
   universe, not "RSI > 70"-style fixed cutoffs. (INDICATORS.md §0.2)

---

## Data handling rules

- **Never silently swallow a data-source failure.** A missing feed must
  surface as a visible gap, not a zero or a stale value.
- **Never fabricate or interpolate market data.** If delivery % is
  unavailable for a day, that signal is unavailable — not estimated.
- **Respect API rate limits explicitly.** Broker APIs cap both WebSocket
  subscriptions and REST throughput; budget them rather than discovering
  limits in production. (PROJECT_SCOPE §4.1)
- **Non-trading days are a real case, not an edge case.** Weekends, market
  holidays, and mid-session halts must be handled in every scheduled job.

---

## Reminders

- This is **personal-use only**. If that ever changes, SEBI Research Analyst
  regulations become relevant — raise it before building anything outward-facing.
- Costs are real and recurring (broker API, LLM spend, possibly X API). Flag
  anything that would materially increase running cost before building it.
- The tool **scores and explains** — it is not a black-box signal. Every
  score must be traceable to the factors that produced it.
