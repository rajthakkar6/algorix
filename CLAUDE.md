# CLAUDE.md — Working Guide for Algorix

Read this before doing anything in this repo.

## Running it

```bash
.venv/bin/python -m algorix.refresh          # bring data up to date
.venv/bin/python -m algorix.scan --no-send   # score + print the digest
.venv/bin/python -m algorix.schedule         # preview the cron entry
.venv/bin/python -m pytest -m "not network"  # offline test suite
```

Telegram delivery needs `ALGORIX_TELEGRAM_TOKEN` and
`ALGORIX_TELEGRAM_CHAT_ID` in the environment. Without them the scan still
runs, journals, and reports delivery as unconfigured.

Module map: `calendar` → `storage` → `universe`/`ingestion`/`delivery`/
`metals` → `refresh` (data layer); `series` → `indicators`/`cross_sectional`/
`regime` → `scoring` → `journal`/`notify` → `scan` (analysis layer).

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
