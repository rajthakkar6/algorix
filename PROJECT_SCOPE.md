# Algorix — Project Scope

An AI-assisted trading analysis tool for personal use. It scores tradeable
opportunities across a defined universe so that signals a trader could miss
in the noise of daily market information get surfaced automatically.

This document is the working scope for review before any code is written.
Sections marked **[OPEN]** are decisions still needed.

---

## 0. Build Status

**The MVP morning scanner is built and working end to end** (485 tests),
and has now been run against live data: 57 instruments, ~28.5k bars, a
full 31-session delivery window, and all four regime gates reporting.

| Layer | Status |
|---|---|
| Data layer — calendar, storage, universe, prices, delivery, metals, orchestration | ✅ |
| Indicators — A1–A7 score contributors, C1–C2 risk inputs | ✅ |
| Regime gates — B1 breadth, B2 India VIX, B3 efficiency ratio, B4 FII/DII | ✅ |
| Composite scoring + eligibility gates + ATR risk sizing | ✅ |
| Trade journal | ✅ |
| Telegram digest | ✅ (needs credentials) |
| Cron scheduling | ✅ (opt-in install) |
| Local web UI — dashboard, stock detail, backtest, journal | ✅ |
| Backtest / signal evaluation (IC, quintile spread) | ✅ |
| News/sentiment — G1 NSE announcements + G4 LLM extraction | ✅ (wired into `refresh`; needs `ANTHROPIC_API_KEY` for G4) |

Not yet built: watchlist background jobs (§3.4), real-time lookup (§3.5),
G2/G3 sources (§3.6), LLM concluder (§3.7).

**§3.6 in progress (Sep 2026).** G1 (NSE/BSE corporate announcements) is
built: `announcements.py` fetches, parses and stores structured filings per
instrument, keyed on NSE's own `seq_id` for idempotent upserts. No
truncation defect was found on this endpoint after testing 30–365 day
lookback windows (a false positive from an earlier test script, corrected
before shipping — see the module docstring).

G4 (LLM structured extraction) is also built: `sentiment.py` turns each
announcement into event_type/entities/polarity/materiality/risk_flag via
Claude Sonnet 5 on the Batch API (user's choice over Haiku 4.5 after
sizing). Cost was measured, not guessed, before building: 485 real
announcements across the Nifty 50 over 30 days (~16/day universe-wide) →
roughly $0.70–$1/month at Sonnet 5 batch rates, a few dollars/month even
generously assuming an earnings-season spike. Two-phase (`submit_*`/
`collect_*`) because the Batch API is asynchronous — up to 24h — so this is
not one synchronous call like the rest of the data layer. Degrades to a
clear "not configured" report without `ANTHROPIC_API_KEY`, mirroring
`notify.py`'s Telegram pattern exactly. Nothing here is scored — G0
forbids sentiment as a directional score input, and `scoring.py` does not
import `sentiment.py`.

**Both are now wired into `refresh`** (Sep 2026): every run ingests new
announcements for the tracked universe, collects any earlier G4 batch that
has finished, then submits whatever is newly unextracted — capped, and
independently skippable via `--skip-announcements`/`--skip-sentiment`.
Live-verified against the real project database: 463 real announcements
landed across 49 of 50 Nifty 50 names on the first run, a second run
re-fetched the incremental overlap window without creating a single
duplicate row, and with no `ANTHROPIC_API_KEY` configured the sentiment
step reported itself unconfigured on every run while the rest of the
refresh (bars, delivery, announcements) completed normally — no live G4
extraction has actually run yet, since this environment has no Anthropic
credentials; that will be the first true end-to-end proof of the G4 half.
G2 (mainstream news RSS) and G3 (social, already [OPEN] on cost grounds)
are unbuilt.

**Open finding — the scored universe shows no edge yet.** The first live
backtest (121 sessions to 2026-09-23) returns a slightly *negative* rank
IC (-0.03 at 5d, -0.06 at 20d) and a negative top-minus-bottom quintile
spread. On the statistically honest non-overlapping-window test (6
independent 20-session months, since daily-sampled 20d windows reuse 19/20
of their data and inflate significance), IC was -0.089 (t=-1.49, not
significant) -- negative in 5 of 6 months, driven largely by A1-A4 (which
INDICATORS.md §0.1 already flagged as near-redundant "price rose recently"
restatements) moving as one block whenever a sector moved as one block: a
July 2026 holdout found every Nifty 50 IT name in the bottom five ranks
together, then IT rallied 9-27%.

**Fixed (Sep 2026): sector-neutral A1-A4.** NSE's own sector classification
(already fetched, previously discarded -- see storage schema v5) is now used
to demean A1-A4 against their sector's average before ranking (INDICATORS.md
A1-A4 section). Re-run on the same July holdout: quintile spread went from
-7.85% to +1.13%. Re-run across all six non-overlapping windows: IC went
from -0.089 to +0.005 -- essentially zero rather than negative, and still
not a statistically significant *positive* signal at n=6. This resolves the
observed structural failure (a whole sector being indistinguishable from a
weak individual stock); it does not by itself establish predictive validity,
which still needs more independent observations than six months provides.
Scores from before and after are labelled differently and never compared
(`SCORING_VERSION` 1→2, `StockScore.method`).

The run's own caveats -- survivorship bias, delivery history
starting 2026-08-11, retroactively adjusted prices -- mean this is not yet
conclusive either way, but it is the thing to resolve before the scores are
trusted or the weights are tuned further.

## 1. Confirmed Scope

| Decision | Value |
|---|---|
| Trading style | Swing trading (days–weeks holding period) |
| Markets | Indian equities (NSE/BSE) + Gold & Silver (MCX / spot) |
| Initial universe | Nifty 50 constituents + Gold/Silver |
| Usage | Personal use only (not shared publicly — no SEBI Research Analyst registration concerns) |
| Core anchor feature | Automated morning scanner (pre-market, runs before 9:15 AM IST) |

---

## 2. Vision

Give a swing trader a wider field of view than they can maintain manually —
scan a full universe every morning, score each name across technical,
volume/conviction, and market-regime signals, and surface the ranked
opportunities with a plain-language reason, before the market opens.

The tool **scores and explains** — it is not a black-box "buy this" signal.
Every score should be traceable to the factors that produced it.

---

## 3. MVP Feature Set

### 3.1 Morning Scanner (the anchor feature)
- Scheduled job, runs ~7:30–8:30 AM IST daily (before market open).
- Pulls previous session's EOD data for Nifty 50 + Gold/Silver.
- Computes the composite score per instrument and ranks the universe.
- Delivers a digest (Telegram bot message) with top N ranked
  opportunities + score breakdown.
- Logs every day's scan result to a local store (the seed of the trade
  journal — needed to later check whether scores predicted anything).

### 3.2 MVP Scoring Signals

> **Superseded — see [INDICATORS.md](INDICATORS.md)** for the reviewed,
> evidence-weighted indicator set. The table below was the first-pass
> proposal and is kept only for history. Key revisions: RSI, MACD and the
> 20 DMA were cut as redundant; 52-week high proximity, Donchian breakout,
> ATR and a market regime gate were added; FII/DII was re-classified from a
> per-stock signal to a market-level gate.

| Category | Signal | Why it's in v1 |
|---|---|---|
| Trend | Price vs. 20/50 DMA, MA stack direction | Cheap, reliable, core swing filter |
| Momentum | RSI, MACD | Standard, well understood |
| Relative strength | Stock return vs. Nifty 50 return (rolling window) | Surfaces leaders, not "everything went up" |
| Volume/conviction | Delivery % trend, volume vs. 20-day avg | India-specific genuine-interest signal |
| Market regime | FII/DII net flow (market-wide context) | Down-weights bullish signals on institutional sell-off days |
| Gold/Silver | MCX trend + USD/INR direction | Separates commodity strength from rupee weakness |

Deliberately **excluded from MVP** (see v2+ roadmap): fundamentals, news/sentiment
NLP, options flow, bulk/block deal data, promoter pledge tracking.
Exception under review: **PEAD (post-earnings drift)** may be worth pulling
forward — see INDICATORS.md §F1.

### 3.3 Trade Journal (minimal, MVP)
- Every scan's output stored with date + score breakdown.
- No UI needed yet — this is a data-integrity requirement, not a feature,
  so the scoring can be validated against real outcomes later.

### 3.4 Watchlist-Triggered Background Analysis

Adding a stock to the watchlist starts a background job for that stock.

**Trigger:** stock added to watchlist → job enqueued → runs async.

**Two job types, deliberately separated:**

| Job | When | Does |
|---|---|---|
| **Deep analysis** (one-shot) | On add | Full history pull, all Bucket A–C indicators, peer/sector comparison, ATR-derived stop & position size, score + written breakdown |
| **Monitor** (recurring) | While on watchlist | Re-checks score daily, watches price against levels the deep analysis produced, fires alert on threshold crossing or score change |

**Removal from watchlist cancels the recurring job** — otherwise dead jobs
accumulate and burn API quota.

**Why background rather than synchronous:** a deep analysis pulls years of
history plus delivery data for one stock; at broker-API rate limits this is
seconds-to-minutes, too slow to block a UI action on.

### 3.5 Real-Time Stock Lookup (any ticker)

On-demand information for **any** NSE/BSE stock, not just the Nifty 50
universe.

**Important distinction — real-time price ≠ real-time score:**
Most of the scoring model runs on daily bars, and two key inputs are
*published only after market close by NSE*: delivery % (A6) and FII/DII flow
(B4). A score recomputed on every tick would be partly stale by construction
and would imply a precision the model does not have.

So the layers are split:

| Layer | Cadence | Contents |
|---|---|---|
| **Real-time** | Live / on-demand | Price, change %, volume, day range, basic intraday context |
| **Score** | Daily (post-close / pre-open) | Full composite score + breakdown |
| **Alerting** | Live | Price crossing levels the *daily* score produced |

Real-time is for **monitoring and triggers**, not for re-scoring. This keeps
the swing-trading model honest and avoids inviting intraday overtrading,
which is outside the stated trading style.

**Universe/ranking problem this creates:** see [INDICATORS.md](INDICATORS.md)
§0.2 — cross-sectional percentile ranking requires a defined universe. An
arbitrary off-universe ticker has nothing to be ranked against. **[OPEN]**,
options documented in INDICATORS.md.

### 3.6 News & Sentiment Analyzer

Adds the emotional/narrative factor. Sources, method and the constraints on
how sentiment may be used are specified in [INDICATORS.md](INDICATORS.md)
Bucket G — summary: NSE/BSE corporate announcements first (free, structured,
un-astroturfable), mainstream financial news second, X/social last and
**[OPEN]** on cost grounds.

Sentiment is used for **event detection, risk flags, and contrarian
extremes** — deliberately *not* as a "positive mentions → higher score"
contributor. Rationale in INDICATORS.md §G0.

### 3.7 LLM Concluder

An LLM reviews everything extracted — quant score and its breakdown, news,
filings, sentiment, risk/eligibility flags — and produces a written
conclusion.

#### 3.7.1 The constraint that matters: synthesizer, not decider

**The LLM must never silently override or mutate the quantitative score.**

The reason is the whole premise of the project. The quant score is
reproducible and backtestable — that is what lets the trade journal
eventually answer "does this system work?". An LLM verdict is
non-deterministic and cannot be cleanly backtested. If the LLM can overrule
the score, the primary number becomes unvalidatable and the journal stops
being evidence.

Worse: an LLM can *always* construct a plausible narrative for either
direction. Uncritically trusted, it reintroduces exactly the narrative bias
this tool exists to counteract — with more eloquence than a human would
manage.

**Therefore:**

| Rule |
|---|
| The quant score remains the primary, testable number |
| The LLM output is a **separate, clearly-labelled qualitative layer** |
| On disagreement, **surface the conflict** — never average the two into one number |
| Both are logged to the journal, independently, so each can be scored later |

#### 3.7.2 What the LLM is actually good for here

- **Structured extraction** from unstructured news/filings (this *is* the
  §3.6 sentiment analyzer).
- **Explaining** the quant score in plain language.
- **Qualitative risk flags the quant model structurally cannot see** — QIP
  announced, auditor resigned, regulatory order, pledge spike.
- **Red-teaming** — "what would make this trade fail?" Arguably its highest-
  value role, and the one that most counteracts confirmation bias.

#### 3.7.3 Journal integrity requirements

Because conclusions get logged and compared over months:

- **Structured output** against a fixed schema (`output_config.format`), not
  free prose — so verdicts stay machine-comparable.
- **Versioned prompt**; log prompt version + model ID with every verdict.
  Without this, months of journal entries are uninterpretable because you
  cannot tell whether the model or the prompt changed underneath them.
- Verdict schema should include a **confidence** and an explicit
  **disagrees-with-quant-score** boolean.

#### 3.7.4 Model selection

| Job | Model | Why |
|---|---|---|
| Bulk news/sentiment extraction | Claude Haiku 4.5 or Sonnet 5 | High-volume classification/extraction; cheap per item |
| The concluder itself | **Claude Opus 5** (`claude-opus-5`) | Low volume (one per candidate), high stakes, needs genuine synthesis across conflicting evidence |

Use **adaptive thinking** (`thinking: {type: "adaptive"}`) on the concluder —
weighing conflicting quantitative and qualitative evidence is exactly the
case for it.

Run the nightly universe pass through the **Batch API (50% cost)** — the
morning scan is prepared overnight and is not latency-sensitive. Watchlist
and on-demand lookups (§3.4–3.5) use the normal synchronous API.

**[OPEN]** Should the concluder run on the full scanned universe every night,
or only on the top-N ranked candidates plus watchlist names? Top-N is
materially cheaper and probably sufficient — you do not need a written
thesis on the 40th-ranked stock.

---

## 4. Architecture (revised)

§3.4 and §3.5 move this past a single cron job. It now needs a persistent
service, a job queue, and a live data feed.

| Layer | Choice | Why |
|---|---|---|
| Language | Python | Best fit for data/finance libs (pandas, pandas-ta) |
| API service | FastAPI | Serves watchlist actions + on-demand lookups; async-native |
| Job queue | Celery or RQ + Redis | Background watchlist jobs, retries, cancellation |
| Scheduling | Celery beat (or cron for the nightly batch) | Recurring monitor jobs + nightly universe scan |
| Storage | SQLite → Postgres if it grows | Personal use; Postgres only if concurrency demands it |
| Live data | Broker WebSocket (Kite/Upstox/Angel One) | Only path to real-time NSE quotes |
| Cache | Redis | Quote caching, rate-limit budget, dedupe |
| Delivery | Telegram bot | Digest + alerts, same channel |
| Indicators | `pandas-ta` + custom | Custom needed for delivery %, efficiency ratio, gold/silver ratio |
| News/sentiment | RSS + NSE announcements feed | Free, structured, low manipulation risk (INDICATORS.md §G1–G2) |
| LLM | Anthropic `anthropic` SDK — Opus 5 concluder, Haiku 4.5/Sonnet 5 extraction | Structured outputs for journal comparability (§3.7) |

### 4.1 Three data tiers (rate-limit shaped)

Broker APIs cap both WebSocket subscriptions and REST throughput, so
"real-time for every stock" is really:

| Tier | Scope | Mechanism |
|---|---|---|
| 1. Streaming | Watchlist only (bounded, ~tens) | WebSocket subscription |
| 2. On-demand | Any ticker, when asked | REST quote, short-TTL cached |
| 3. Batch EOD | Full universe (Nifty 50 → 500) | Nightly bulk pull |

Tier 1 is bounded because streaming thousands of symbols is neither
necessary nor within quota. Tier 2 gives the "any and every stock" coverage
without holding open subscriptions for stocks nobody is watching.

**This forces the broker API decision** (scope §8.2) — `yfinance`/bhavcopy
is EOD-only and cannot serve tiers 1 or 2.

---

## 5. Data Sources (candidates)

| Need | Free/low-cost option | Paid option |
|---|---|---|
| NSE/BSE EOD price data | NSE bhavcopy, `yfinance` (`.NS`/`.BO`) | Kite Connect, Upstox, Angel One SmartAPI |
| Intraday/real-time (later) | Angel One SmartAPI, Upstox free tier | Kite Connect (~₹2000/mo), TrueData |
| Delivery %, bulk/block deals, FII/DII | NSE website (free CSV/scrape) | — |
| MCX Gold/Silver | MCX website EOD | Kite/Upstox for live |
| Global spot gold/silver, USD/INR | Alpha Vantage, exchange-rate APIs | — |
| Fundamentals (v2+) | Screener.in (scraping, no official API) | Tijori, Trendlyne |

**[OPEN]** Do you already hold a broker account with API access (Zerodha
Kite, Upstox, Angel One)? That changes which data source is the default
rather than a fallback.

### 5.1 Observed feed defects (verified live, Sept 2026)

Every one of these returns **plausible-looking wrong data with a success
status** — none raises an error. This is why each source has a date-and-shape
guard rather than a bare try/except.

| Source | Defect | Handling |
|---|---|---|
| yfinance | Emits **phantom bars on NSE holidays** — flat OHLC, zero volume (e.g. 2026-09-14 for every symbol) | Rejected via the NSE trading calendar |
| yfinance | **Silently omits sessions** for individual stocks — 11 of 50 Nifty names were missing 2026-09-17, which NSE's bhavcopy does have | Surfaced by gap detection; backfill **[OPEN]**, see below |
| yfinance | Returns **NaN prices with real volume** (GOLDBEES/SILVERBEES, 2026-09-18, ~28M volume) | Rejected as incomplete; reported as a gap |
| yfinance | Returns an **empty frame, not an error**, for delisted/misspelled symbols | Raised as `DataUnavailableError` |
| NSE bhavcopy | On a holiday, serves the **previous session's file with HTTP 200** | Rejected by comparing the file's own `DATE1` to the requested date |
| NSE bhavcopy | Writes unpublished delivery as `'-'` (275 of 3508 rows sampled) | Recorded as *unavailable*, never as zero |
| NSE (both) | Serves an **HTML block page with HTTP 200** when rate-limiting | Detected and refused before parsing |

**[OPEN] Gap backfill.** NSE's bhavcopy carries full OHLC and could fill
yfinance's missing sessions — but bhavcopy prices are **unadjusted** while
yfinance's are split/dividend-adjusted. Blending them naively would inject
artificial jumps at corporate-action dates. A safe backfill would have to
verify no split or dividend fell between the gap and the present. Deferred as
its own task; current gaps are 1–2 sessions per affected instrument.

---

## 6. Full Feature Landscape (beyond MVP)

Captured here so nothing discussed gets lost, even though most of this is
v2+.

- **Data ingestion**: fundamentals, news/event NLP, social sentiment,
  options flow, institutional/insider filings, promoter pledge tracking,
  macro indicators (rates, VIX, yield curve).
- **Technical engine**: pattern recognition (breakouts, support/resistance),
  multi-timeframe confirmation, sector rotation/breadth view.
- **Scoring engine**: explainability breakdown per score, confidence/
  conviction separate from the score, configurable weighting profiles,
  backtest-driven weight tuning.
- **Opportunity discovery**: universe-wide screener beyond Nifty 50 (e.g.
  Nifty 500), anomaly/outlier detection, sector/theme rotation view,
  correlation-aware suggestions (avoid recommending 5 versions of the same bet).
- **Risk management**: position sizing by volatility/account risk %,
  stop-loss/target with risk:reward shown alongside score, portfolio-level
  correlation and concentration view.
- **Feedback loop**: full trade journal UI, realized-outcome tracking,
  continuous recalibration of scoring weights against actual performance.
- **Alerting**: intraday threshold-crossing alerts, not just the morning digest.
- **UX**: dashboard with score breakdown (radar/waterfall chart), watchlist +
  screener + journal in one flow, paper-trading mode.

---

## 7. Compliance / Safety Notes

- Personal use only, confirmed — no SEBI Research Analyst registration
  requirement as currently scoped.
- If this ever expands to sharing scores/calls with anyone outside personal
  use, revisit this before that happens — regulatory posture changes.
- Tool should always frame output as a **score + explanation**, not a
  definitive buy/sell instruction, to keep this framing intact even
  informally.

---

## 8. Open Questions Before Coding

1. **[OPEN]** Confirm tech stack (Python assumed) — any existing tooling
   preference?
2. **[OPEN]** Broker API access available (Kite/Upstox/Angel One), or start
   from free sources (`yfinance` + NSE bhavcopy)?
3. **[OPEN]** Telegram bot acceptable as the delivery channel, or prefer
   email/other?
4. **[OPEN]** Any Nifty 50 exclusions/inclusions, or use the index as-is?
5. **[OPEN]** Score scale/format preference (0–100? A–F grade? star rating?)
   — cosmetic but worth deciding once.
6. **[OPEN]** Any signals in the MVP list above that should be cut further,
   or any from the "beyond MVP" list that should be pulled forward?
   → Addressed in [INDICATORS.md](INDICATORS.md); open indicator-level
   decisions are tracked there.
7. **[OPEN]** X/Twitter sentiment — include despite recurring API cost, or
   defer and run on free news + NSE announcements only? (INDICATORS.md §G3)
8. **[OPEN]** Concluder scope — full universe nightly, or top-N + watchlist
   only? (§3.7.4)
9. **[OPEN]** Monthly running-cost ceiling for the whole system (broker API +
   X API + LLM spend). Worth fixing a number now, since three separate
   components can each quietly grow into it.

---

## 9. Roadmap After MVP Validation

Once the morning scanner has run for a few weeks and the journal shows
whether the MVP scoring signals actually correlate with outcomes:

1. Expand universe (Nifty 500).
2. Add fundamentals + news/sentiment signals.
3. Add risk management layer (position sizing, stop/target suggestions).
4. Add a proper dashboard/UI beyond the Telegram digest.
5. Add backtest-driven weight tuning instead of hand-set weights.
