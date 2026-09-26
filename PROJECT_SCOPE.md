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
| Indicators — A1–A8 score contributors (A8/PEAD added Sep 2026), C1–C2 risk inputs | ✅ |
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

**A8 (PEAD) added Sep 2026.** INDICATORS.md Bucket F deferred this pending
an obtainable earnings-surprise feed. One was found and confirmed:
`yfinance` (already a dependency) covers 49 of 50 Nifty 50 names, cross-
verified against a real NSE announcement date. Tested for signal on real
price history before building (390 real events, IC +0.10 to +0.12 across
5/10/20-day horizons) — see INDICATORS.md A8 for the full evidence,
including the one unresolved risk (estimate-field provenance is
undocumented) and an honest composite-level check after building: the
event-level signal does not move the full composite's own non-overlapping
IC (-0.008 without A8, -0.009 with), because A8 is active for only a
minority of the universe on any given day. `SCORING_VERSION` bumped 2→3.

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

**Revised (Sep 2026): bulk extraction is provider-configurable, not
Claude-only.** Built with Claude Sonnet 5 first (this table's original
choice); the user then asked for provider flexibility to avoid lock-in.
`ALGORIX_LLM_PROVIDER=anthropic|openai` picks the active one (default
`anthropic`), one at a time -- not a fallback chain. gpt-4o-mini was
verified as the OpenAI default: cheaper per token than either Anthropic
option at batch rates, with equivalent structured-output support (checked
against OpenAI's current docs and SDK, not assumed). See CLAUDE.md and
`sentiment.py`. The concluder (unbuilt) is not addressed by this —
whether it should also become provider-configurable, or stay Opus-5-only
given the higher stakes of that specific job, is an open question of its
own if/when §3.7 gets built.

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

### 6.1 Chart UI / "advance tooling" phase (started Sep 2026)

User asked to move into a "next phase: advance tooling and user
flexibility," starting with the stock detail page's chart. Two parts:

**Part A — indicator audit (DONE, no code involved).** User pasted a
list of indicators popular among Indian retail/intraday traders (VWAP,
Supertrend, RSI, MACD, Bollinger Bands, EMA crossover) and asked which
this project has. Answer recorded in INDICATORS.md Bucket E: RSI/MACD/
Bollinger were already formally rejected with reasons; VWAP/Supertrend/
EMA are new evaluations, all three recommended against (VWAP is an
intraday-execution concept that conflicts with invariant 3's daily
cadence; Supertrend's own cited evidence shows default parameters lose
money, exactly the overfitting risk this project's equal-weighting
principle exists to avoid; EMA crossover is structurally what MACD
already reduces to). Read INDICATORS.md Bucket E for the full reasoning
before reconsidering any of the three.

**Part B — interactive chart (DONE).** The stock detail page's chart was
a static, hand-built SVG polyline of closing prices only — no OHLC, no
volume, no pan/zoom. Replaced with a real candlestick + volume chart,
pannable/scrollable/zoomable.

- **Library chosen:** TradingView's `lightweight-charts` v5.2.0
  (MIT-licensed, ~45KB, loaded via a pinned CDN `<script>` tag --
  `https://unpkg.com/lightweight-charts@5.2.0/dist/lightweight-charts.standalone.production.js`).
  No npm, no build step. Every API call used (`createChart`,
  `chart.addSeries(CandlestickSeries, ...)`,
  `chart.addSeries(HistogramSeries, ..., 1)` for a separate volume pane,
  `chart.panes()[1].setStretchFactor(...)`, `autoSize: true`) was checked
  against the library's own current docs and its official SKILL.md before
  writing it -- not recalled from training data, which is untrustworthy
  for a library with no bundled skill in this environment. One thing
  deliberately left unset: `crosshair.mode` -- two sources disagreed on
  whether it takes a string or a numeric enum, so it was left at the
  library's own default rather than guessed at.
- **Design deviation, flagged not silently made:** `web/server.py`'s own
  module docstring says "Deliberately boring technology... No build step,
  no npm, no client framework." This introduces client-side JS for the
  first time in this app. The "no build step" half is preserved (a CDN
  script tag, no bundler); the "no client framework" half is now
  narrowly not true, scoped to this one chart. Worth a conscious decision
  on whether that's acceptable going forward, not just accepted by
  default because it already happened.
- **What exists:** a read-only JSON route `GET /stock/{symbol}/bars` in
  `web/server.py` returning OHLCV for the loaded window (reuses data
  already loaded for the page, no new ingestion), and `stock.html`'s
  sparkline block replaced with a `<div id="price-chart">` + inline
  `<script>` that fetches that route and renders candles + a volume pane,
  themed from the page's existing light/dark CSS custom properties
  (`--good`/`--bad`/`--accent`/`--line`/`--muted`) so it matches both
  themes without extra work.
- **Closed out (2026-09-25):** the old sparkline's dead code
  (`_sparkline()`, the `closes` list, the unused context keys) is
  removed. `tests/test_web.py` has positive + negative coverage for
  `/stock/{symbol}/bars` (known instrument, unknown symbol, instrument
  with no bars stored) and `test_stock_page_draws_a_sparkline` was
  replaced with `test_stock_page_embeds_the_chart`. Full offline suite
  passes (649 tests). The app was actually run against the real DB
  (`~/.algorix/algorix.db`, EICHERMOT) and driven with Playwright, not
  just curl: candles + volume render in both light and dark theme, the
  CDN script loads with no console errors, and wheel-zoom + drag-to-pan
  both work (crosshair and date label confirmed interactively).
**Part C — chart annotations: trendlines + breakout/dip highlights
(DONE, 2026-09-26).** The open question from Part B ("should a drawn
line persist?") was answered: **yes, persisted to the database.**
Mid-session the user also asked for a second, related feature: the
system should auto-highlight breakouts/dips on the chart, and the user
should be able to place the same kind of marker manually.

- **Schema:** new `chart_drawings` table (migration `_SCHEMA_V9`),
  generic `tool_type` + JSON `points` columns -- adding a future drawing
  type (horizontal line, rectangle, fib retracement) is a new
  `tool_type` string and a client-side renderer, never a migration. Only
  user-placed annotations are rows here (`ChartDrawing` in `models.py`,
  `ChartDrawingRepository` in `storage.py` -- the first `DELETE FROM` in
  that file, instrument-scoped so a delete can't cross symbols).
- **Auto-detected breakout/dip markers are NOT persisted.** They're
  recomputed every page load in `stock_detail()` from indicators the
  page already shows (A4 Donchian position >= 95%, or A5 short-term
  return <= -3% while A3's trend state is an uptrend) -- see
  `_auto_markers()` in `web/server.py`. UI-only heuristic thresholds,
  never read by `scoring.py`. Only the latest session is evaluated
  (these indicators aren't computed per-historical-bar on this page); a
  full historical breakout/dip series would need rolling recomputation
  across the whole window, explicitly out of scope here.
- **The user can also manually place a breakout/dip marker** (1 click)
  or a trendline (2 clicks) -- both go through the same
  `POST /stock/{symbol}/drawings` route and persist. `_DRAWING_POINT_COUNTS`
  in `web/server.py` is the tool-type allow-list (`trendline`: 2,
  `breakout`: 1, `dip`: 1) -- extending it is one dict entry.
- **First POST/DELETE routes and first Pydantic request body in this
  codebase** (`DrawingPoint`/`DrawingCreate`). Convention: an unknown
  symbol is informational-empty (200 `[]`) for the GET, but a real error
  (404) for POST/DELETE, since a write against nothing has nothing to
  attach to.
- **Client-side:** `web/static/chart-drawings.js` (first static mount in
  the app) ports TradingView's own official `TrendLine` primitive
  example for lightweight-charts v5 (re-verified against the upstream
  source while writing it, not recalled from memory) plus a small
  `AlgorixDrawingTools` registry (`trendline`/`breakout`/`dip` today).
  Manual breakout/dip markers use v5's `createSeriesMarkers()`. Trendline
  selection/delete is a plain DOM list with a delete button, deliberately
  not canvas click-hit-testing -- simpler to build and far more reliable
  to test than pixel-precise clicking on a thin line.
- **Tests:** model validation (`test_models.py`), full repository suite
  including the cross-instrument-delete negative case
  (`test_storage.py`), full route suite including unknown-symbol/
  unknown-id/wrong-instrument-delete/malformed-body cases
  (`test_web.py`), and direct unit tests for `_auto_markers()` (breakout,
  dip-within-uptrend, pullback-outside-uptrend is correctly ignored,
  unavailable indicators, no-trend-state). Full offline suite passes
  (683 tests).
- **Verified interactively, not just asserted:** ran the app against the
  real DB with Playwright -- drew a trendline and a breakout marker,
  reloaded the page (the actual persistence proof, both still rendered),
  deleted both, reloaded again (confirmed gone), and confirmed toggling
  a tool off before placing any point fires no request. Auto-marker
  rendering (arrow, no delete button since it isn't a stored row) was
  verified against a throwaway DB copy seeded with a synthetic breakout
  bar, in both light and dark theme -- the real Nifty 50 data at the
  time had no symbol past either threshold. The real database was left
  with its schema migrated (permanent, harmless) but zero
  `chart_drawings` rows -- every row created during testing was deleted
  before the session ended.

### 6.2 Watchlist, manual scan trigger, AI Q&A (DONE, 2026-09-26)

User asked for three UI additions together. All three shipped, tested
(752 offline tests), and verified against the real app with real data
(including one real LLM call and one real full refresh/scan run).

- **Watchlist.** New `watchlist` table (boolean membership,
  `instrument_id` as its own PK -- no synthetic id needed),
  `WatchlistRepository`, toggle button on the stock page, and a
  `/watchlist` page. The watchlist is filtered out of the *same* scored
  universe `dashboard()` computes (`_score_current_session`, shared by
  both) -- a watchlisted symbol is never scored in isolation, which would
  produce percentiles incomparable to the dashboard's (invariant 8). A
  watchlisted symbol that's dropped from the tracked index, ineligible, or
  genuinely unscored shows that reason explicitly, never a fabricated
  score.
- **Manual scan trigger.** `POST /scan/run` runs the identical
  whole-universe `refresh()` → `run_scan()` pipeline cron already runs at
  07:30, in a `BackgroundTasks` job (first use of that in this app), with
  a `GET /scan/status` polling endpoint and a button on the dashboard.
  Two gaps this exposed for the first time and closed: (1) nothing
  previously stopped a cron-triggered and a manually-triggered refresh
  from overlapping -- fixed with `refresh.refresh_lock()`, an advisory
  `fcntl.flock` shared by `refresh.main()`'s cron path and the new
  web-triggered path, so there is exactly one lock implementation; (2)
  unthrottled manual triggering would multiply G4 sentiment-batch cost
  and NSE/yfinance rate-limit exposure past the daily cron cadence --
  fixed with a 45-minute cooldown, keyed off the lock file's own
  "started at" timestamp (not "finished successfully," so a failed run
  can't be retried immediately). `refresh()`/`run_scan()` both *collect*
  per-instrument failures into `.errors` rather than raising, so
  `_run_pipeline` checks `.errors`, not just exceptions, or a partially-
  failed run would report "success." **Verified with a real full run**
  (clicked the button for real, ~2 minutes, completed successfully,
  delivered to the real configured Telegram chat) -- not a mocked
  approximation.
- **AI Q&A.** New `src/algorix/qa.py`, sitting beside `sentiment.py` the
  way `sentiment.py` sits beside `scoring.py`. One-shot (no conversation
  memory), **synchronous** -- the first non-batch LLM call in this
  codebase (`sentiment.py`'s `Extractor` protocol is batch-shaped only).
  Deliberately separate env vars from G4 (`ALGORIX_QA_PROVIDER`/
  `ALGORIX_QA_MODEL`, not `ALGORIX_LLM_PROVIDER`/`_MODEL`) so a
  G4 cost-driven model change can't silently change Q&A quality too.
  `gather_stock_context()` assembles the same indicators/trend/score the
  stock page already shows (`indicators.stock_indicator_rows`, extracted
  out of `web/server.py` so page and prompt can never silently diverge)
  plus delivery, PEAD, recent G4-extracted events, and recent OHLCV --
  every missing piece renders as explicit "unavailable" text, never
  omitted. The system prompt states the quant score as this tool's single
  authoritative number and requires any disagreement to be labelled, not
  substituted (invariant 1); nothing here is read by `scoring.py`
  (invariant 2). Every answer is logged to a new `qa_queries` table with
  the answerer's own resolved `model_id` and `PROMPT_VERSION` (invariant
  7) -- full context stored, not a hash, since this is a low-volume
  personal tool and full context is what actually lets a bad answer be
  debugged later. Cost is flagged explicitly in the module docstring:
  unlike G4's batch discount, this is full-price per question. **Verified
  with real LLM calls** (via the OpenAI provider, the credential actually
  configured) -- the answers correctly cited the exact indicator values
  shown on the same page (e.g. "relative volume is 0.47 times the
  average," "ATR is 1.45%," matching the Indicators table exactly), and
  the audit row landed in the real database with the correct model id.
  The default (Anthropic, unconfigured in this environment) was also
  verified to degrade cleanly to a visible "not configured" message, in
  the browser, not a crash.

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
- **UX**: dashboard with score breakdown (radar/waterfall chart), screener +
  journal in one flow (watchlist itself shipped, see §6.2), paper-trading mode.

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
