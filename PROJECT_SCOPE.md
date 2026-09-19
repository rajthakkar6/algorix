# Algorix — Project Scope

An AI-assisted trading analysis tool for personal use. It scores tradeable
opportunities across a defined universe so that signals a trader could miss
in the noise of daily market information get surfaced automatically.

This document is the working scope for review before any code is written.
Sections marked **[OPEN]** are decisions still needed.

---

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

---

## 9. Roadmap After MVP Validation

Once the morning scanner has run for a few weeks and the journal shows
whether the MVP scoring signals actually correlate with outcomes:

1. Expand universe (Nifty 500).
2. Add fundamentals + news/sentiment signals.
3. Add risk management layer (position sizing, stop/target suggestions).
4. Add a proper dashboard/UI beyond the Telegram digest.
5. Add backtest-driven weight tuning instead of hand-set weights.
