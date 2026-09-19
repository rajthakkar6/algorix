# Algorix — Indicator Selection & Rationale

A reviewed, evidence-weighted plan for which indicators enter the scoring
model and which are deliberately rejected. Companion to [PROJECT_SCOPE.md](PROJECT_SCOPE.md);
this document supersedes the provisional signal table in scope §3.2.

The bar for inclusion: **an indicator must add information no already-included
indicator provides.** Anything that merely restates existing information is
rejected, however popular it is.

---

## 0. Design Principles (decided before picking anything)

### 0.1 Redundancy is the main failure mode
Most classic indicators are mathematical transforms of the same underlying
quantity — price change over a lookback window. RSI, Stochastics, CCI,
Williams %R, MACD and MFI are all momentum restatements. Stacking them
produces **false confirmation**: five "agreeing" indicators is one signal
counted five times, and it makes a model feel robust while it is actually
concentrated. Research consensus is explicit that combining redundant
oscillators (e.g. RSI + Stochastics) adds nothing, and that 2–3
*complementary* indicators beat a large correlated stack.

**Rule adopted:** before any weight is assigned, compute the correlation
matrix across candidate signals on historical data. Any pair above ~0.8
gets one of the two dropped.

### 0.2 Cross-sectional ranking, not absolute thresholds
For a fixed universe (Nifty 50), rank each stock's signal as a **percentile
within the universe** rather than against an absolute cutoff ("RSI > 70").
Absolute thresholds are regime-dependent and break when overall volatility
shifts; ranks are self-normalising. This also makes signals directly
combinable — every signal becomes a 0–100 percentile on the same scale.

### 0.3 Three distinct roles — do not mix them into one number
A serious failure in retail scoring tools is treating every indicator as a
directional score contributor. Signals fall into three separate roles:

| Role | What it does | Where it belongs |
|---|---|---|
| **Score contributor** | Ranks instruments against each other | The composite score |
| **Regime gate** | Decides whether to act on scores *at all* today | Market-level filter, applied after scoring |
| **Risk/sizing input** | Determines stop distance and position size | Never in the score |

### 0.4 Why regime gating is non-negotiable here
An 18-year NSE backtest of the Nifty 200 Momentum 30 index shows momentum
beating the Nifty 50 (14.01% vs 10.42% CAGR) — but through a **-70.5%
drawdown with a 65-month recovery**. And through much of 2025–26, Indian
momentum was flat-to-negative (UTI Nifty200 Momentum 30 trailing 1-year ≈
-1%) *while global momentum was the single best-performing factor* — the
factor is regime- and geography-dependent, not a constant.

A scanner that ranks momentum without asking "is this a market where
momentum works?" will confidently hand you its best picks straight into a
drawdown. **The regime gate is what separates this tool from a screener.**

---

## Bucket A — Core Score Contributors (INCLUDE)

These carry actual predictive claims and are mutually complementary.

### A1. Cross-sectional relative momentum ★ highest weight
- **What:** stock return vs. universe, over multiple windows — 1M, 3M, and
  6M or 12M-skip-1M (skip the most recent month to avoid short-term reversal
  contamination, standard Jegadeesh–Titman convention).
- **Why:** the most replicated anomaly in the academic literature, confirmed
  in Indian equities specifically over 18 years of NSE data. Nothing else on
  this list has comparable evidentiary support.
- **Note:** the MD's current "relative strength vs Nifty" is the right idea
  but under-specified — multi-window with a skip-month is materially better
  than a single naive window.

### A2. 52-week high proximity
- **What:** current price as % of 52-week high.
- **Why:** a documented standalone anomaly (George & Hwang), *not* just a
  momentum restatement — it captures an anchoring effect and often works
  when return-based momentum is noisy. Trivial to compute, zero extra data.

### A3. Trend state (price vs. 50 DMA and 200 DMA)
- **What:** boolean/graded position of price relative to the 50 and 200 DMA,
  plus MA slope direction.
- **Why:** time-series momentum (trend following) is a separate, independently
  documented effect from cross-sectional momentum — a stock can rank high vs.
  peers while in its own downtrend. Serves double duty as a long-only veto.
- **Change from current MD:** 20 DMA adds little over 50 DMA for a days–weeks
  hold and is noisier; **200 DMA replaces it** as the structural trend anchor.

### A4. Donchian breakout state (N-day high)
- **What:** is price at/near an N-day high (e.g. 20/55-day)?
- **Why:** the cleanest, most testable formulation of "breakout" — objective,
  backtestable trend-following pedigree. Strictly preferable to subjective
  chart-pattern recognition, which has weak out-of-sample support.

### A5. Short-term pullback within an uptrend
- **What:** short-horizon (3–10 day) return, scored *inversely*, applied only
  to names already passing A1/A3.
- **Why:** short-term reversal is a real documented effect that runs opposite
  to long-horizon momentum. Combining "strong over months, weak over days"
  is meaningfully smarter than naive momentum and directly serves a
  swing-trade entry timing decision. **This is where RSI's only legitimate
  use case lives** — see Bucket E.

### A6. Delivery percentage trend ★ India-specific edge
- **What:** delivery volume as % of traded volume, trend vs. its own baseline.
- **Why:** separates genuine accumulation from intraday speculative churn.
  Not available in most markets — this is the most genuinely differentiated
  signal available to an Indian-market tool. Keep exactly as scoped.

### A7. Relative volume confirmation
- **What:** volume vs. 20-day average, used as a confirmation multiplier on
  A4 rather than a standalone score.
- **Why:** volume has weak standalone directional value but meaningful value
  confirming a breakout. Scoped as a modifier, not a contributor.

---

## Bucket B — Regime Gates (INCLUDE, market-level, not per-stock)

Applied *after* scoring, to decide whether and how aggressively to act.

### B1. Market breadth — % of Nifty 50 above its 50 DMA
- Cleanest available measure of whether the trend is broad or narrow.
  Narrow breadth is the classic precondition for momentum unwinds.

### B2. India VIX level and percentile
- Freely available, India-specific risk-regime gate. High-VIX regimes are
  where momentum strategies historically break.

### B3. Trend efficiency ratio (Kaufman)
- **What:** net directional move ÷ sum of absolute moves over the window.
- **Why:** distinguishes trending from choppy markets — the single condition
  that determines whether momentum signals work or whipsaw. **Chosen over
  ADX**, which measures the same property with more lag and less clarity.

### B4. FII/DII net flow
- Keep as scoped. Genuinely informative in Indian markets and freely
  published daily by NSE. Best used as a regime input, not per-stock.
- **Correction vs current MD:** scope §3.2 lists this in the same table as
  per-stock signals; it is market-level and belongs in this bucket.

---

## Bucket C — Risk & Sizing Inputs (INCLUDE, never in the score)

### C1. ATR (Average True Range) — **missing from current MD, should be added**
Arguably more useful than any oscillator, for three jobs: stop-loss distance,
volatility-normalised position sizing, and normalising breakout thresholds
across instruments of different volatility. It is not directional and must
never be scored as such.

### C2. Realized volatility percentile
Contextualises whether current ATR is historically high or low — a
volatility-contraction regime often precedes expansion. This captures the
only genuinely useful part of Bollinger Bands, more cleanly.

### C3. Correlation to index / sector concentration
Prevents the scanner surfacing five names that are really one bet. Matters
more as the universe expands beyond Nifty 50.

---

## Bucket D — Gold & Silver Specific

### D1. Trend following (A3/A4 applied to MCX series)
Trend following has *stronger* documented performance in commodities than in
equities. Same machinery, reused.

### D2. USD/INR decomposition ★ essential
MCX gold/silver prices move on both global commodity prices and the rupee.
Without decomposing, "gold looks strong" may purely be rupee weakness.
Correctly identified in the current MD — keep, and treat as mandatory, not
optional.

### D3. Gold/Silver ratio
Classic relative-value measure between the two instruments. Near-zero
computation cost and directly answers "if buying a precious metal, which
one?" — a question the tool otherwise cannot address.

### D4. Real interest rate proxy (v2, optional)
Gold is a zero-yield asset, so real rates are its dominant fundamental
driver. Useful but requires an external data source; defer unless cheap.

**Rejected for gold/silver:** Indian festival/wedding seasonality — real but
too noisy and low-frequency to score on reliably.

---

## Bucket E — REJECTED (with reasons)

| Indicator | Verdict | Reason |
|---|---|---|
| **MACD** | **Cut from MVP** | Literally two moving averages subtracted — almost fully redundant with A3 trend state, while adding lag. Currently in scope §3.2; recommend removal. |
| **RSI (classic 14, overbought/oversold)** | **Cut as scoped; narrow reuse only** | Weak standalone predictive evidence, and *actively conflicts* with momentum — the strongest winners stay "overbought" for months, so an RSI-70 rule sells exactly the stocks A1 is trying to buy. Only defensible use is short-horizon pullback timing, which A5 already expresses more directly. Currently in scope §3.2; recommend removal. |
| Stochastics, CCI, Williams %R | Reject | Momentum restatements; correlated with A1/A5. Pure redundancy. |
| Bollinger Bands | Reject | Volatility envelope; the useful part (squeeze/bandwidth) is captured more cleanly by C1/C2. |
| ADX | Reject | Superseded by B3 efficiency ratio — same job, less lag. |
| OBV, Money Flow Index, Chaikin | Reject | Crude volume-momentum hybrids. A6 delivery % is strictly better information in Indian markets. |
| Ichimoku Cloud | Reject | Composite of moving averages; redundant with A3 and harder to explain — conflicts with the explainability goal. |
| Fibonacci retracements | Reject | No credible empirical support; levels are arbitrary. |
| Candlestick patterns | Reject | Weak to no out-of-sample evidence; high false-positive rate. |
| Elliott Wave | Reject | Unfalsifiable and subjective — cannot be scored reproducibly. |
| 20 DMA | Demoted | Redundant beside 50 DMA at a days–weeks horizon; 200 DMA adds more (see A3). |

---

## Bucket F — High Value, Data-Gated (DECIDE)

Worth pulling forward from v2 **if** data access is workable:

### F1. Post-Earnings Announcement Drift (PEAD) ★ strongest candidate
One of the most robust anomalies in the literature, and its horizon (drift
over weeks following an earnings surprise) maps *exactly* onto a swing-trade
holding period. The current scope defers all fundamentals to v2, but PEAD is
not generic fundamental analysis — it is a timing signal that happens to need
earnings data. **Recommend reconsidering for MVP** if an earnings-surprise
feed is obtainable; the constraint is data sourcing in India, not merit.

### F2. Bulk & block deals
Free from NSE, genuine institutional-footprint signal. Moderate parsing work.

### F3. Promoter pledge changes
Strong *negative*/risk-flag signal. Low relevance for Nifty 50 (large, clean
promoters), high relevance when the universe expands to Nifty 500 — defer
until then.

---

## Summary of Changes vs. Current PROJECT_SCOPE.md §3.2

**Remove:** MACD, RSI (as scoped), 20 DMA.
**Add:** 52-week high proximity, Donchian breakout, short-term pullback,
200 DMA, ATR, realized-vol percentile, market breadth, India VIX,
efficiency ratio, gold/silver ratio.
**Re-specify:** relative momentum → multi-window with skip-month.
**Re-classify:** FII/DII from per-stock signal → market-level regime gate.

Net effect: fewer *directional* indicators than originally scoped, but
better separated by role, with redundancy removed and a regime gate added.

---

## Open Decisions

1. **[OPEN]** Accept dropping RSI and MACD? They are the two most familiar
   indicators here, so this is the change most worth pushing back on if you
   disagree — but the case against both is redundancy, not obscurity.
2. **[OPEN]** Pull PEAD (F1) into MVP, or hold for v2? Depends on whether an
   earnings-surprise data source is reachable for free.
3. **[OPEN]** Weighting approach: equal-weight z-scores across uncorrelated
   signal families (robust default, hard to overfit) vs. hand-tuned weights
   vs. backtest-fitted weights. **Recommendation: equal-weight to start** —
   out-of-sample, it usually beats hand-tuning, and it avoids baking in
   assumptions before there is journal data to validate against.
4. **[OPEN]** Should the regime gate *suppress* the morning digest entirely
   in hostile regimes, or still send it with a prominent risk warning?
   (Suggest the latter — silence is ambiguous, a warning is information.)

---

## Sources

- [Nifty 200 Momentum 30 — 18-Year NSE Backtest (BacktestIndia)](https://backtestindia.com/blog/momentum-investing-india-backtest)
- [Momentum vs Value: Which Factor Led Markets in 2026? (Value Research)](https://www.valueresearchonline.com/learn/equity-funds/momentum-vs-value-factor-investing-2026/)
- [Momentum Is Still 2025's Top Performer For Equity Risk Factors (Capital Spectator)](https://capitalspectator.substack.com/p/momentum-is-still-2025s-top-performer)
- [Technical Analysis and Discrete False Discovery Rate: Evidence from MSCI Indices (arXiv)](https://arxiv.org/pdf/1811.06766)
- [Technical Analysis: Modern Perspectives (CMT Association)](https://cmtassociation.org/wp-content/uploads/2019/01/Technical-Analysis-Modern-Perspectives.pdf)
- [Introduction to Technical Indicators and Oscillators (StockCharts ChartSchool)](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/introduction-to-technical-indicators-and-oscillators)
- [An Empirical Analysis of the Profitability of Technical Trading Rules (Lund University)](https://lup.lub.lu.se/luur/download?func=downloadFile&recordOId=8905915&fileOId=8905916)
