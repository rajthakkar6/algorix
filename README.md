# Algorix

A personal swing-trading (days–weeks) analysis and scoring tool for Indian
equities (NSE/BSE) plus gold/silver. Every morning it scores the Nifty 50
across momentum, trend, breakout, delivery accumulation, and post-earnings
drift, ranks them cross-sectionally, and explains *why* — it scores and
explains, it does not hand you a black-box "buy this."

For the design behind the indicator set and the non-negotiable rules this
project is built to, see [CLAUDE.md](CLAUDE.md), [PROJECT_SCOPE.md](PROJECT_SCOPE.md),
and [INDICATORS.md](INDICATORS.md).

This is a **personal-use tool**, not a public product.

---

## Is there a UI?

Yes — a local web dashboard (FastAPI, server-rendered, no JS build step),
served on your own machine only:

| Page | What it shows |
|---|---|
| `/` | Today's ranked scores, the regime banner, and why each stock scored what it did |
| `/stock/{symbol}` | One instrument: every indicator, price chart, journal history |
| `/backtest` | Does the score actually predict returns? IC, quintile spread, per-signal breakdown |
| `/journal` | Every past scan, recorded, so scores can be checked against what actually happened |

Run it with `python -m algorix.web` (see below) and open
`http://127.0.0.1:8000`.

There is also a Telegram digest (opt-in, see [Credentials](#credentials-optional)
below) for the same ranked list delivered to your phone each morning, and
an AI layer (also opt-in) that reads corporate announcements and extracts
structured risk flags — neither replaces the dashboard, both add to it.

---

## Setup

Requires Python 3.11+.

```bash
git clone <this repo>
cd algorix
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

That installs everything: the data pipeline, the scoring engine, the web
UI (FastAPI/uvicorn/Jinja2), and both LLM provider SDKs (Anthropic and
OpenAI) for the optional AI extraction layer.

### Credentials (optional)

None of these are required to run the core scanner — every one of them
degrades to "reports itself as unconfigured" if absent, never a crash.
Put them in a `.env` file at the repo root; it's picked up automatically
and gitignored, so credentials never need to be exported by hand or
committed.

```bash
# Telegram digest delivery
ALGORIX_TELEGRAM_TOKEN=...
ALGORIX_TELEGRAM_CHAT_ID=...

# AI extraction over corporate announcements (G4) -- pick ONE provider
ALGORIX_LLM_PROVIDER=anthropic      # default; needs ANTHROPIC_API_KEY
ANTHROPIC_API_KEY=...
# -- or --
ALGORIX_LLM_PROVIDER=openai         # needs OPENAI_API_KEY instead
OPENAI_API_KEY=...

# ALGORIX_LLM_MODEL=...              # optional, overrides the provider's default model
```

### Data location (optional)

Data lives at `~/.algorix/algorix.db` (SQLite) by default. Override with
`ALGORIX_DATA_DIR` or `ALGORIX_DB_PATH` if you want it elsewhere.

---

## Running it

**First run** — bring the database up to date, then look at what it found:

```bash
.venv/bin/python -m algorix.refresh          # pulls prices, delivery %, announcements, earnings
.venv/bin/python -m algorix.scan --no-send   # scores the universe, prints the digest
.venv/bin/python -m algorix.web              # browse the dashboard at :8000
```

The first `refresh` pulls ~2 years of history per instrument and will take
a few minutes; every run after that is incremental and fast.

### Daily use

```bash
.venv/bin/python -m algorix.refresh
.venv/bin/python -m algorix.scan             # sends the Telegram digest, if configured
```

Or install both as a weekday morning cron job:

```bash
.venv/bin/python -m algorix.schedule --install    # 07:30 local time by default
.venv/bin/python -m algorix.schedule              # preview the crontab entry without installing
.venv/bin/python -m algorix.schedule --remove     # uninstall it
```

On macOS, cron needs Full Disk Access for your terminal (System Settings →
Privacy & Security → Full Disk Access).

### Commands reference

```
refresh    Bring the data layer up to date (prices, delivery %, universe
           membership, corporate announcements, earnings surprises, and
           the AI extraction submit/collect cycle).
  --skip-universe        don't re-sync index membership
  --skip-delivery        don't fetch delivery %
  --skip-announcements   don't fetch corporate announcements
  --skip-sentiment       don't touch AI extraction (also a no-op with no API key)
  --skip-earnings        don't fetch earnings-surprise history
  --index IDX            NIFTY50 (default) / NIFTY200 / NIFTY500 / NIFTYMIDCAP150
  --history-days N       initial history depth for a newly-tracked instrument
  --db PATH              database path (overrides config)

scan       Score the universe for the latest session and deliver the digest.
  --no-send      print the digest, don't send to Telegram
  --print        also print the digest even when sending
  --top N        entries in the digest (default 8)
  --index IDX    which index to scan
  --db PATH

backtest   Replay past scores against what actually happened.
  --start YYYY-MM-DD / --end YYYY-MM-DD   date range (default: last 120 sessions)
  --sessions N    sessions back from --end when --start is omitted
  --signals       also print a per-signal (A1-A8) IC breakdown
  --signal-horizon N
  --index IDX
  --db PATH

web        Serve the local dashboard.
  --host HOST     default 127.0.0.1
  --port PORT     default 8000
  --index IDX
  --db PATH

schedule   Manage the daily cron job.
  --install / --remove
  --hour H --minute M   default 07:30 local time, before NSE opens at 09:15 IST
  --log PATH             append cron output to a file
  --db PATH
```

---

## Testing

```bash
.venv/bin/python -m pytest -m "not network"    # offline suite -- run this
.venv/bin/python -m pytest                     # includes tests that hit live feeds
```

---

## What's built, what isn't

Built: the morning scanner (all indicators A1–A8, the four regime gates,
composite scoring, ATR-based risk sizing), the trade journal, Telegram
delivery, the web dashboard, backtesting, corporate-announcement ingestion,
and AI-based event/risk extraction over those announcements.

Not built yet: per-stock watchlists with background monitoring, real-time
lookup for arbitrary tickers, and the LLM "concluder" that writes a plain-
language verdict alongside the score. See [PROJECT_SCOPE.md](PROJECT_SCOPE.md)
§0 for the current, honest status of each.

**The score's predictive validity is still an open question, not a
settled one** — see PROJECT_SCOPE §0 and INDICATORS.md for the actual
backtest numbers and what they do and don't show. This tool surfaces
signals and explains them; it does not claim to have proven they work yet.
