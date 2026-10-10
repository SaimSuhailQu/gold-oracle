# Gold Oracle — macro-factor probability model for gold (XAUUSD) on Exness

Predicts **P(gold higher in 20 trading days)** from 12 macro drivers, and an
MT5 Expert Advisor that trades the signal on an Exness account.

## How it works

```
gold_oracle/data.py     free market data (yfinance, no API key)
gold_oracle/factors.py  12 drivers -> bullish-positive z-scored features
gold_oracle/model.py    composite score -> calibrated probability (logistic)
gold_oracle/backtest.py walk-forward evaluation (no lookahead)
gold_oracle/signal.py   live signal -> signals/gold_signal.json
mql5/GoldOracle_Exness.mq5  EA: reads the JSON, executes XAUUSD with risk engine
```

The 12 factors: real rates (TIP), 10Y nominal yield, dollar (UUP), Fed stance
(T-bill), fear (VIX), risk appetite (S&P), BTC competition, GLD flow proxy,
gold momentum, yield curve, duration bid (TLT), central-bank buying regime.

### Signal pipeline (v2)

The live signal stacks four layers on top of the base 12-factor model:

```
base 12-factor P(up)  ──►  ensemble ML (50% logistic + 35% boosted trees
  │                         + sentiment nudge, clipped to [0.05, 0.95])
  │                    ──►  raw LONG/SHORT/FLAT at 0.80/0.20
  │                         (GOLD_LONG_T / GOLD_SHORT_T env overrides)
  │                    ──►  trend filter (MA50/MA200, ATR14 from factors.py)
  ▼
Narrative layer (three free sources, weights renormalize over whichever have data):
- `public_apis_ai.py`: macro headlines (Google News RSS) → lexicon sentiment
- `youtube_memory.py`: curated gold/macro YouTube channels (see below)
- `papers.py`: arXiv gold-forecasting papers → literature consensus
Blended narrative → rolling 30-day z-score (`narrative.py`) → only |z|>2
anomalies nudge the ensemble, max ±10%. (FinBERT/LLM backends are a
documented drop-in via `narrative.set_backend()`.)
  ▼
microstructure: **real 1-minute data** (`intraday.py`: Dukascopy XAU/USD ticks,
Kraken PAXG fallback) → session VWAP, VWAP z-score, POC/VAH/VAL, volume ratio
+ order-flow audit (audit.py). Falls back to the old daily synthesis only if
both feeds fail — and says so in `microstructure.data_source`.
  ▼
decision critic (critic.py): tiered clearance —
  FULL (1.0x): directional P ≥ 80% and MCDA ≥ 0.70, no fatal flaws
  QUARTER (0.25x): P ≥ 65% and MCDA ≥ 0.50 — exploratory, gathers OOS data
  FLAT: anything else or any fatal flaw
  ▼
regime exposure (`backtest.regime_exposure`): 1.5x in confirmed low-volatility
bull regimes (price > 200MA and 20d vol < 1y median), else 1.0x.
Exposed to the EA as `exposure_mult` (toggleable via InpUseDynamicExposure).
```

`daemon.py` runs the pipeline on a loop; `heatmap.py` renders factor heatmaps.
`shadow.py` logs every layer's daily recommendation without executing — see below.

### Watch-and-learn: YouTube channels + research papers

The bot studies two outside sources and folds them into the narrative layer:

**YouTube** (`gold_oracle/youtube_memory.py`, keyless):
- Watches curated gold/macro channels via their official RSS feeds —
  default list in `data/watch_channels.json` (Bloomberg Television, Real Vision
  Finance, Peter Schiff, Milk Road Macro). Edit the file to add handles; they
  resolve to channel IDs automatically.
- Scores each video's gold stance (title + description + captions when
  available) with a transparent lexicon; recent videos weigh more.
- **Learning loop:** every video stores the gold price at watch time. Once a
  video is 20 days old, the engine grades it — did gold move with the video's
  stance? — building a per-channel hit rate. Channels with a proven-bad record
  get down-weighted automatically.
- Note: YouTube 404s the feed endpoint for some datacenter IPs. It works on
  normal residential connections; `yt-dlp` (if installed) is used as fallback.
  The dashboard shows the feed status honestly.

**Research papers** (`gold_oracle/papers.py`, keyless via arXiv API):
- Pulls gold price forecasting papers, dedupes by arXiv ID.
- Scores directional findings from abstracts (usually neutral — most papers
  are methodological, and the engine is honest about that).
- More usefully: extracts which predictors the literature keeps citing, mapped
  to our 12 factors (e.g. dollar, momentum, real yields) — literature backing
  (or a warning) for the model's factor set. Shown in the dashboard.

The daemon refreshes both on its loop; `signal.py` only ever reads the local
memory files (no network), so signal generation stays fast and offline-safe.

### Shadow logging (validating the unproven layers)

The ensemble, narrative, microstructure and critic layers have no walk-forward
history. `gold_oracle/shadow.py` records every layer's daily recommendation
(base stance, ensemble stance, critic stance, narrative z) to
`data/shadow_log.jsonl` — **without routing anything to execution**. Mature
entries (20+ days old) are graded against the actual gold move:

```bash
python -m gold_oracle.shadow --report
```

Review after ~3 months / 60+ graded entries per layer before granting the new
layers capital allocation. The daemon and `signal.py` log automatically.

### Cron observability

Wrap the scheduled run so failures page you:

```bash
export HEALTHCHECKS_URL="https://hc-ping.com/<your-uuid>"
# and/or
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
python -m gold_oracle.notify -- python -m gold_oracle.signal
```

Pings `/start` on launch, success on exit 0, `/fail` on crash — plus a Discord
embed on failure. Without either env var it runs the command unwatched (with
a warning, never silently).

### Environment

Standardize on the pinned requirements and a venv so local runs match prod:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m gold_oracle.signal   # real sklearn ensemble, not the fallback
```

## Dashboard

A live GUI for the whole model — base factors, ensemble, sentiment, critic,
microstructure, the bot status, and the backtest:

```bash
python3 -m gold_oracle.signal   # refresh the signal first
python3 dashboard/serve.py      # -> http://localhost:8081
```

Shows: ensemble P(up) hero with the audited stance, all 12 factor cards
(live z-scores, weights, contributions), the composite → logistic math,
ensemble blend breakdown, AI-sentiment and critic verdict panels, VWAP /
volume-profile microstructure, signal history chart, walk-forward backtest
vs buy-and-hold, and a "bot right now" panel mirroring the EA's logic
(stale >36h ⇒ flat by design).

## Honest backtest (2019–2026, walk-forward, refit yearly)

Base 12-factor model, stance thresholds 0.53/0.47 (env-overridable defaults),
dynamic exposure 1.5x in confirmed low-volatility bull regimes,
**realistic costs: ~8bps spread per one-way position change** (`GOLD_COST_BPS`):

| | Accuracy | AUC | CAGR | Sharpe | MaxDD | Win rate | Profit factor |
|---|---|---|---|---|---|---|---|
| Model stances (positioned 72% of the time) | 61% hit rate when positioned | 0.50 | — | — | — | — | — |
| Signal strategy | — | — | 13.9% | 0.96 | −16.8% | 61.7% | 2.48 |
| Buy-and-hold gold | — | — | 15.8% | 1.04 | −23.2% | — | — |

Read this carefully: the model **still doesn't beat buy-and-hold in a bull
market** — but the gap is small (13.9% vs 15.8% CAGR) with a shallower max
drawdown (−16.8% vs −23.2%). Dynamic exposure (mean 1.16x, boosted ~1/3 of the
time in calm bull regimes) captured upside the static 1.0x cap left behind.
Expectancy is +1.77% per 20-day trade over 60 sampled trades.
Use it as a **regime filter / risk overlay**, not a money printer.

Note: the walk-forward above covers the base 12-factor model only. The ensemble ML,
AI sentiment, microstructure and critic layers are newer than the test window's
methodology and have no walk-forward history — treat their contribution as unproven.

**2026-10-10: full-pipeline walk-forward** (ensemble + critic probability tiers,
no sentiment/microstructure — no historical data exists for those): the critic's
80%/65% bars proved nearly unreachable — the ensemble's blended probability maxes
at ~0.76, so it took only 5 trades in 7 years (0.01% CAGR). Two real bugs were fixed
along the way: the "50%+35%" blend was never renormalized (capped at 0.85), and the
boosted-trees leg boosted raw probabilities instead of log-odds (max confidence 0.63).
Even fixed, the honest conclusion stands: **genuine 80%+ confidence on a 20-day gold
move is rare because the features are weak (AUC 0.50)** — the bar was set from aspiration,
not from what the model can produce. Recalibrating the bar to the model's actual range
(FULL ≥0.70, QUARTER ≥0.60) is the pragmatic fix; the sentiment nudge (±10%) remains
the designed path into 80%+ when news is extreme.

Data note: weekend/holiday rows where gold didn't trade are dropped (crypto trades
24/7 and was keeping NaN-gold rows alive). All figures are on trading days.

## Setup

### 1. Python brain (any machine; ideally the one running MT5)

```bash
pip install -r requirements.txt
python -m gold_oracle.backtest   # verify for yourself first
python -m gold_oracle.signal     # writes signals/gold_signal.json
```

Run `signal.py` on a schedule (cron / Windows Task Scheduler, e.g. every 6h).
Point it at MT5's Files folder so the EA can read it:

```bash
# Windows example
set GOLD_SIGNAL_PATH=C:\Users\<you>\AppData\Roaming\MetaQuotes\Terminal\<id>\MQL5\Files\gold_signal.json
```

Optional: `FRED_API_KEY` is not required — everything works keyless. Central-bank
regime lives in `cb_regime.json`; refresh it quarterly from World Gold Council
data (default reflects the post-2022 record-buying regime).

### 2. MT5 EA on Exness

1. Open Exness MT5 → File → Open Data Folder → `MQL5/Experts` → copy
   `mql5/GoldOracle_Exness.mq5` there.
2. Open MetaEditor (F4), compile (F7). **This file was carefully written but
   not compiled by its author — compile before use.**
3. Copy the latest `gold_signal.json` into `MQL5/Files/`.
4. Drag the EA onto an **XAUUSD** chart. Enable **Algo Trading**.
5. Inputs: risk per trade (default 1%), SL/TP in USD/oz (30/60), max spread,
   signal max age (36h — stale signals are ignored).

Behavior: LONG when signal=LONG, SHORT when SHORT, flat/close when FLAT or
the signal is stale. One position at a time, fixed-fractional sizing, spread
filter, broker stops-level respected.

### 3. Exness specifics

- Symbol is `XAUUSD` on standard Exness accounts (contract = 100 oz).
- Exness allows EAs/algo trading on MT5. Start on a **demo** account.
- Gold spread on Exness is usually tiny; the 500-point default cap is generous.

## Risk rules (non-negotiable)

- Demo first. Weeks, not hours.
- 0.5% risk per trade (EA default `InpRiskPercent`).
- EA risk engine (all on by default, tune via inputs):
  - **Daily loss limit** (`InpMaxDailyLossPct=2.0`): no new entries for the rest of the day.
  - **Max drawdown kill** (`InpMaxDrawdownPct=10.0`): drawdown from peak equity halts everything.
  - **Loss pause** (`InpMaxConsecutiveLosses=3`, `InpPauseHours=24`): auto-pause after 3 straight losses.
  - **Break-even** (+$20/oz → SL to entry +$2), **trailing stop** (from +$30, $10 steps), **partial TP** (50% at +$40/oz).
  - **Broker-error retries** (3×), **disconnect guard**, **CSV trade journal** (`MQL5/Files/goldoracle_journal.csv`), master `InpKillSwitch`.
- Python critic adds: news blackout (±30 min around NFP/CPI/FOMC), London/NY session filter, ≥1.5:1 risk-reward filter, technical-alignment confirmation.
- A stale or missing signal file = the EA does nothing (by design).
- Re-run the backtest yourself after any change: `python -m gold_oracle.backtest`.

## 50-point trading checklist coverage

The full checklist (technical stack 1–26, risk engine 28–39, validation 40–50) is implemented across the Python brain and the MQL5 EA:

| # | Item | Where |
|---|------|-------|
| 1 | XAU/USD only | EA symbol hardcoded |
| 2 | Multi-timeframe (H4/H1/M15/M5) | `technicals.py` → `mtf_trends`, `mtf_score` |
| 3–8 | Trend, EMA 20/50/200, RSI, MACD, ATR, ADX | `technicals.py` (daily) |
| 9–13 | S/R, BOS/CHOCH, sweeps, breakout+retest, candle patterns | `technicals.py` (M15/H1) |
| 14–15 | Volume, VWAP | `intraday.py` |
| 16 | News filter (CPI/NFP/FOMC) | `economic_calendar.py` blackout → critic fatal flaw |
| 17 | London/NY session filter | `technicals.py` session → critic fatal flaw |
| 18–19 | Spread filter, slippage protection | EA inputs |
| 20 | Regime detection | `regime_exposure()` |
| 21–23 | AI filter, quality score, multi-condition confirmation | ensemble + critic (now incl. Critic 9: tech alignment) |
| 24–26 | Auto SL/TP, R:R filter | signal ATR-based; critic + EA require ≥1.5:1 |
| 27–28 | Dynamic sizing, 0.5% risk/trade | critic tier × exposure; `InpRiskPercent` |
| 29–30, 36, 50 | Daily loss limit, drawdown kill, loss pause, kill switch | EA risk engine |
| 31–32 | Max open trades, no duplicates | EA one-position logic |
| 33–35 | Break-even, trailing, partials | EA `ManagePosition()` |
| 37–38 | Execution retries, disconnect guard | EA |
| 39–43 | Journal, dashboard, win-rate, profit factor, expectancy | EA CSV journal; dashboard; backtest `stats()` |
| 44–46, 48 | Realistic costs, out-of-sample, walk-forward, ridge+refit | `backtest.py` (`GOLD_COST_BPS`) |
| 47, 49 | Demo forward test, live monitoring | yours to run (EA is demo-ready); `notify.py` + dashboard |

## Limitations

- Factors are slow-moving; the model times 20-day swings, not intraday moves.
- Central-bank demand is a manual quarterly input, not a live feed.
- Fed-expectations are proxied via yields/curve, not Fed funds futures.
- MQL5 file uncompiled (no MetaEditor here) — compile and forward-test on demo.
