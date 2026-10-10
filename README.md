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
Blended narrative nudges the ensemble at most ±10%.
  ▼
microstructure (vwap.py, profile.py, footprint.py, gex.py) + order-flow
audit (audit.py) → VWAP z-score, POC/VAH/VAL, audit verdict
  ▼
decision critic (critic.py): 5 adversarial checks (probability floor 80%,
macro/momentum contradiction, trend disconnect, VWAP overextension, model
divergence) + MCDA score ≥ 0.70 → final_audited_signal (or FLAT if blocked)
```

`daemon.py` runs the pipeline on a loop; `heatmap.py` renders factor heatmaps.
Thresholds and the critic's probability floor are deliberately strict — the
pipeline prefers standing aside to forcing a trade.

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

Base 12-factor model, stance thresholds 0.53/0.47 (env-overridable defaults):

| | Accuracy | AUC | CAGR | Sharpe | MaxDD |
|---|---|---|---|---|---|\n| Model stances (positioned 72% of the time) | 61% hit rate when positioned | 0.50 | — | — | — |
| Signal strategy | — | — | 11.9% | 0.95 | −16.7% |
| Buy-and-hold gold | — | — | 15.8% | 1.04 | −23.2% |

Read this carefully: the model **still does not beat buy-and-hold in a bull market**,
but with the wider stance bands it stays invested through most of the rally instead of
sitting out — 11.9% CAGR at 0.95 Sharpe with a shallower max drawdown (−16.7% vs −23.2%).
Use it as a **regime filter / risk overlay** (size up when P(up) is high, stand aside
when low) — not as a money printer. It cuts drawdowns; it does not manufacture
return in a one-way market.

Note: the walk-forward above covers the base 12-factor model only. The ensemble ML,
AI sentiment, microstructure and critic layers are newer than the test window's
methodology and have no walk-forward history — treat their contribution as unproven.

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
- 1% risk per trade max. The model is a filter, not an oracle.
- A stale or missing signal file = the EA does nothing (by design).
- Re-run the backtest yourself after any change: `python -m gold_oracle.backtest`.

## Limitations

- Factors are slow-moving; the model times 20-day swings, not intraday moves.
- Central-bank demand is a manual quarterly input, not a live feed.
- Fed-expectations are proxied via yields/curve, not Fed funds futures.
- MQL5 file uncompiled (no MetaEditor here) — compile and forward-test on demo.
