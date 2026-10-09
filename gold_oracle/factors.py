"""Factor engineering: the 12 gold drivers -> bullish-positive z-scored features.

Every feature is oriented so POSITIVE = bullish for gold, and z-scored on a
rolling window so the model sees *unusual* moves, not raw levels. All
features at row t use data available at t (no lookahead).

Feature map (driver -> proxy):
  real_rates   TIP ETF 20d return            (TIP rises when real yields fall)
  nom_yield   -TNX 20d change               (yields up = bearish)
  dollar      -UUP 20d return               (dollar up = bearish)
  fed_stance  -IRX 20d change               (short rates up = bearish)
  fear        +VIX 20d change               (fear up = bullish)
  risk_app    -SPX 20d return               (stocks up = mild headwind)
  btc_comp    -BTC 20d return               (crypto bid = mild headwind)
  etf_flow     GLD volume surge x price dir (investment flow)
  momentum     gold 60d return              (trend)
  curve        TNX-IRX slope level          (steeper = inflation bets = bullish)
  duration    +TLT 20d return               (bonds up = yields down = bullish)
  cb_regime    central-bank buying regime   (structural; slow-moving input)
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

FEATURES = [
    "real_rates", "nom_yield", "dollar", "fed_stance", "fear", "risk_app",
    "btc_comp", "etf_flow", "momentum", "curve", "duration", "cb_regime",
]

REGIME_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cb_regime.json")
ZSCORE_WIN = 252
RET_WIN = 20
MOM_WIN = 60
HORIZON = 20  # trading days ahead for the probability target


def _zscore(s: pd.Series, win: int = ZSCORE_WIN) -> pd.Series:
    mu = s.rolling(win, min_periods=win // 2).mean()
    sd = s.rolling(win, min_periods=win // 2).std(ddof=0)
    return (s - mu) / sd.replace(0, np.nan)


def _ret(s: pd.Series, win: int = RET_WIN) -> pd.Series:
    return 100.0 * (np.log(s) - np.log(s.shift(win)))


def load_cb_regime() -> dict:
    """Slow-moving central-bank demand input. WGC quarterly data, manually
    refreshed; values are z-like scores (0 = neutral, +1 = record buying)."""
    if os.path.exists(REGIME_PATH):
        with open(REGIME_PATH) as f:
            return json.load(f)
    return {"default": 0.5, "note": "post-2022 record buying regime"}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # Work on gold trading days only: yfinance includes weekend rows (BTC
    # trades, gold doesn't) whose NaNs would poison rolling windows.
    d = df[df["gold"].notna()].copy()
    f = pd.DataFrame(index=d.index)

    f["real_rates"] = _zscore(_ret(d["tip"]))
    f["nom_yield"] = _zscore(-d["tnx"].diff(RET_WIN))
    f["dollar"] = _zscore(-_ret(d["dollar"]))
    f["fed_stance"] = _zscore(-d["irx"].diff(RET_WIN))
    f["fear"] = _zscore(d["vix"].diff(RET_WIN))
    f["risk_app"] = _zscore(-_ret(d["spx"]))
    f["btc_comp"] = _zscore(-_ret(d["btc"]))

    if "gld_volume" in d.columns:
        vol_ratio = d["gld_volume"].rolling(RET_WIN).mean() / d["gld_volume"].rolling(120, min_periods=60).mean()
        f["etf_flow"] = _zscore(vol_ratio * np.sign(_ret(d["gld"], RET_WIN).fillna(0)))
    else:
        f["etf_flow"] = _zscore(np.sign(_ret(d["gld"], RET_WIN).fillna(0)).astype(float))

    f["momentum"] = _zscore(_ret(d["gold"], MOM_WIN))
    f["curve"] = _zscore(d["tnx"] - d["irx"])
    f["duration"] = _zscore(_ret(d["tlt"]))

    # Central-bank regime: pre-2022 neutral, 2022+ strong buying (WGC data:
    # >1000t/yr vs ~500t historical norm). Refreshed manually via cb_regime.json.
    regime = load_cb_regime()
    base = float(regime.get("default", 0.5))
    f["cb_regime"] = np.where(f.index >= pd.Timestamp("2022-01-01"), base, 0.0)

    # Technical & Trend Context: 50-day and 200-day moving averages
    f["ma50"] = d["gold"].rolling(50, min_periods=20).mean()
    f["ma200"] = d["gold"].rolling(200, min_periods=50).mean()

    # Volatility indicator: 14-day True Range
    high = d["gold"]  # yfinance Close acts as proxy, diff captures daily swings
    prev_close = d["gold"].shift(1)
    tr = (d["gold"] - prev_close).abs()
    f["atr14"] = tr.rolling(14, min_periods=7).mean().bfill()

    # Target: 1 if gold is higher HORIZON days out, else 0
    f["fwd_ret"] = 100.0 * (np.log(d["gold"].shift(-HORIZON)) - np.log(d["gold"]))
    f["target"] = (f["fwd_ret"] > 0).astype(float)

    return f.dropna(subset=FEATURES + ["target"])
