"""Live signal: fit on all history, emit today's P(gold up) + factor attribution.

Writes signals/gold_signal.json — the MT5 EA on Exness reads this file.
Run on a schedule (e.g. every 6h / daily) on the same machine as MT5, with
SIGNAL_PATH pointing at <MT5>/MQL5/Files/gold_signal.json, or copy it over.

Schema:
  { timestamp, symbol, horizon_days, p_up, signal, confidence,
    gold_price, factors: {name: contribution}, model_weights: {...} }
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pandas as pd

from .data import latest
from .factors import FEATURES, HORIZON, build_features
from .model import GoldProbabilityModel

LONG_T = 0.60
SHORT_T = 0.40

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGNAL_PATH = os.environ.get(
    "GOLD_SIGNAL_PATH", os.path.join(BASE, "signals", "gold_signal.json")
)


def generate(df: pd.DataFrame | None = None) -> dict:
    df = latest() if df is None else df
    f = build_features(df)
    X, y = f[FEATURES], f["target"]
    model = GoldProbabilityModel().fit(X, y)

    row = f.iloc[[-1]]
    p_up = float(model.predict_proba(row[FEATURES])[0])
    signal = "LONG" if p_up >= LONG_T else ("SHORT" if p_up <= SHORT_T else "FLAT")
    contrib = model.contributions(f[FEATURES].iloc[-1])
    zscores = {c: round(float(f[c].iloc[-1]), 4) for c in FEATURES}
    composite = round(float(sum(contrib.values())), 4)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "timestamp_epoch": int(datetime.now(timezone.utc).timestamp()),
        "symbol": "XAUUSD",
        "horizon_days": HORIZON,
        "p_up": round(p_up, 4),
        "p_down": round(1 - p_up, 4),
        "signal": signal,
        "confidence": round(abs(p_up - 0.5) * 2, 4),
        "gold_price": round(float(df["gold"].iloc[-1]), 2),
        "composite": composite,
        "calibration": {"a": round(model.cal_a_, 4), "b": round(model.cal_b_, 4)},
        "factors": {k: round(v, 4) for k, v in contrib.items()},
        "factor_zscores": zscores,
        "model_weights": {k: round(v, 4) for k, v in model.weights_.items()},
    }


HISTORY_PATH = os.path.join(BASE, "signals", "history.jsonl")


def append_history(sig: dict) -> None:
    """One line per signal run — powers the GUI's history charts."""
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    rec = {"ts": sig["timestamp_epoch"], "p_up": sig["p_up"],
           "signal": sig["signal"], "gold_price": sig["gold_price"],
           "zscores": sig["factor_zscores"]}
    with open(HISTORY_PATH, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def main() -> None:
    sig = generate()
    os.makedirs(os.path.dirname(SIGNAL_PATH), exist_ok=True)
    with open(SIGNAL_PATH, "w") as fh:
        json.dump(sig, fh, indent=2)
    append_history(sig)
    print(json.dumps(sig, indent=2))
    print(f"[signal] wrote {SIGNAL_PATH} + history")


if __name__ == "__main__":
    main()
