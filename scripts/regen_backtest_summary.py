#!/usr/bin/env python3
"""Regenerate data/backtest_summary.json with costs + trade stats (items 41-44)."""
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from gold_oracle.backtest import walk_forward, walk_forward_full, summarize, FEATURES
from gold_oracle.data import download
from gold_oracle.factors import build_features
from gold_oracle.model import GoldProbabilityModel

df = download()
f = build_features(df)
out = walk_forward(f, gold=df["gold"], verbose=False)
out_full = walk_forward_full(f, gold=df["gold"], verbose=False)

# final fit on all data for the summary weights
ff = f.dropna()
model = GoldProbabilityModel()
model.fit(ff[FEATURES], ff["target"])

s = summarize(out, model)
s_full = summarize(out_full, model)
s["full_pipeline"] = s_full["strategy"]
s["full_pipeline_note"] = ("ensemble (50% logistic + 35% boosted trees) + critic probability tiers "
    "(FULL ≥80%, QUARTER ≥65%) + exposure + costs; sentiment nudge, microstructure, MCDA, "
    "news-blackout and session checks are live-only and not in this test")
st = s["strategy"]
fp = s["full_pipeline"]
print("BASE :", round(st["cagr"], 4), round(st["sharpe"], 2), round(st["maxdd"], 4),
      "win", st["win_rate"], "trades", st["n_trades"])
print("FULL :", round(fp["cagr"], 4), round(fp["sharpe"], 2), round(fp["maxdd"], 4),
      "win", fp["win_rate"], "trades", fp["n_trades"])
with open(os.path.join(BASE, "data", "backtest_summary.json"), "w") as fh:
    json.dump(s, fh, indent=1)
print("written")
