#!/usr/bin/env python3
"""Regenerate data/backtest_summary.json with costs + trade stats (items 41-44)."""
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from gold_oracle.backtest import walk_forward, summarize, FEATURES
from gold_oracle.data import download
from gold_oracle.factors import build_features
from gold_oracle.model import GoldProbabilityModel

df = download()
f = build_features(df)
out = walk_forward(f, gold=df["gold"], verbose=False)

# final fit on all data for the summary weights
ff = f.dropna()
model = GoldProbabilityModel()
model.fit(ff[FEATURES], ff["target"])

s = summarize(out, model)
st = s["strategy"]
print("CAGR:", st["cagr"], "Sharpe:", st["sharpe"], "MaxDD:", st["maxdd"])
print("win_rate:", st["win_rate"], "PF:", st["profit_factor"],
      "expectancy:", st["expectancy"], "trades:", st["n_trades"])
with open(os.path.join(BASE, "data", "backtest_summary.json"), "w") as fh:
    json.dump(s, fh, indent=1)
print("written")
