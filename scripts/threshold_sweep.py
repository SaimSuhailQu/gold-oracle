#!/usr/bin/env python3
"""Threshold sweep for critic probability tiers (evidence-based recalibration).

Runs the full-pipeline walk-forward ONCE to get out-of-sample blended
probabilities, then sweeps FULL/QUARTER threshold candidates and reports
trade count + performance for each. Pick thresholds from this table, not
from aspiration.
"""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gold_oracle.backtest import (build_features, walk_forward_full, summarize,
                                  regime_exposure, COST_BPS, HORIZON)

def tier_stats(out, full_t, quarter_t):
    p = out["p_up"].to_numpy()
    p_dir = np.maximum(p, 1.0 - p)
    direction = np.where(p >= 0.5, 1.0, -1.0)
    size_mult = np.where(p_dir >= full_t, 1.0, np.where(p_dir >= quarter_t, 0.25, 0.0))
    pos = direction * size_mult
    n_trades = int((pd.Series(pos).diff().abs().fillna(0) > 0).sum() // 2)
    positioned = (pos != 0)
    win = ((np.sign(pos[positioned]) == np.sign(out["y"].to_numpy()[positioned])).mean()
           if positioned.any() else float("nan"))
    # rough CAGR from cumulative strat return
    tmp = out.copy()
    tmp["pos"] = pos
    turnover = tmp["pos"].diff().abs().fillna(0.0)
    tmp["cost"] = turnover * COST_BPS / 10000.0 * tmp["exposure"]
    tmp["strat_ret"] = tmp["pos"] * tmp["fwd_ret"] / 100.0 * tmp["exposure"] - tmp["cost"]
    eq = (1 + tmp["strat_ret"]).cumprod()
    yrs = len(tmp) / 252
    cagr = eq.iloc[-1] ** (1 / max(yrs, 1e-9)) - 1
    return n_trades, positioned.mean(), win, cagr

def main():
    from gold_oracle.data import download
    df = download()
    f = build_features(df)
    print(f"[sweep] features: {len(f)} rows — running walk-forward (one fit)...", flush=True)
    out = walk_forward_full(f, verbose=False, gold=df["gold"])
    print(f"[sweep] {len(out)} out-of-sample predictions", flush=True)
    print(f"{'FULL':>6} {'QUARTER':>7} {'trades':>6} {'pos%':>6} {'win':>6} {'CAGR':>7}")
    for full_t in (0.68, 0.70, 0.72, 0.75, 0.80):
        for quarter_t in (0.58, 0.60, 0.62, 0.65):
            if quarter_t >= full_t:
                continue
            n, pos_pct, win, cagr = tier_stats(out, full_t, quarter_t)
            print(f"{full_t:>6.2f} {quarter_t:>7.2f} {n:>6d} {pos_pct:>5.1%} {win:>6.1%} {cagr:>6.1%}")

if __name__ == "__main__":
    main()
