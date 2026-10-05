"""Walk-forward evaluation of the gold probability model.

Protocol (no lookahead):
  - features/targets built only from data available at each date
  - expanding training window, model refit every REFIT_DAYS
  - predict P(up over next HORIZON days) out-of-sample
  - strategy: long gold if p >= 0.60, short if p <= 0.40, else flat
  - benchmark: buy-and-hold gold

Metrics: accuracy, AUC, Brier score, strategy CAGR / Sharpe / max drawdown.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .factors import FEATURES, HORIZON, build_features
from .model import GoldProbabilityModel

REFIT_DAYS = 252
MIN_TRAIN = 4 * 252
LONG_T = 0.60
SHORT_T = 0.40


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(p)
    y = y[order]
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(p)) + 1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def walk_forward(f: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    f = f.dropna().copy()
    X_all, y_all = f[FEATURES], f["target"]
    idx = f.index
    preds: list[float] = []
    pred_idx = []
    model = GoldProbabilityModel()

    t = MIN_TRAIN
    while t < len(f) - HORIZON:
        model.fit(X_all.iloc[:t], y_all.iloc[:t])
        end = min(t + REFIT_DAYS, len(f) - HORIZON)
        p = model.predict_proba(X_all.iloc[t:end])
        preds.extend(p.tolist())
        pred_idx.extend(idx[t:end])
        t = end

    out = pd.DataFrame(index=pd.DatetimeIndex(pred_idx))
    out["p_up"] = preds
    out["y"] = y_all.loc[out.index].to_numpy()
    out["fwd_ret"] = f.loc[out.index, "fwd_ret"].to_numpy()
    # position held over the horizon, scaled by conviction
    out["pos"] = np.where(out["p_up"] >= LONG_T, 1.0, np.where(out["p_up"] <= SHORT_T, -1.0, 0.0))
    # approximate daily P&L attribution: position * forward return / horizon
    out["strat_ret"] = out["pos"] * out["fwd_ret"] / 100.0 / HORIZON * HORIZON
    out["bh_ret"] = out["fwd_ret"] / 100.0
    if verbose:
        _report(out, model)
    return out


def _report(out: pd.DataFrame, model: GoldProbabilityModel) -> None:
    y, p = out["y"].to_numpy(), out["p_up"].to_numpy()
    acc = ((p >= 0.5).astype(int) == y).mean()
    brier = float(np.mean((p - y) ** 2))
    print(f"n={len(out)}  acc={acc:.3f}  AUC={_auc(y, p):.3f}  Brier={brier:.4f}")
    print("weights:")
    for k, v in sorted(model.weights_.items(), key=lambda kv: -kv[1]):
        print(f"  {k:10s} {v:.3f}")
    for label, col in [("strategy", "strat_ret"), ("buy-hold", "bh_ret")]:
        r = out[col]
        # overlapping horizons: resample to non-overlapping for stats
        r_n = r.iloc[::HORIZON]
        years = len(r_n) * HORIZON / 252
        cagr = (1 + r_n).prod() ** (1 / max(years, 1e-9)) - 1
        per = 252 / HORIZON  # periods per year
        sharpe = r_n.mean() / (r_n.std() + 1e-12) * np.sqrt(per)
        dd = float(((1 + r_n).cumprod() / (1 + r_n).cumprod().cummax() - 1).min())
        print(f"{label:9s} CAGR={cagr:7.2%}  Sharpe={sharpe:5.2f}  MaxDD={dd:7.2%}")


def run() -> None:
    from .data import download

    df = download()
    f = build_features(df)
    print(f"feature rows: {len(f)} from {f.index[0].date()} to {f.index[-1].date()}")
    walk_forward(f)


if __name__ == "__main__":
    run()
