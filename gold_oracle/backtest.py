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


def summarize(out: pd.DataFrame, model: GoldProbabilityModel) -> dict:
    """Machine-readable version of _report for the GUI."""
    y, p = out["y"].to_numpy(), out["p_up"].to_numpy()
    acc = float(((p >= 0.5).astype(int) == y).mean())
    brier = float(np.mean((p - y) ** 2))
    hi_conf = (p >= 0.60) | (p <= 0.40)
    if hi_conf.sum():
        hi_hit = float((((p[hi_conf] >= 0.5).astype(int) == y[hi_conf]).mean()))
    else:
        hi_hit = None
    summ = {
        "n": len(out),
        "from": str(out.index[0].date()),
        "to": str(out.index[-1].date()),
        "accuracy": round(acc, 3),
        "auc": round(_auc(y, p), 3),
        "brier": round(brier, 4),
        "hi_conf_hit_rate": round(hi_hit, 3) if hi_hit is not None else None,
        "hi_conf_n": int(hi_conf.sum()),
        "weights": {k: round(v, 3) for k, v in
                     sorted(model.weights_.items(), key=lambda kv: -kv[1])},
    }
    for label, col in [("strategy", "strat_ret"), ("buy_hold", "bh_ret")]:
        r = out[col]
        r_n = r.iloc[::HORIZON]
        years = len(r_n) * HORIZON / 252
        cagr = (1 + r_n).prod() ** (1 / max(years, 1e-9)) - 1
        per = 252 / HORIZON
        sharpe = r_n.mean() / (r_n.std() + 1e-12) * np.sqrt(per)
        dd = float(((1 + r_n).cumprod() / (1 + r_n).cumprod().cummax() - 1).min())
        summ[label] = {"cagr": round(float(cagr), 4),
                       "sharpe": round(float(sharpe), 2),
                       "maxdd": round(dd, 4)}
    # downsampled equity curves for the chart
    for label, col in [("strategy", "strat_ret"), ("buy_hold", "bh_ret")]:
        eq = (1 + out[col].iloc[::HORIZON]).cumprod()
        idx = np.linspace(0, len(eq) - 1, min(120, len(eq))).astype(int)
        summ[label]["equity_curve"] = [
            [str(eq.index[i].date()), round(float(eq.iloc[i]), 3)] for i in idx]
    return summ


def _report(out: pd.DataFrame, model: GoldProbabilityModel) -> None:
    s = summarize(out, model)
    print(f"n={s['n']}  acc={s['accuracy']:.3f}  AUC={s['auc']:.3f}  Brier={s['brier']:.4f}")
    print("weights:")
    for k, v in s["weights"].items():
        print(f"  {k:10s} {v:.3f}")
    print(f"strategy CAGR={s['strategy']['cagr']:7.2%}  Sharpe={s['strategy']['sharpe']:5.2f}  MaxDD={s['strategy']['maxdd']:7.2%}")
    print(f"buy-hold CAGR={s['buy_hold']['cagr']:7.2%}  Sharpe={s['buy_hold']['sharpe']:5.2f}  MaxDD={s['buy_hold']['maxdd']:7.2%}")


def run() -> None:
    from .data import download

    df = download()
    f = build_features(df)
    print(f"feature rows: {len(f)} from {f.index[0].date()} to {f.index[-1].date()}")
    walk_forward(f)


if __name__ == "__main__":
    run()
