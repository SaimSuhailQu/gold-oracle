"""Walk-forward evaluation of the gold probability model.

Protocol (no lookahead):
  - features/targets built only from data available at each date
  - expanding training window, model refit every REFIT_DAYS
  - predict P(up over next HORIZON days) out-of-sample
  - strategy: long gold if p >= LONG_T, short if p <= SHORT_T, else flat
    (defaults 0.53/0.47, override with GOLD_LONG_T / GOLD_SHORT_T env vars)
  - benchmark: buy-and-hold gold

Metrics: accuracy, AUC, Brier score, strategy CAGR / Sharpe / max drawdown.
"""
from __future__ import annotations

import numpy as np
import os
import pandas as pd

from .factors import FEATURES, HORIZON, build_features
from .model import GoldProbabilityModel

REFIT_DAYS = 252
MIN_TRAIN = 4 * 252
LONG_T = float(os.environ.get("GOLD_LONG_T", "0.53"))
SHORT_T = float(os.environ.get("GOLD_SHORT_T", "0.47"))
BULL_MULT = float(os.environ.get("GOLD_BULL_MULT", "1.5"))  # exposure in confirmed low-vol bull


def regime_exposure(gold: pd.Series, bull_mult: float = BULL_MULT) -> pd.Series:
    """Dynamic exposure: bull_mult in a confirmed low-volatility bull regime
    (price above 200-day MA *and* 20d realized vol below its 1y median),
    otherwise 1.0x. A regime filter's job is drawdown protection, but capping
    winners at 1.0x in a calm bull market leaves return on the table."""
    ma200 = gold.rolling(200, min_periods=50).mean()
    vol20 = gold.pct_change().rolling(20).std() * np.sqrt(252)
    vol_med = vol20.rolling(252, min_periods=100).median()
    bull_lowvol = (gold > ma200) & (vol20 < vol_med)
    return pd.Series(np.where(bull_lowvol.fillna(False), bull_mult, 1.0), index=gold.index)


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(p)
    y = y[order]
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(p)) + 1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def walk_forward(f: pd.DataFrame, verbose: bool = True,
                 gold: pd.Series | None = None) -> pd.DataFrame:
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
    # dynamic exposure: scale up in confirmed low-vol bull regimes
    if gold is not None:
        out["exposure"] = regime_exposure(gold).reindex(out.index).fillna(1.0).values
    else:
        out["exposure"] = 1.0
    # approximate daily P&L attribution: position * forward return * exposure
    out["strat_ret"] = out["pos"] * out["fwd_ret"] / 100.0 * out["exposure"]
    out["bh_ret"] = out["fwd_ret"] / 100.0
    if verbose:
        _report(out, model)
    return out


def _report(out: pd.DataFrame, model: GoldProbabilityModel) -> None:
    y, p = out["y"].to_numpy(), out["p_up"].to_numpy()
    acc = ((p >= 0.5).astype(int) == y).mean()
    brier = float(np.mean((p - y) ** 2))
    print(f"n={len(out)}  acc={acc:.3f}  AUC={_auc(y, p):.3f}  Brier={brier:.4f}")
    if "exposure" in out.columns:
        ex = out["exposure"].to_numpy()
        print(f"exposure: mean={ex.mean():.2f}x  boosted={(ex > 1.0).mean():.1%} of the time")
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
    walk_forward(f, gold=df["gold"])


def summarize(out: pd.DataFrame, model: GoldProbabilityModel) -> dict:
    """Compact JSON-serializable summary for the dashboard (data/backtest_summary.json)."""
    y, p = out["y"].to_numpy(), out["p_up"].to_numpy()
    n = len(out)
    acc = float(((p >= 0.5).astype(int) == y).mean())
    auc = float(_auc(y, p))
    brier = float(np.mean((p - y) ** 2))
    hi_conf = (p >= LONG_T) | (p <= SHORT_T)
    if hi_conf.sum():
        hi_hit = float((((p[hi_conf] >= 0.5).astype(int) == y[hi_conf]).mean()))
    else:
        hi_hit = None
    out_s = out.iloc[::HORIZON]
    years = len(out_s) * HORIZON / 252
    per = 252 / HORIZON

    def stats(col):
        r = out_s[col].to_numpy()
        cagr = float((1 + r).prod() ** (1 / max(years, 1e-9)) - 1)
        sharpe = float(r.mean() / (r.std() + 1e-12) * np.sqrt(per))
        eq = (1 + r).cumprod()
        dd = float(((eq / np.maximum.accumulate(eq)) - 1).min())
        curve = [[d.strftime("%Y-%m-%d"), round(float(v), 4)]
                 for d, v in zip(out_s.index[:: max(1, len(out_s) // 90)],
                                 eq[:: max(1, len(out_s) // 90)])]
        return {"cagr": round(cagr, 4), "sharpe": round(sharpe, 2),
                "maxdd": round(dd, 4), "equity_curve": curve}

    return {
        "n": n, "from": str(out.index[0].date()), "to": str(out.index[-1].date()),
        "long_t": LONG_T, "short_t": SHORT_T,
        "accuracy": round(acc, 4), "auc": round(auc, 4), "brier": round(brier, 4),
        "hi_conf_hit_rate": round(hi_hit, 4) if hi_hit is not None else None,
        "hi_conf_n": int(hi_conf.sum()),
        "exposure_mean": round(float(out["exposure"].mean()), 3) if "exposure" in out.columns else 1.0,
        "exposure_boosted_pct": round(float((out["exposure"] > 1.0).mean()), 3) if "exposure" in out.columns else 0.0,
        "weights": {k: round(float(v), 4) for k, v in model.weights_.items()},
        "strategy": stats("strat_ret"), "buy_hold": stats("bh_ret"),
    }


if __name__ == "__main__":
    run()
