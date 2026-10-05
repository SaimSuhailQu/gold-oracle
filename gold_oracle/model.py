"""Probability model: factor z-scores -> P(gold up over HORIZON days).

Transparent by design: composite = sum(w_i * z_i) with weights from each
factor's univariate information coefficient (floored at 0 — every feature is
built bullish-positive), then Platt-style logistic calibration maps the
composite to a probability. No black boxes; every weight is inspectable.

Logistic regression is implemented in numpy (no sklearn dependency).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .factors import FEATURES


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _logistic_fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 100) -> np.ndarray:
    """Newton-Raphson logistic regression. X includes intercept column."""
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(iters):
        pr = _sigmoid(X @ beta)
        w = pr * (1 - pr) + 1e-8
        grad = X.T @ (y - pr) - l2 * beta
        hess = -(X.T * w) @ X - l2 * np.eye(p)
        step = np.linalg.solve(hess, grad)
        beta -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


class GoldProbabilityModel:
    def __init__(self) -> None:
        self.weights_: dict[str, float] = {}
        self.cal_a_: float = 0.0  # intercept
        self.cal_b_: float = 1.0  # slope on composite

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GoldProbabilityModel":
        X = X[FEATURES].astype(float)
        yv = y.astype(float).to_numpy()
        # Univariate IC per factor -> non-negative normalized weights
        w: dict[str, float] = {}
        for col in FEATURES:
            xv = X[col].to_numpy()
            mask = np.isfinite(xv) & np.isfinite(yv)
            if mask.sum() < 50:
                w[col] = 0.0
                continue
            c = np.corrcoef(xv[mask], yv[mask])[0, 1]
            w[col] = float(max(0.0, c)) if np.isfinite(c) else 0.0
        total = sum(w.values()) or 1.0
        self.weights_ = {k: v / total for k, v in w.items()}

        comp = self._composite(X)
        Xc = np.column_stack([np.ones(len(comp)), comp])
        beta = _logistic_fit(Xc, yv)
        self.cal_a_, self.cal_b_ = float(beta[0]), float(beta[1])
        return self

    def _composite(self, X: pd.DataFrame) -> np.ndarray:
        return sum(X[c].to_numpy() * self.weights_.get(c, 0.0) for c in FEATURES)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        comp = self._composite(X[FEATURES].astype(float))
        return _sigmoid(self.cal_a_ + self.cal_b_ * comp)

    def contributions(self, row: pd.Series) -> dict[str, float]:
        """Per-factor contribution to today's composite (for explainability)."""
        return {c: float(row[c]) * self.weights_.get(c, 0.0) for c in FEATURES}
