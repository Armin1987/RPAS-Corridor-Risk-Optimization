"""Scenario weights for the first stage, from historical (context, outcome) pairs.

Training data: for each past day d, context X_d (known day-ahead: forecast wind
profile, month, weekday) and outcome Y_d (realised hourly wind trajectory, requests).
A first-stage decision for a new day with context x uses scenarios {Y_d} with weights:

  deterministic      single scenario = point forecast (Y_hat(x))
  SAA                w_d = 1/n                                   (ignores x)
  seasonal SAA       w_d = 1/|S| for days in the same calendar window (fair 'no forecast'
                     benchmark: uses seasonality but not the forecast)
  kNN contextual     w_d = 1/k for the k nearest X_d to x        (Bertsimas & Kallus 2020)
  RF contextual      w_d = mean over trees of 1{d shares x's leaf}/|leaf|   (idem)
Whole-day trajectories are kept intact, so intraday wind correlation is preserved.
Leakage rule: training days strictly precede every evaluation day.
"""
from __future__ import annotations
import numpy as np


def saa_weights(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def seasonal_weights(train_doy: np.ndarray, doy: int, half_window: int = 30) -> np.ndarray:
    d = np.abs(train_doy - doy)
    d = np.minimum(d, 365 - d)
    m = (d <= half_window).astype(float)
    if m.sum() == 0:
        return saa_weights(train_doy.size)
    return m / m.sum()


def knn_weights(X_train: np.ndarray, x: np.ndarray, k: int, scale: np.ndarray | None = None):
    s = np.ones(X_train.shape[1]) if scale is None else scale
    d = np.sqrt((((X_train - x) / s) ** 2).sum(axis=1))
    idx = np.argsort(d, kind="stable")[:k]
    w = np.zeros(X_train.shape[0]); w[idx] = 1.0 / k
    return w


def rf_weights(forest, X_train: np.ndarray, x: np.ndarray) -> np.ndarray:
    """forest: fitted sklearn RandomForestRegressor."""
    leaves_tr = forest.apply(X_train)          # (n, n_trees)
    leaf_x = forest.apply(x.reshape(1, -1))[0]
    w = np.zeros(X_train.shape[0])
    for j in range(leaves_tr.shape[1]):
        same = leaves_tr[:, j] == leaf_x[j]
        w[same] += 1.0 / same.sum()
    return w / leaves_tr.shape[1]


def reduce_scenarios(w: np.ndarray, max_n: int):
    """Keep the max_n largest weights (ties broken by index) and renormalise; returns
    (indices, weights).  Exact when the support is already <= max_n."""
    idx = np.flatnonzero(w > 0)
    if idx.size > max_n:
        idx = idx[np.argsort(-w[idx], kind="stable")[:max_n]]
    ww = w[idx] / w[idx].sum()
    return idx, ww
