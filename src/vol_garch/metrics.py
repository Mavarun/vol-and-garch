"""Volatility forecast loss functions: MSE and QLIKE."""

from __future__ import annotations

import numpy as np


def mse_loss(forecast: np.ndarray, realized: np.ndarray) -> float:
    """Mean squared error between variance forecast and target (e.g. r²)."""
    f = np.asarray(forecast, dtype=float)
    y = np.asarray(realized, dtype=float)
    mask = np.isfinite(f) & np.isfinite(y)
    if mask.sum() == 0:
        return float("nan")
    err = f[mask] - y[mask]
    return float(np.mean(err ** 2))


def qlike_loss(forecast: np.ndarray, realized: np.ndarray) -> float:
    """QLIKE / Gaussian quasi-likelihood loss: log(f) + y/f (f>0).

    Lower is better. Common in volatility forecast evaluation (Patton 2011).
    """
    f = np.asarray(forecast, dtype=float)
    y = np.asarray(realized, dtype=float)
    mask = np.isfinite(f) & np.isfinite(y) & (f > 0)
    if mask.sum() == 0:
        return float("nan")
    ff = f[mask]
    yy = y[mask]
    return float(np.mean(np.log(ff) + yy / ff))


def summarize_losses(forecast: np.ndarray, realized: np.ndarray) -> dict[str, float]:
    return {
        "mse": mse_loss(forecast, realized),
        "qlike": qlike_loss(forecast, realized),
        "n": float(np.sum(np.isfinite(forecast) & np.isfinite(realized))),
    }
