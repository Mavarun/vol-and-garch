"""Realized / rolling volatility estimators from daily returns."""

from __future__ import annotations

import numpy as np


def rolling_realized_variance(returns: np.ndarray, window: int = 21) -> np.ndarray:
    """Trailing mean of squared returns (proxy for realized variance on daily data).

    Index ``t`` uses returns ``[t-window+1, ..., t]`` inclusive.
    Leading ``window-1`` entries are NaN.
    """
    r = np.asarray(returns, dtype=float)
    if window < 2:
        raise ValueError("window must be >= 2")
    if r.ndim != 1:
        raise ValueError("returns must be 1-d")

    sq = r ** 2
    out = np.full(r.shape[0], np.nan, dtype=float)
    if r.shape[0] < window:
        return out
    csum = np.cumsum(sq)
    # sum over [i-window+1, i] = csum[i] - csum[i-window]
    out[window - 1 :] = (csum[window - 1 :] - np.concatenate([[0.0], csum[: -window]])) / window
    return out


def rolling_std_variance(returns: np.ndarray, window: int = 21) -> np.ndarray:
    """Trailing sample variance of returns (ddof=1), aligned like realized variance.

    Used as a simple baseline forecast of next-day variance (squared return).
    """
    r = np.asarray(returns, dtype=float)
    if window < 2:
        raise ValueError("window must be >= 2")
    if r.ndim != 1:
        raise ValueError("returns must be 1-d")

    out = np.full(r.shape[0], np.nan, dtype=float)
    if r.shape[0] < window:
        return out
    # rolling variance via Welford-style two-pass on windows is fine at research scale
    for t in range(window - 1, r.shape[0]):
        w = r[t - window + 1 : t + 1]
        out[t] = float(np.var(w, ddof=1))
    return out


def corr_vs_true(estimate: np.ndarray, true_var: np.ndarray) -> float:
    """Pearson correlation between finite pairs of estimate and true variance."""
    e = np.asarray(estimate, dtype=float)
    t = np.asarray(true_var, dtype=float)
    mask = np.isfinite(e) & np.isfinite(t)
    if mask.sum() < 3:
        return float("nan")
    return float(np.corrcoef(e[mask], t[mask])[0, 1])
