"""Tests for realized / rolling volatility helpers."""

from __future__ import annotations

import numpy as np

from vol_garch.data import GarchParams, simulate_garch
from vol_garch.realized import corr_vs_true, rolling_realized_variance, rolling_std_variance


def test_rolling_realized_variance_shape_and_warmup():
    r = np.arange(10, dtype=float)
    out = rolling_realized_variance(r, window=5)
    assert out.shape == (10,)
    assert np.all(np.isnan(out[:4]))
    assert np.isfinite(out[4:]).all()
    # manual check at index 4: mean of squares of [0,1,2,3,4]
    expected = np.mean(np.arange(5) ** 2)
    assert abs(out[4] - expected) < 1e-12


def test_realized_tracks_true_garch_variance():
    series = simulate_garch(n=1500, seed=7, params=GarchParams(omega=1e-6, alpha=0.1, beta=0.85))
    rv = rolling_realized_variance(series.returns, window=21)
    corr = corr_vs_true(rv, series.sigma2)
    assert corr > 0.4  # noisy but should clearly co-move


def test_rolling_std_positive():
    rng = np.random.default_rng(0)
    r = rng.standard_normal(100)
    v = rolling_std_variance(r, window=10)
    assert np.nanmin(v[9:]) > 0
