"""Tests for GARCH(1,1) fit and forecast recursion."""

from __future__ import annotations

import numpy as np

from vol_garch.data import GarchParams, simulate_garch
from vol_garch.garch import filter_variance, fit_garch11, one_step_variance_forecast


def test_filter_matches_one_step_recursion():
    series = simulate_garch(n=200, seed=1)
    params = series.params
    resid = series.returns - params.mu
    sigma2 = filter_variance(resid, params)
    # rebuild step by step
    s = params.unconditional_var
    for t in range(1, 50):
        s_next = one_step_variance_forecast(resid[t - 1], s, params)
        assert abs(s_next - sigma2[t]) < 1e-12
        s = s_next


def test_fit_recovers_persistence_roughly():
    true = GarchParams(omega=2e-6, alpha=0.07, beta=0.90)
    series = simulate_garch(n=3000, seed=42, params=true)
    fit = fit_garch11(series.returns)
    assert fit.success or fit.loglik < 0  # allow optimizer quirks; params should be sane
    assert fit.params.omega > 0
    assert 0 < fit.params.alpha < 0.4
    assert 0.5 < fit.params.beta < 0.99
    assert fit.params.alpha + fit.params.beta < 1.0
    # persistence close-ish on long sample
    true_pers = true.alpha + true.beta
    fit_pers = fit.params.alpha + fit.params.beta
    assert abs(fit_pers - true_pers) < 0.15


def test_one_step_forecast_positive():
    p = GarchParams()
    f = one_step_variance_forecast(0.01, p.unconditional_var, p)
    assert f > 0
