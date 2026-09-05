"""Tests for forecast evaluation and walk-forward bookkeeping."""

from __future__ import annotations

import numpy as np

from vol_garch.data import simulate_garch
from vol_garch.evaluate import (
    evaluate_insample_forecasts,
    evaluate_realized_tracking,
    run_full_slice,
    walk_forward_forecasts,
)
from vol_garch.metrics import mse_loss, qlike_loss


def test_qlike_finite_for_positive_forecasts():
    f = np.array([0.1, 0.2, 0.15])
    y = np.array([0.08, 0.25, 0.1])
    q = qlike_loss(f, y)
    assert np.isfinite(q)
    assert mse_loss(f, y) >= 0


def test_insample_garch_beats_or_matches_rolling_on_mse():
    series = simulate_garch(n=1200, seed=3)
    rows = {r.model: r for r in evaluate_insample_forecasts(series.returns, window=21)}
    # On DGP that is literally GARCH, filtered GARCH should not lose badly to rolling std
    assert rows["garch11"].mse <= rows["rolling_std"].mse * 1.05


def test_walk_forward_produces_oos_rows():
    series = simulate_garch(n=800, seed=5)
    rows = walk_forward_forecasts(
        series.returns, train_size=400, test_size=50, step=50, window=21
    )
    assert {r.split for r in rows} == {"OOS"}
    assert all(r.n > 0 for r in rows)


def test_run_full_slice_keys():
    report = run_full_slice(n=700, seed=2, train_size=300, test_size=50, step=50)
    assert "hypothesis_1_realized_tracks_true" in report
    assert "edges" in report
    assert report["hypothesis_1_realized_tracks_true"]["corr_realized_vs_true_var"] > 0.3
    tracking = evaluate_realized_tracking(simulate_garch(n=700, seed=2))
    assert tracking["corr_realized_vs_true_var"] > 0.3
