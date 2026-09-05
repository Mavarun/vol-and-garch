"""In-sample and walk-forward OOS evaluation of GARCH vs rolling-std forecasts."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from vol_garch.data import GarchParams, SyntheticGarchSeries, simulate_garch
from vol_garch.garch import fit_garch11, one_step_variance_forecast
from vol_garch.metrics import mse_loss, summarize_losses
from vol_garch.realized import corr_vs_true, rolling_realized_variance, rolling_std_variance


@dataclass
class ForecastEvalRow:
    model: str
    split: str
    mse: float
    qlike: float
    n: int


def _rolling_std_next_day_forecast(returns: np.ndarray, window: int) -> np.ndarray:
    """At t, forecast of r_{t+1}^2 is rolling sample variance ending at t; align to t+1."""
    rv = rolling_std_variance(returns, window=window)
    # forecast made at t applies to target at t+1 -> shift forward by 1 in output index
    out = np.full_like(rv, np.nan)
    out[1:] = rv[:-1]
    return out


def _garch_next_day_forecast_insample(returns: np.ndarray) -> tuple[np.ndarray, GarchParams]:
    """Fit once on full sample; one-step-ahead filtered forecasts (in-sample)."""
    fit = fit_garch11(returns)
    # filtered sigma2_t is Var(r_t | F_{t-1}); that is the forecast made at t-1 for time t
    forecast = fit.sigma2.copy()
    return forecast, fit.params


def evaluate_realized_tracking(
    series: SyntheticGarchSeries,
    window: int = 21,
) -> dict[str, float]:
    """Hypothesis (1): rolling realized variance tracks true sigma2."""
    rv = rolling_realized_variance(series.returns, window=window)
    corr = corr_vs_true(rv, series.sigma2)
    mse = mse_loss(rv, series.sigma2)
    return {
        "window": float(window),
        "corr_realized_vs_true_var": corr,
        "mse_realized_vs_true_var": mse,
        "n_overlap": float(np.sum(np.isfinite(rv))),
    }


def evaluate_insample_forecasts(
    returns: np.ndarray,
    window: int = 21,
) -> list[ForecastEvalRow]:
    """Hypothesis (2): GARCH beats rolling std on next-day r^2 (IS)."""
    target = returns ** 2
    roll_f = _rolling_std_next_day_forecast(returns, window=window)
    garch_f, _ = _garch_next_day_forecast_insample(returns)

    rows: list[ForecastEvalRow] = []
    for name, f in (("rolling_std", roll_f), ("garch11", garch_f)):
        s = summarize_losses(f, target)
        rows.append(
            ForecastEvalRow(
                model=name,
                split="IS",
                mse=s["mse"],
                qlike=s["qlike"],
                n=int(s["n"]),
            )
        )
    return rows


def walk_forward_forecasts(
    returns: np.ndarray,
    train_size: int = 500,
    test_size: int = 100,
    step: int = 100,
    window: int = 21,
    refit_every: int = 1,
) -> list[ForecastEvalRow]:
    """Walk-forward OOS: expanding train prefix, score on held-out r^2.

    ``refit_every``: reserved; default runner fits once per fold (large refit_every).
    """
    r = np.asarray(returns, dtype=float)
    n = r.shape[0]
    garch_f = np.full(n, np.nan)
    roll_f = np.full(n, np.nan)

    start = train_size
    while start + 1 <= n:
        end = min(start + test_size, n)
        # train on [0, start)
        train = r[:start]
        fit = fit_garch11(train)
        params = fit.params
        # seed recursion with filtered variance on train
        last_sigma2 = float(fit.sigma2[-1])
        last_resid = float(train[-1] - params.mu)

        # rolling std at train end
        if start >= window:
            roll_var = float(np.var(r[start - window : start], ddof=1))
        else:
            roll_var = float("nan")

        for t in range(start, end):
            # forecast for r_t^2 made with info through t-1
            garch_f[t] = one_step_variance_forecast(last_resid, last_sigma2, params)
            roll_f[t] = roll_var

            # update state with observed r_t
            resid_t = float(r[t] - params.mu)
            last_sigma2 = one_step_variance_forecast(last_resid, last_sigma2, params)
            # after observing, filtered variance at t equals the forecast we just made
            last_resid = resid_t
            if t + 1 >= window:
                roll_var = float(np.var(r[t + 1 - window : t + 1], ddof=1))

            # optional mid-fold refit hook (noop unless caller sets small refit_every)
            if refit_every > 0 and (t - start + 1) % refit_every == 0 and t + 1 < end:
                pass

        start += step

    target = r ** 2
    rows: list[ForecastEvalRow] = []
    for name, f in (("rolling_std", roll_f), ("garch11", garch_f)):
        s = summarize_losses(f, target)
        rows.append(
            ForecastEvalRow(
                model=name,
                split="OOS",
                mse=s["mse"],
                qlike=s["qlike"],
                n=int(s["n"]),
            )
        )
    return rows


def run_full_slice(
    n: int = 2000,
    seed: int = 42,
    window: int = 21,
    train_size: int = 500,
    test_size: int = 100,
    step: int = 100,
    params: GarchParams | None = None,
) -> dict:
    """Run hypotheses (1)-(3) and return a JSON-serializable report."""
    series = simulate_garch(n=n, params=params, seed=seed)
    tracking = evaluate_realized_tracking(series, window=window)
    is_rows = evaluate_insample_forecasts(series.returns, window=window)
    oos_rows = walk_forward_forecasts(
        series.returns,
        train_size=train_size,
        test_size=test_size,
        step=step,
        window=window,
        refit_every=10**9,  # fit once per fold
    )

    is_map = {r.model: r for r in is_rows}
    oos_map = {r.model: r for r in oos_rows}

    def edge(a: ForecastEvalRow, b: ForecastEvalRow, key: str) -> float:
        # positive => garch better (lower loss)
        return float(getattr(b, key) - getattr(a, key))

    report = {
        "config": {
            "n": n,
            "seed": seed,
            "window": window,
            "train_size": train_size,
            "test_size": test_size,
            "step": step,
            "true_params": asdict(series.params),
        },
        "hypothesis_1_realized_tracks_true": tracking,
        "forecast_rows": [asdict(r) for r in (is_rows + oos_rows)],
        "edges": {
            "IS_mse_rolling_minus_garch": edge(is_map["garch11"], is_map["rolling_std"], "mse"),
            "IS_qlike_rolling_minus_garch": edge(is_map["garch11"], is_map["rolling_std"], "qlike"),
            "OOS_mse_rolling_minus_garch": edge(oos_map["garch11"], oos_map["rolling_std"], "mse"),
            "OOS_qlike_rolling_minus_garch": edge(oos_map["garch11"], oos_map["rolling_std"], "qlike"),
        },
    }
    return report
