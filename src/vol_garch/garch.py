"""Simple Gaussian GARCH(1,1) MLE fit and one-step variance forecast."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from vol_garch.data import GarchParams


@dataclass(frozen=True)
class GarchFitResult:
    params: GarchParams
    sigma2: np.ndarray
    loglik: float
    success: bool
    message: str


def _neg_loglik(theta: np.ndarray, residuals: np.ndarray) -> float:
    """Negative Gaussian log-likelihood under GARCH(1,1) variance recursion."""
    omega, alpha, beta = theta
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
        return 1e12

    n = residuals.shape[0]
    var0 = float(np.mean(residuals ** 2))
    sigma2 = np.empty(n, dtype=float)
    sigma2[0] = max(var0, omega / max(1e-12, 1.0 - alpha - beta))

    for t in range(1, n):
        sigma2[t] = omega + alpha * residuals[t - 1] ** 2 + beta * sigma2[t - 1]
        if sigma2[t] <= 1e-18:
            return 1e12

    # 0.5 * sum(log(2π) + log(σ²) + r²/σ²)
    ll = 0.5 * np.sum(np.log(2.0 * np.pi) + np.log(sigma2) + residuals ** 2 / sigma2)
    if not np.isfinite(ll):
        return 1e12
    return float(ll)


def filter_variance(residuals: np.ndarray, params: GarchParams) -> np.ndarray:
    """Forward variance recursion given fixed parameters."""
    r = np.asarray(residuals, dtype=float)
    n = r.shape[0]
    sigma2 = np.empty(n, dtype=float)
    sigma2[0] = params.unconditional_var
    for t in range(1, n):
        sigma2[t] = params.omega + params.alpha * r[t - 1] ** 2 + params.beta * sigma2[t - 1]
    return sigma2


def fit_garch11(
    returns: np.ndarray,
    mu: float | None = None,
    x0: tuple[float, float, float] | None = None,
) -> GarchFitResult:
    """Fit Gaussian GARCH(1,1) by MLE (scipy L-BFGS-B on unconstrained map)."""
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or r.shape[0] < 50:
        raise ValueError("need at least 50 returns")

    mean = float(np.mean(r)) if mu is None else float(mu)
    resid = r - mean
    var = float(np.var(resid, ddof=1))
    if not np.isfinite(var) or var <= 0:
        raise ValueError("non-positive sample variance")

    # Parameterize: omega>0, alpha>0, beta>0, alpha+beta<1 via soft transforms
    # theta_raw = (log omega, logit-ish alpha, logit-ish beta fraction of remaining)
    def pack(omega: float, alpha: float, beta: float) -> np.ndarray:
        # inverse of softplus / stick-breaking-ish
        # use logs of positive pieces with slack
        eps = 1e-8
        a = np.clip(alpha, eps, 1 - 2 * eps)
        b = np.clip(beta, eps, 1 - a - eps)
        # map to R^3
        return np.array(
            [
                np.log(max(omega, eps)),
                np.log(a / max(eps, 1.0 - a - b)),
                np.log(b / max(eps, 1.0 - a - b)),
            ],
            dtype=float,
        )

    def unpack(z: np.ndarray) -> np.ndarray:
        omega = float(np.exp(z[0]))
        e1, e2 = np.exp(z[1]), np.exp(z[2])
        s = 1.0 + e1 + e2
        alpha = float(e1 / s)
        beta = float(e2 / s)
        # leave slack so alpha+beta < 1
        return np.array([omega, alpha, beta], dtype=float)

    if x0 is None:
        # method-of-moments-ish start: persistence ~0.95, alpha small
        alpha0, beta0 = 0.05, 0.90
        omega0 = var * (1.0 - alpha0 - beta0)
        z0 = pack(omega0, alpha0, beta0)
    else:
        z0 = pack(*x0)

    def objective(z: np.ndarray) -> float:
        return _neg_loglik(unpack(z), resid)

    res = minimize(objective, z0, method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-10})
    omega, alpha, beta = unpack(res.x)
    params = GarchParams(omega=omega, alpha=alpha, beta=beta, mu=mean)
    sigma2 = filter_variance(resid, params)
    nll = _neg_loglik(np.array([omega, alpha, beta]), resid)
    return GarchFitResult(
        params=params,
        sigma2=sigma2,
        loglik=-nll,
        success=bool(res.success),
        message=str(res.message),
    )


def one_step_variance_forecast(last_resid: float, last_sigma2: float, params: GarchParams) -> float:
    """σ²_{t+1|t} = ω + α r_t² + β σ²_t."""
    return float(params.omega + params.alpha * last_resid ** 2 + params.beta * last_sigma2)
