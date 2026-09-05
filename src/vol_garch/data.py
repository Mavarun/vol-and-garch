"""Synthetic GARCH(1,1) return paths with known conditional variance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GarchParams:
    """GARCH(1,1): sigma2_t = omega + alpha * r2_{t-1} + beta * sigma2_{t-1}."""

    omega: float = 1e-6
    alpha: float = 0.08
    beta: float = 0.90
    mu: float = 0.0

    def __post_init__(self) -> None:
        if self.omega <= 0:
            raise ValueError("omega must be > 0")
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("alpha and beta must be >= 0")
        if self.alpha + self.beta >= 1.0:
            raise ValueError("alpha + beta must be < 1 for covariance stationarity")

    @property
    def unconditional_var(self) -> float:
        return self.omega / (1.0 - self.alpha - self.beta)


@dataclass(frozen=True)
class SyntheticGarchSeries:
    returns: np.ndarray
    sigma2: np.ndarray
    params: GarchParams
    seed: int


def simulate_garch(
    n: int = 2000,
    params: GarchParams | None = None,
    seed: int = 42,
    burn_in: int = 500,
) -> SyntheticGarchSeries:
    """Simulate Gaussian GARCH(1,1) returns; return path after burn-in."""
    if n < 10:
        raise ValueError("n must be >= 10")
    p = params or GarchParams()
    rng = np.random.default_rng(seed)

    total = n + burn_in
    eps = rng.standard_normal(total)
    sigma2 = np.empty(total, dtype=float)
    returns = np.empty(total, dtype=float)

    sigma2[0] = p.unconditional_var
    returns[0] = p.mu + np.sqrt(sigma2[0]) * eps[0]

    for t in range(1, total):
        sigma2[t] = p.omega + p.alpha * (returns[t - 1] - p.mu) ** 2 + p.beta * sigma2[t - 1]
        returns[t] = p.mu + np.sqrt(sigma2[t]) * eps[t]

    return SyntheticGarchSeries(
        returns=returns[burn_in:].copy(),
        sigma2=sigma2[burn_in:].copy(),
        params=p,
        seed=seed,
    )
