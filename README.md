# vol-and-garch

Research slice: **realized volatility** and a simple **GARCH(1,1)** one-step variance forecast vs next-day squared returns (MSE / QLIKE).

This does **not** claim live PnL or trading alpha.

## Hypothesis

1. On synthetic Gaussian GARCH data, trailing realized variance (mean of squared returns) **tracks true conditional variance**.
2. A simple GARCH(1,1) one-step forecast of next-day $r_t^2$ **beats** a rolling sample-variance baseline on **MSE** and **QLIKE**.
3. Under **walk-forward OOS** refits, that edge **shrinks** relative to in-sample filtered scores — report both honestly.

## Method

- **Data:** in-process synthetic Gaussian GARCH(1,1) with known $\sigma_t^2$ (`seed=42` default). True DGP: $\sigma_t^2=\omega+\alpha r_{t-1}^2+\beta\sigma_{t-1}^2$.
- **Realized vol:** trailing mean of squared returns (`window=21`) as a daily realized-variance proxy; correlate / MSE vs true $\sigma_t^2$.
- **Baseline forecast:** rolling sample variance (`ddof=1`, same window), used as one-step variance forecast.
- **GARCH(1,1):** Gaussian MLE via `scipy.optimize` (soft-constrained $\omega>0$, $\alpha,\beta>0$, $\alpha+\beta<1$); filtered $\sigma_t^2$ is the one-step forecast of $r_t^2$.
- **OOS:** walk-forward folds — fit on prefix `[0, train)`, forecast `test_size` steps with recursive updates (refit each fold), step=`test_size`.
- **Losses:** MSE and QLIKE $\log(f)+y/f$ (Patton-style); lower is better. Target $y=r^2$.

## Defaults

| Knob | Default |
|------|---------|
| DGP | $\omega=10^{-6}$, $\alpha=0.08$, $\beta=0.90$, `n=2000`, `seed=42` |
| realized / roll window | 21 |
| walk-forward | `train_size=500`, `test_size=100`, `step=100` |

## Metrics (local synthetic run)

Command: `python scripts/run_garch_slice.py`  
Settings: defaults above.

### Hypothesis 1 — realized vs true $\sigma^2$

| metric | value |
| --- | --- |
| corr(realized, true $\sigma^2$) | 0.9474 |
| mse(realized, true $\sigma^2$) | 1.221e-10 |
| n overlap | 1980 |

### Forecast losses (target $r^2$)

| model | split | mse | qlike | n |
| --- | --- | --- | --- | --- |
| rolling_std | IS | 6.443e-09 | -8.89376 | 1979 |
| garch11 | IS | 6.367e-09 | -8.92489 | 2000 |
| rolling_std | OOS | 7.019e-09 | -8.88603 | 1500 |
| garch11 | OOS | 6.881e-09 | -8.90478 | 1500 |

### Edges (rolling_std - garch11; >0 means GARCH better)

| edge | value |
| --- | --- |
| IS MSE | 7.64e-11 |
| IS QLIKE | 0.03113 |
| OOS MSE | 1.38e-10 |
| OOS QLIKE | 0.01875 |

**Reading:** (1) Realized variance co-moves strongly with true $\sigma^2$ (corr ~ 0.95). (2) GARCH wins IS on both MSE and QLIKE. (3) The **QLIKE** edge shrinks OOS (0.031 -> 0.019); MSE edges are tiny either way — do not over-read absolute MSE on this scale. Not live PnL.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pytest
python scripts/run_garch_slice.py
python scripts/run_garch_slice.py --json --out artifacts/metrics.json
```

## Package layout

```
src/vol_garch/   data, realized, garch, metrics, evaluate
scripts/         run_garch_slice.py
tests/           realized tracking, GARCH fit, walk-forward eval
```

## Limits / why it can fail

- DGP **is** GARCH(1,1) — favors the parametric model; real equity returns have jumps, leverage, and heavier tails.
- Daily realized variance without intraday data is a noisy proxy; window choice matters.
- MLE can be flat / fragile on short folds; no Student-t, EGARCH, or GJR.
- Walk-forward refits once per fold (not every bar) for speed — slightly optimistic vs continuous re-estimation cost, slightly pessimistic vs perfect filtering.
- Losses use $r^2$ as a noisy proxy for latent variance; QLIKE is more informative than raw MSE here.
- **Not** production risk systems; no portfolio, options, or live market data path in this slice.
