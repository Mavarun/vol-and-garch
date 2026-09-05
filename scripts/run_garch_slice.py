#!/usr/bin/env python3
"""Run realized-vol + GARCH(1,1) research slice; print metrics table / JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vol_garch.data import GarchParams
from vol_garch.evaluate import run_full_slice


def _fmt(x: float) -> str:
    if x != x:  # NaN
        return "nan"
    if abs(x) >= 1e-3:
        return f"{x:.6g}"
    return f"{x:.6e}"


def print_tables(report: dict) -> None:
    h1 = report["hypothesis_1_realized_tracks_true"]
    print("## Hypothesis 1 -- realized variance vs true sigma2")
    print(
        f"corr={_fmt(h1['corr_realized_vs_true_var'])}  "
        f"mse={_fmt(h1['mse_realized_vs_true_var'])}  "
        f"n={int(h1['n_overlap'])}"
    )
    print()
    print("## Forecast losses (target = next-day / same-index r^2 as documented)")
    headers = ["model", "split", "mse", "qlike", "n"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for r in report["forecast_rows"]:
        print(
            "| "
            + " | ".join(
                [
                    str(r["model"]),
                    str(r["split"]),
                    _fmt(r["mse"]),
                    _fmt(r["qlike"]),
                    str(int(r["n"])),
                ]
            )
            + " |"
        )
    print()
    print("## Edges (rolling_std loss - garch11 loss; >0 means GARCH better)")
    for k, v in report["edges"].items():
        print(f"{k}: {_fmt(v)}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window", type=int, default=21)
    p.add_argument("--train-size", type=int, default=500)
    p.add_argument("--test-size", type=int, default=100)
    p.add_argument("--step", type=int, default=100)
    p.add_argument("--omega", type=float, default=1e-6)
    p.add_argument("--alpha", type=float, default=0.08)
    p.add_argument("--beta", type=float, default=0.90)
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    params = GarchParams(omega=args.omega, alpha=args.alpha, beta=args.beta)
    report = run_full_slice(
        n=args.n,
        seed=args.seed,
        window=args.window,
        train_size=args.train_size,
        test_size=args.test_size,
        step=args.step,
        params=params,
    )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_tables(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
