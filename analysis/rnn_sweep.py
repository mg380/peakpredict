"""Hyperparameter sweep for the peak-age RNN, with Comet.ml logging.

Runs architecture/training configs through the SAME athlete-grouped held-out
split, ranks by VALIDATION MAE (selection never sees test), and reports held-out
test MAE. If COMET_API_KEY is set (in .secrets), every trial streams to Comet for
visualization; otherwise it runs locally.

Run: python analysis/rnn_sweep.py [--seeds 2]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from rnn_experiment import (
    DEFAULT_CFG,
    _split,
    build_base_samples,
    ridge_eval,
    start_experiment,
    train_rnn,
)

# curated search space — each dict overrides DEFAULT_CFG
GRID: list[dict] = [
    {},  # current default (lstm, hidden 24, 1 layer, dropout 0.3)
    {"rnn_type": "gru"},
    {"hidden": 16},
    {"hidden": 32},
    {"hidden": 48},
    {"hidden": 32, "rnn_type": "gru"},
    {"hidden": 32, "num_layers": 2},
    {"hidden": 24, "bidirectional": True},
    {"dropout": 0.2},
    {"dropout": 0.5},
    {"lr": 3e-3},
    {"weight_decay": 1e-3},
]


def _name(cfg: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in cfg.items()) or "default"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="RNN hyperparameter sweep (Comet-logged)")
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, default=2, help="seeds per config")
    args = p.parse_args(argv)

    base = build_base_samples(args.processed)
    pids = sorted({s["pid"] for s in base})
    rows = []
    for cfg in GRID:
        full = {**DEFAULT_CFG, **cfg}
        vals, tests = [], []
        for i in range(args.seeds):
            seed = args.seed + i
            tr, va, te = _split(pids, seed)
            exp = start_experiment({**full, "seed": seed}, tags=["sweep"])
            test, val = train_rnn(base, tr, va, te, seed, full, comet=exp)
            if exp is not None:
                exp.log_metric("test_mae", test)
                exp.log_metric("best_val_mae", val)
                exp.end()
            vals.append(val)
            tests.append(test)
        mv, mt = float(np.mean(vals)), float(np.mean(tests))
        rows.append({"config": _name(cfg), "val_mae": mv, "test_mae": mt})
        print(f"  done: {_name(cfg):38} val {mv:.3f}  test {mt:.3f}", flush=True)

    df = pd.DataFrame(rows).sort_values("val_mae").reset_index(drop=True)
    tr, va, te = _split(pids, args.seed)
    ref = ridge_eval(args.processed, te)["ridge"]
    print(f"\nRNN SWEEP ({args.seeds} seeds/config; ranked by val MAE) | ridge {ref:.2f}")
    print(f"  {'config':40}{'val MAE':>9}{'test MAE':>10}")
    for r in df.itertuples(index=False):
        print(f"  {r.config:40}{r.val_mae:>9.3f}{r.test_mae:>10.3f}")
    best = df.iloc[0]
    print(f"\nbest-by-val: {best['config']}  ->  test MAE {best['test_mae']:.3f}y "
          f"(default RNN ~1.53)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
