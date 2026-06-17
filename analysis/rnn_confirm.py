"""Confirm candidate configs vs the default over N seeds (Comet-logged).

Used to verify a sweep winner robustly (more seeds than the sweep itself).

Run: python analysis/rnn_confirm.py [--seeds 5]
"""

from __future__ import annotations

import argparse

import comet_ml  # noqa: F401 - before torch
import numpy as np
from rnn_experiment import _split, build_base_samples, start_experiment, train_rnn

CONFIGS = {"default": {}, "bidirectional": {"bidirectional": True}}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="confirm RNN configs over N seeds")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    base = build_base_samples("data/processed")
    pids = sorted({s["pid"] for s in base})
    out = {}
    for name, cfg in CONFIGS.items():
        tests, vals = [], []
        for i in range(args.seeds):
            seed = args.seed + i
            tr, va, te = _split(pids, seed)
            exp = start_experiment({**cfg, "seed": seed, "config": name}, tags=["confirm", name])
            t, v = train_rnn(base, tr, va, te, seed, {**cfg, "epochs": 200}, comet=exp)
            if exp is not None:
                exp.log_metric("test_mae", t)
                exp.log_metric("best_val_mae", v)
                exp.end()
            tests.append(t)
            vals.append(v)
        te_m, te_s, va_m = float(np.mean(tests)), float(np.std(tests)), float(np.mean(vals))
        out[name] = (te_m, te_s, va_m)
        print(f"  {name:14} test {te_m:.3f} +/- {te_s:.3f}  (val {va_m:.3f})", flush=True)
    delta = out["default"][0] - out["bidirectional"][0]
    print(f"\nbidirectional vs default: {delta:+.3f}y over {args.seeds} seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
