"""Comet Optimizer — guided (Bayesian) search over the RNN architecture.

Comet's Optimizer suggests configs; each is trained on the SAME held-out split,
with mean validation MAE as the objective (minimize). Every trial streams to the
Comet dashboard, where you can watch the search and steer it. Requires
COMET_API_KEY / COMET_WORKSPACE / COMET_PROJECT in .secrets.

Run: python analysis/rnn_optimizer.py [--trials 15] [--seeds 2]
"""

from __future__ import annotations

import argparse
import os

import comet_ml
import numpy as np
from rnn_experiment import DEFAULT_CFG, _split, build_base_samples, train_rnn

from peakpredict.common.config import get_secret

SEARCH = {
    "algorithm": "bayes",
    "spec": {"metric": "val_mae", "objective": "minimize"},
    "parameters": {
        "hidden": {"type": "discrete", "values": [16, 24, 32, 48, 64]},
        "rnn_type": {"type": "categorical", "values": ["lstm", "gru"]},
        "num_layers": {"type": "discrete", "values": [1, 2]},
        "bidirectional": {"type": "discrete", "values": [0, 1]},
        "head_dim": {"type": "discrete", "values": [16, 32, 64]},
        "dropout": {"type": "float", "min": 0.1, "max": 0.5},
        "lr": {"type": "float", "min": 5e-4, "max": 5e-3, "scalingType": "loguniform"},
        "weight_decay": {"type": "float", "min": 1e-5, "max": 1e-2, "scalingType": "loguniform"},
    },
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Comet-driven RNN hyperparameter search")
    p.add_argument("--trials", type=int, default=15)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    # the Optimizer authenticates from the environment
    os.environ["COMET_API_KEY"] = get_secret("COMET_API_KEY")
    workspace = get_secret("COMET_WORKSPACE", required=False)
    project = get_secret("COMET_PROJECT", required=False) or "peakpredict-rnn"

    base = build_base_samples("data/processed")
    pids = sorted({s["pid"] for s in base})

    opt = comet_ml.Optimizer(SEARCH)
    best = (float("inf"), None)
    for n, exp in enumerate(opt.get_experiments(project_name=project, workspace=workspace), 1):
        if n > args.trials:
            exp.end()
            break
        exp.add_tag("optimizer")
        cfg = {
            "hidden": int(exp.get_parameter("hidden")),
            "rnn_type": exp.get_parameter("rnn_type"),
            "num_layers": int(exp.get_parameter("num_layers")),
            "bidirectional": bool(int(exp.get_parameter("bidirectional"))),
            "head_dim": int(exp.get_parameter("head_dim")),
            "dropout": float(exp.get_parameter("dropout")),
            "lr": float(exp.get_parameter("lr")),
            "weight_decay": float(exp.get_parameter("weight_decay")),
            "epochs": DEFAULT_CFG["epochs"],
        }
        vals, tests = [], []
        for i in range(args.seeds):
            tr, va, te = _split(pids, args.seed + i)
            t, v = train_rnn(base, tr, va, te, args.seed + i, cfg)
            vals.append(v)
            tests.append(t)
        mv, mt = float(np.mean(vals)), float(np.mean(tests))
        exp.log_metric("val_mae", mv)
        exp.log_metric("test_mae", mt)
        exp.end()
        best = min(best, (mv, {**cfg, "test_mae": round(mt, 3)}))
        print(f"trial {n:>2}: val {mv:.3f} test {mt:.3f} | "
              f"h={cfg['hidden']} {cfg['rnn_type']} bidir={cfg['bidirectional']} "
              f"drop={cfg['dropout']:.2f} lr={cfg['lr']:.1e}", flush=True)

    print(f"\nbest-by-val trial: val {best[0]:.3f} | {best[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
