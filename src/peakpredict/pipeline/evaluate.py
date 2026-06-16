"""B7 — temporal, athlete-grouped forward-prediction evaluation.

Splits by athlete (no athlete in both train and test), trains each model on the
train fold, and predicts every held-out athlete's cutoff rows — simulating
predicting a developing athlete's peak from partial data. Reports MAE, bias,
prediction-interval coverage, and skill versus the population-mean baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import MODEL_FACTORIES, NUMERIC, TARGET


def temporal_evaluate(features: pd.DataFrame, n_splits: int = 5) -> dict:
    """Group-by-athlete CV report keyed by model name, plus skill vs baseline."""
    from sklearn.model_selection import GroupKFold

    n_groups = features["pid"].nunique()
    if n_groups < 2:
        return {"error": "not enough athletes to evaluate", "n_athletes": int(n_groups)}
    n_splits = max(2, min(n_splits, n_groups))

    collected: dict[str, list[tuple[float, float, float, float]]] = {n: [] for n in MODEL_FACTORIES}
    gkf = GroupKFold(n_splits)
    for tr, te in gkf.split(features, features[TARGET], features["pid"]):
        train, test = features.iloc[tr], features.iloc[te]
        assert not (set(train["pid"]) & set(test["pid"])), "athlete leakage across split"
        for name, Factory in MODEL_FACTORIES.items():
            # calibrate=True so the fold's prediction intervals match what production
            # ships (out-of-fold residual std), making reported coverage faithful
            model = Factory().fit(train, calibrate=True)
            for _, r in test.iterrows():
                feats = {k: r[k] for k in NUMERIC}
                pred, lo, hi = model.predict_one(feats, r["event_id"], int(r["sex"]))
                collected[name].append((float(r[TARGET]), pred, lo, hi))

    report: dict = {}
    for name, rows in collected.items():
        arr = np.array(rows)
        true, pred, lo, hi = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        report[name] = {
            "mae": float(np.abs(true - pred).mean()),
            "rmse": float(np.sqrt(((true - pred) ** 2).mean())),
            "bias": float((pred - true).mean()),
            "interval_coverage": float(((true >= lo) & (true <= hi)).mean()),
            "n": int(len(rows)),
        }
    base = report["baseline"]["mae"] or 1.0
    for name in report:
        report[name]["skill_vs_baseline"] = float((base - report[name]["mae"]) / base)
    report["n_athletes"] = int(n_groups)
    report["n_splits"] = int(n_splits)
    return report
