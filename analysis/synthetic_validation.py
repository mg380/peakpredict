"""Synthetic mechanism check — decompose the held-out error.

The real held-out MAE (~1.7y) is measured against *descriptive* peak labels,
which are themselves noisy quadratic-fit vertices. This experiment generates
athletes whose true peak age is known exactly, but whose careers are simulated
with the SAME curvature, form-noise, span and debut/decline offsets as the real
data. Running them through the real pipeline separates two error sources:

  - labeling noise   : descriptive label (fit vertex) vs the true peak
  - model error      : the predictor's error against the true peak

Run: python -m analysis.synthetic_validation   (or: python analysis/synthetic_validation.py)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from peakpredict.common.io import read_parquet
from peakpredict.pipeline.features import build_features
from peakpredict.pipeline.labels import build_labels
from peakpredict.pipeline.model import NUMERIC, PooledRidge
from peakpredict.pipeline.trajectory import fit_trajectory


def estimate_params(processed: str) -> dict:
    """Calibrate the generator from real labelled careers."""
    sb = read_parquet(f"{processed}/season_bests.parquet")
    lab = read_parquet(f"{processed}/labels.parquet")
    groups = sb.groupby(["pid", "event_id", "sex"])
    cur, res, pre, dec, lvl = [], [], [], [], []
    for row in lab.itertuples(index=False):
        try:
            g = groups.get_group((row.pid, row.event_id, row.sex)).sort_values("age")
        except KeyError:
            continue
        ages, scores = g["age"].to_numpy(), g["score"].to_numpy()
        fit = fit_trajectory(ages, scores)
        if fit is None or fit.peak_age is None:
            continue
        fitted = fit.a * ages**2 + fit.b * ages + fit.c
        cur.append(fit.a)
        res.append(float(np.std(scores - fitted)))
        lvl.append(fit.peak_score)
        pre.append(fit.peak_age - ages.min())
        dec.append(ages.max() - fit.peak_age)
    return {
        "peak_mean": float(lab["peak_age"].mean()),
        "peak_sd": float(lab["peak_age"].std()),
        "cur": np.array(cur), "res": np.array(res), "lvl": np.array(lvl),
        "pre": np.array(pre), "dec": np.array(dec),
    }


def generate(p: dict, n: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate n athletes with known true peak age, calibrated to real data."""
    rng = np.random.default_rng(seed)
    pick = lambda arr: arr[rng.integers(len(arr))]  # noqa: E731 - bootstrap sample
    sb_rows, truth = [], []
    for pid in range(1, n + 1):
        true_peak = float(rng.normal(p["peak_mean"], p["peak_sd"]))
        a = -abs(float(pick(p["cur"])))
        level, sigma = float(pick(p["lvl"])), float(pick(p["res"]))
        pre, dec = float(pick(p["pre"])), float(pick(p["dec"]))
        ages = np.arange(int(round(true_peak - pre)), int(round(true_peak + dec)) + 1)
        if len(ages) < 5:
            continue
        scores = a * (ages - true_peak) ** 2 + level + rng.normal(0, sigma, len(ages))
        for age, sc in zip(ages, scores, strict=False):
            sb_rows.append({"pid": pid, "event_id": "40", "sex": 2, "season": int(age),
                            "age": float(age), "mark": np.nan, "score": float(sc)})
        truth.append({"pid": pid, "A_star": true_peak})
    return pd.DataFrame(sb_rows), pd.DataFrame(truth)


def _predict(model, df: pd.DataFrame) -> np.ndarray:
    preds = []
    for r in df.itertuples(index=False):
        feats = {k: getattr(r, k) for k in NUMERIC}
        preds.append(model.predict_one(feats, r.event_id, int(r.sex))[0])
    return np.array(preds)


def run(processed: str = "data/processed", n: int = 2500, seed: int = 7) -> None:
    params = estimate_params(processed)
    sb, truth = generate(params, n, seed)
    labels = build_labels(sb)  # descriptive peak (fit vertex) = Â
    lab = labels.merge(truth, on="pid")
    label_err = (lab["peak_age"] - lab["A_star"]).to_numpy()

    feats = build_features(sb, labels).merge(truth, on="pid")
    rng = np.random.default_rng(seed)
    pids = feats["pid"].unique()
    rng.shuffle(pids)
    test_pids = set(pids[: int(0.25 * len(pids))])
    train = feats[~feats["pid"].isin(test_pids)]
    test = feats[feats["pid"].isin(test_pids)]

    ridge_hat = PooledRidge().fit(train)                                  # trained on Â
    ridge_star = PooledRidge().fit(train.assign(peak_age=train["A_star"]))  # trained on truth
    pred_hat, pred_star = _predict(ridge_hat, test), _predict(ridge_star, test)
    a_true, a_hat = test["A_star"].to_numpy(), test["peak_age"].to_numpy()
    mae = lambda x, y: float(np.abs(x - y).mean())  # noqa: E731

    m_label = mae(lab["peak_age"], lab["A_star"])
    r_label = float(np.sqrt((label_err**2).mean()))
    m_noisy, m_true, m_floor = mae(pred_hat, a_hat), mae(pred_hat, a_true), mae(pred_star, a_true)

    print(f"\nSYNTHETIC MECHANISM CHECK  (calibrated to real data, seed={seed})")
    print(f"synthetic athletes labelled: {len(lab)}  |  test predictions: {len(test)}\n")
    print("error decomposition (years):")
    print(f"  labeling noise  label vs TRUE peak      MAE {m_label:.2f}  RMSE {r_label:.2f}")
    print(f"  model vs label  pred vs noisy label     MAE {m_noisy:.2f}  (~ real held-out)")
    print(f"  model vs TRUE   trained on noisy labels  MAE {m_true:.2f}")
    print(f"  model floor     trained on TRUE labels   MAE {m_floor:.2f}")


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="synthetic mechanism-check validation")
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--n", type=int, default=2500)
    p.add_argument("--seed", type=int, default=7)
    p.parse_args(argv)
    args = p.parse_args(argv)
    run(args.processed, args.n, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
