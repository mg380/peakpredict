"""Held-out forward-prediction validation.

Splits athletes so the test set is never used for training, then for each
held-out athlete truncates their career to the first k seasons, predicts the
peak age, and compares to their actual (full-career) peak. Reports accuracy for
three predictors so the trained model can be judged against simple baselines:

  B0  population mean   — "sprinters peak at ~25"
  B1  naive quadratic   — extrapolate the vertex from the first k points
  B2  Ridge (the model) — learned from early-career features

Run: python -m analysis.holdout_validation  [--processed data/processed] [--seed 42]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from peakpredict.common.io import read_parquet
from peakpredict.pipeline.model import NUMERIC, TARGET, GroupMeanBaseline, PooledRidge
from peakpredict.pipeline.trajectory import fit_trajectory

PLAUSIBLE = (12.0, 45.0)


def _naive_vertex(group: pd.DataFrame, k: int, fallback: float) -> float:
    """B1: fit a quadratic to the first k season-bests and take the vertex."""
    if group is None or len(group) < k:
        return fallback
    obs = group.iloc[:k]
    fit = fit_trajectory(obs["age"].to_numpy(), obs["score"].to_numpy())
    if fit is None or fit.peak_age is None:
        return fallback
    pred = float(fit.peak_age)
    return pred if PLAUSIBLE[0] <= pred <= PLAUSIBLE[1] else fallback


def run(processed: str = "data/processed", seed: int = 42, test_frac: float = 0.25) -> None:
    features = read_parquet(f"{processed}/features.parquet")
    season_bests = read_parquet(f"{processed}/season_bests.parquet")
    athletes = read_parquet(f"{processed}/athletes.parquet").set_index("pid")
    sb_groups = {
        key: g.sort_values("age")
        for key, g in season_bests.groupby(["pid", "event_id", "sex"])
    }

    # split by ATHLETE so a held-out athlete never appears in training
    rng = np.random.default_rng(seed)
    pids = features["pid"].unique()
    rng.shuffle(pids)
    n_test = int(len(pids) * test_frac)
    test_pids, train_pids = set(pids[:n_test]), set(pids[n_test:])
    train = features[features["pid"].isin(train_pids)]
    test = features[features["pid"].isin(test_pids)]

    base = GroupMeanBaseline().fit(train)
    ridge = PooledRidge().fit(train)

    rows = []
    for _, r in test.iterrows():
        feats = {k: r[k] for k in NUMERIC}
        p_base, _, _ = base.predict_one(feats, r["event_id"], int(r["sex"]))
        p_ridge, lo, hi = ridge.predict_one(feats, r["event_id"], int(r["sex"]))
        group = sb_groups.get((int(r["pid"]), r["event_id"], int(r["sex"])))
        p_naive = _naive_vertex(group, int(r["cutoff_k"]), fallback=p_base)
        rows.append({
            "pid": int(r["pid"]), "event_id": r["event_id"], "sex": int(r["sex"]),
            "cutoff_k": int(r["cutoff_k"]),
            "debut_age": r["debut_age"], "cutoff_age": r["cutoff_age"],
            "actual": float(r[TARGET]), "B0_pop": p_base, "B1_naive": p_naive,
            "B2_ridge": p_ridge, "lo": lo, "hi": hi,
        })
    res = pd.DataFrame(rows)
    for col in ("B0_pop", "B1_naive", "B2_ridge"):
        res[f"abs_{col}"] = (res[col] - res["actual"]).abs()

    print(f"\nHELD-OUT FORWARD-PREDICTION VALIDATION  (seed={seed})")
    print(f"test athletes: {res['pid'].nunique()}  |  test predictions: {len(res)}  "
          f"(train athletes: {len(train_pids)})\n")

    print(f"{'predictor':14}{'MAE':>7}{'RMSE':>7}{'bias':>7}{'≤1y':>7}{'≤2y':>7}{'≤3y':>7}")
    ladder = (("B0 pop-mean", "B0_pop"), ("B1 naive-quad", "B1_naive"), ("B2 ridge", "B2_ridge"))
    for name, col in ladder:
        err = res[col] - res["actual"]
        mae, rmse, bias = err.abs().mean(), np.sqrt((err**2).mean()), err.mean()
        within = [(err.abs() <= t).mean() for t in (1, 2, 3)]
        print(f"{name:14}{mae:7.2f}{rmse:7.2f}{bias:+7.2f}"
              f"{within[0]:7.0%}{within[1]:7.0%}{within[2]:7.0%}")

    cov = ((res["actual"] >= res["lo"]) & (res["actual"] <= res["hi"])).mean()
    print(f"\nridge 80% interval coverage: {cov:.0%}")
    print("ridge MAE by truncation horizon (fewer seasons = harder):")
    by_k = res.groupby("cutoff_k")["abs_B2_ridge"].mean()
    for k, m in by_k.items():
        print(f"  first {k} seasons -> MAE {m:.2f}y")

    print("\nexamples (held-out athletes, truncated to first 5 seasons):")
    n5 = int((res["cutoff_k"] == 5).sum())
    sample = res[res["cutoff_k"] == 5].sample(min(8, n5), random_state=seed)
    print(f"  {'athlete':<22}{'evt':>5}{'seen':>9}"
          f"{'pred peak [80% CI]':>22}{'actual':>9}{'err':>8}")
    for _, r in sample.iterrows():
        name = (
            str(athletes.loc[r["pid"], "name"])[:20]
            if r["pid"] in athletes.index else str(r["pid"])
        )
        seen = f"{r['debut_age']:.0f}-{r['cutoff_age']:.0f}"
        pred = f"{r['B2_ridge']:.1f} [{r['lo']:.0f}-{r['hi']:.0f}]"
        err = r["B2_ridge"] - r["actual"]
        print(f"  {name:<22}{r['event_id']:>5}{seen:>9}{pred:>22}{r['actual']:>9.1f}{err:>+8.1f}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="held-out forward-prediction validation")
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-frac", type=float, default=0.25)
    args = p.parse_args(argv)
    run(args.processed, args.seed, args.test_frac)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
