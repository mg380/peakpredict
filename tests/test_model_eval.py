import numpy as np
import pandas as pd

from peakpredict.pipeline.aggregates import build_population_aggregates, build_similarity_index
from peakpredict.pipeline.evaluate import temporal_evaluate
from peakpredict.pipeline.indicators import compute_indicators
from peakpredict.pipeline.model import NUMERIC, GroupMeanBaseline, PooledRidge


def make_features(n=40, seed=0, signal=False):
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(1, n + 1):
        cbs = float(rng.normal(1.0, 0.4))
        peak = 20.0 + 2.0 * cbs if signal else 24.0 + float(rng.normal(0, 1.5))
        for k in (3, 5):
            rows.append(
                {
                    "pid": pid, "event_id": "70", "sex": 2, "cutoff_k": k, "cutoff_age": 17.0 + k,
                    "n_seasons": k, "debut_age": 17.0, "debut_score": float(rng.normal()),
                    "current_age": 17.0 + k, "current_best_score": cbs,
                    "span_observed": float(k), "progression_rate": float(rng.normal(0.2, 0.1)),
                    "recent_slope": float(rng.normal(0.1, 0.1)), "mean_score": float(rng.normal()),
                    "score_std": abs(float(rng.normal(0, 0.2))), "peak_age": peak,
                }
            )
    return pd.DataFrame(rows)


def test_baseline_predicts_group_mean():
    df = make_features()
    base = GroupMeanBaseline().fit(df)
    pred, lo, hi = base.predict_one({}, "70", 2)
    assert lo < pred < hi
    # single (event, sex) group -> prediction is that group's mean peak age
    assert abs(pred - df["peak_age"].mean()) < 1e-6


def test_ridge_fits_and_predicts():
    df = make_features()
    ridge = PooledRidge().fit(df)
    feats = {k: df.iloc[0][k] for k in NUMERIC}
    pred, lo, hi = ridge.predict_one(feats, "70", 2)
    assert np.isfinite(pred) and lo < pred < hi


def test_temporal_evaluate_reports_models_without_leakage():
    df = make_features(40)
    rep = temporal_evaluate(df, n_splits=4)
    assert {"baseline", "ridge"} <= set(rep)
    assert rep["baseline"]["mae"] > 0
    assert "skill_vs_baseline" in rep["ridge"]
    # leakage assertion lives inside temporal_evaluate; reaching here means it held


def test_indicators_detect_strong_signal():
    df = make_features(60, signal=True)
    rep = compute_indicators(df)
    top = rep["indicators"][0]
    assert top["feature"] == "current_best_score"
    assert abs(top["pearson_r"]) > 0.8


def test_aggregates_percentiles_are_ordered():
    rng = np.random.default_rng(1)
    sb = pd.DataFrame(
        {
            "event_id": "70", "sex": 2,
            "age": rng.integers(18, 30, 300).astype(float),
            "score": rng.normal(0, 1, 300),
        }
    )
    agg = build_population_aggregates(sb)
    assert (agg["p10"] <= agg["p50"]).all()
    assert (agg["p50"] <= agg["p90"]).all()


def test_similarity_index_one_row_per_athlete():
    df = make_features(10)
    idx = build_similarity_index(df)
    assert len(idx) == df[["pid", "event_id", "sex"]].drop_duplicates().shape[0]
    assert (idx["cutoff_k"] == 5).all()  # largest cutoff kept
