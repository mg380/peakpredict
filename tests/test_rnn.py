import joblib
import numpy as np
import pandas as pd

from peakpredict.pipeline.rnn import RNNPredictor

# fast config so the unit test trains in well under a second
FAST = {"epochs": 6, "patience": 3, "hidden": 8, "head_dim": 8}


def make_data(n=60, seed=0):
    """Synthetic season_bests + leakage-safe features at cutoffs (3, 5)."""
    rng = np.random.default_rng(seed)
    sb_rows, feat_rows = [], []
    for pid in range(1, n + 1):
        peak = 24.0 + float(rng.normal(0, 1.5))
        ages = np.arange(17, 24, dtype=float)  # 7 seasons
        # quadratic-ish score peaking near `peak`
        scores = -0.03 * (ages - peak) ** 2 + rng.normal(0, 0.05, len(ages))
        for a, sc in zip(ages, scores, strict=False):
            sb_rows.append({"pid": pid, "event_id": "70", "sex": 2, "age": a,
                            "score": float(sc), "season": int(a), "mark": 24.0 - sc})
        for k in (3, 5):
            feat_rows.append({"pid": pid, "event_id": "70", "sex": 2, "cutoff_k": k,
                              "peak_age": peak,
                              "height_cm": float(rng.normal(180, 8)) if rng.random() > 0.3
                              else float("nan"),
                              "weight_kg": float(rng.normal(72, 8)) if rng.random() > 0.3
                              else float("nan")})
    return pd.DataFrame(feat_rows), pd.DataFrame(sb_rows)


def test_rnn_fits_and_predicts_series():
    feats, sb = make_data()
    rnn = RNNPredictor(cfg=FAST).fit(feats, sb, n_splits=3)
    series = sb[sb["pid"] == 1].sort_values("age")[["age", "score"]].head(5)
    pred, lo, hi = rnn.predict_series(series, "70", 2, height_cm=182.0, weight_kg=74.0)
    assert np.isfinite(pred) and lo < pred < hi
    assert 14.0 < pred < 42.0  # plausible peak age


def test_rnn_reports_held_out_metrics():
    feats, sb = make_data()
    rnn = RNNPredictor(cfg=FAST).fit(feats, sb, n_splits=3)
    assert rnn.cv_report is not None
    assert rnn.cv_report["mae"] > 0 and rnn.cv_report["n"] > 0
    assert 0.0 <= rnn.cv_report["interval_coverage"] <= 1.0


def test_rnn_predictor_is_picklable(tmp_path):
    feats, sb = make_data()
    rnn = RNNPredictor(cfg=FAST).fit(feats, sb, n_splits=3)
    path = tmp_path / "predictor.pkl"
    joblib.dump({"primary": "rnn", "model": rnn}, path)
    loaded = joblib.load(path)["model"]
    series = sb[sb["pid"] == 2].sort_values("age")[["age", "score"]].head(5)
    a = rnn.predict_series(series, "70", 2)
    b = loaded.predict_series(series, "70", 2)
    assert np.allclose(a, b)  # deterministic across the pickle round-trip


def test_rnn_predict_handles_missing_physical():
    feats, sb = make_data()
    rnn = RNNPredictor(cfg=FAST).fit(feats, sb, n_splits=3)
    series = sb[sb["pid"] == 3].sort_values("age")[["age", "score"]].head(4)
    pred, lo, hi = rnn.predict_series(series, "70", 2)  # no height/weight -> imputed
    assert np.isfinite(pred) and lo < pred < hi
