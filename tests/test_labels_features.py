import numpy as np
import pandas as pd

from peakpredict.pipeline.features import build_features, compute_features, feature_schema
from peakpredict.pipeline.labels import build_labels


def _career(pid, ages, scores, event_id="70", sex=2):
    return pd.DataFrame(
        {
            "pid": pid, "event_id": event_id, "sex": sex,
            "season": [2000 + int(a) for a in ages], "age": ages,
            "mark": [50.0] * len(ages), "wind": None, "score": scores,
        }
    )


def test_label_built_for_interior_peak():
    ages = np.arange(20, 31).astype(float)  # 11 seasons, span 10
    scores = -0.02 * (ages - 25.0) ** 2 + 3.0
    labels = build_labels(_career(1, ages, scores))
    assert len(labels) == 1
    assert 24.0 < labels.iloc[0]["peak_age"] < 26.0


def test_no_label_for_still_ascending_career():
    ages = np.arange(18, 24).astype(float)
    scores = 0.3 * ages  # monotone rising -> no interior max
    assert build_labels(_career(2, ages, scores)).empty


def test_no_label_when_too_few_points():
    ages = np.array([22.0, 25.0, 28.0])  # interior peak shape but < MIN_LABEL_POINTS
    scores = np.array([1.0, 3.0, 1.0])
    assert build_labels(_career(3, ages, scores)).empty


def test_features_are_leakage_safe():
    # later seasons have the HIGHER scores; a k=3 cutoff must not see them
    ages = np.arange(20, 31).astype(float)
    scores = -0.02 * (ages - 28.0) ** 2 + 3.0  # rising across most of the range
    sb = _career(1, ages, scores)
    labels = build_labels(sb)
    feats = build_features(sb, labels, cutoffs=(3, 5))
    assert not feats.empty
    k3 = feats[feats["cutoff_k"] == 3].iloc[0]
    first3_best = sb.sort_values("age")["score"].iloc[:3].max()
    assert k3["n_seasons"] == 3
    assert k3["current_best_score"] == first3_best  # not the later, higher scores
    assert k3["cutoff_age"] == sb.sort_values("age")["age"].iloc[2]


def test_compute_features_keys_match_schema():
    obs = pd.DataFrame({"age": [20.0, 21.0, 22.0], "score": [1.0, 1.5, 1.2]})
    feats = compute_features(obs)
    schema_fields = {f.name for f in feature_schema().fields}
    assert set(feats) == schema_fields
