import json

import joblib
import numpy as np
import pandas as pd
from tests.test_model_eval import make_features

from peakpredict.common.normalization import ZScoreNormalizer
from peakpredict.pipeline.features import feature_schema
from peakpredict.pipeline.model import NUMERIC
from peakpredict.pipeline.publish import publish


def _make_processed(tmp_path):
    d = tmp_path / "processed"
    d.mkdir()
    rng = np.random.default_rng(2)
    sb = pd.DataFrame(
        {
            "pid": np.repeat(np.arange(1, 21), 6),
            "event_id": "70", "sex": 2,
            "season": np.tile(np.arange(2010, 2016), 20),
            "age": np.tile(np.arange(20, 26), 20).astype(float),
            "mark": rng.normal(52, 1.5, 120),
            "wind": None,
            "score": rng.normal(0, 1, 120),
        }
    )
    features = make_features(20)
    labels = pd.DataFrame(
        {"pid": np.arange(1, 21), "event_id": "70", "sex": 2, "peak_age": rng.normal(24, 1.5, 20)}
    )
    athletes = pd.DataFrame(
        {"pid": np.arange(1, 21), "name": [f"A{i}" for i in range(20)], "country": "USA", "sex": 2}
    )
    sb.to_parquet(d / "season_bests.parquet", index=False)
    features.to_parquet(d / "features.parquet", index=False)
    labels.to_parquet(d / "labels.parquet", index=False)
    athletes.to_parquet(d / "athletes.parquet", index=False)
    norm = ZScoreNormalizer().fit(sb[["event_id", "sex", "mark"]])
    (d / "normalization.json").write_text(json.dumps(norm.to_dict()))
    (d / "feature_schema.json").write_text(feature_schema().model_dump_json())
    return d


def test_publish_writes_complete_bundle(tmp_path):
    processed = _make_processed(tmp_path)
    out, primary, report = publish(processed, tmp_path / "artifacts", version="vtest")

    for fname in (
        "predictor.pkl", "manifest.json", "indicators.json", "validation.json",
        "aggregates.parquet", "similar_index.parquet", "normalization.json", "feature_schema.json",
    ):
        assert (out / fname).exists(), f"missing {fname}"

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["version"] == "vtest"
    assert manifest["events"] == ["40", "50", "70"]
    assert manifest["schema_version"] == "1"

    # the bundled predictor loads and produces a finite, bounded prediction
    bundle = joblib.load(out / "predictor.pkl")
    feats = {k: 1.0 for k in NUMERIC}
    pred, lo, hi = bundle["model"].predict_one(feats, "70", 2)
    assert np.isfinite(pred) and lo < pred < hi
    assert primary in ("ridge", "baseline")
