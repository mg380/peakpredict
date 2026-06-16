"""End-to-end integration: raw store -> build -> publish -> dashboard predict.

Exercises the real code path across all three components with no network, using
a small synthetic raw store, so the A->B->C data contracts are verified together.
"""

from datetime import date

import numpy as np

from peakpredict.common.io import connect, init_raw_store
from peakpredict.common.schemas import UploadedAthlete, UploadedResult
from peakpredict.dashboard.service import check_compatible, load_bundle, predict_uploaded
from peakpredict.pipeline.build import build_processed
from peakpredict.pipeline.publish import publish
from peakpredict.scraper.runner import upsert_athletes, upsert_performances


def _seed_raw(con, n_athletes=12, seed=0):
    """Insert athletes whose 400m careers each have a clear interior peak."""
    rng = np.random.default_rng(seed)
    athletes, perfs = [], []
    birth_year = 1990
    for pid in range(1, n_athletes + 1):
        peak_age = 24.0 + rng.normal(0, 1.5)
        athletes.append(
            {"pid": pid, "name": f"Athlete {pid}", "country": "USA", "sex": 2,
             "dob": date(birth_year, 1, 1), "url": f"u{pid}"}
        )
        for age in range(18, 31):
            mark = 50.0 + 0.05 * (age - peak_age) ** 2 + rng.normal(0, 0.1)  # min near peak
            perfs.append(
                {"pid": pid, "event_id": "70", "indoor": False,
                 "perf_date": date(birth_year + age, 6, 1), "mark_raw": f"{mark:.2f}",
                 "mark": float(mark), "wind": None, "record_flag": None,
                 "round_pos": "1", "competition": "GP", "location": "X"}
            )
    upsert_athletes(con, athletes)
    upsert_performances(con, perfs)


def test_end_to_end_raw_to_prediction(tmp_path):
    db = tmp_path / "store.duckdb"
    con = connect(db)
    init_raw_store(con)
    _seed_raw(con)
    con.close()

    # B (data): raw -> processed
    summary = build_processed(db, tmp_path / "processed")
    assert summary["season_bests"] > 0
    assert summary["labels"] > 0
    assert summary["features"] > 0

    # B (model): processed -> versioned artifact bundle
    out, primary, report = publish(tmp_path / "processed", tmp_path / "artifacts", version="ve2e")
    assert primary in ("ridge", "baseline")
    assert report["baseline"]["mae"] >= 0

    # C: dashboard loads the bundle and predicts a developing athlete
    art = load_bundle(out)
    check_compatible(art)
    assert art.manifest["version"] == "ve2e"

    results = [UploadedResult(age=a, mark=50.5 + 0.05 * (a - 24) ** 2) for a in (18, 19, 20, 21)]
    pred, series = predict_uploaded(art, UploadedAthlete(sex=2, event_id="70", results=results))
    assert len(series) == 4
    assert pred.confidence in ("ok", "low", "out_of_distribution")
    if pred.confidence == "ok":
        assert 14.0 < pred.peak_age < 42.0
