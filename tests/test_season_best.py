from datetime import date

import pandas as pd

from peakpredict.pipeline.season_best import build_season_bests


def _athletes():
    return pd.DataFrame(
        [
            {"pid": 1, "sex": 2, "dob": date(2000, 1, 1)},
            {"pid": 2, "sex": 2, "dob": None},  # no DOB -> dropped
        ]
    )


def _perf():
    return pd.DataFrame(
        [
            # pid 1, event 70 (400m), 2020 season: two marks, best (min) is 52.0
            {"pid": 1, "event_id": "70", "indoor": False, "perf_date": date(2020, 6, 1),
             "mark": 53.0, "wind": None},
            {"pid": 1, "event_id": "70", "indoor": False, "perf_date": date(2020, 7, 1),
             "mark": 52.0, "wind": None},
            # pid 1, event 40 (100m) wind-aided -> dropped
            {"pid": 1, "event_id": "40", "indoor": False, "perf_date": date(2020, 5, 1),
             "mark": 11.0, "wind": 3.5},
            # pid 1, event 40 legal
            {"pid": 1, "event_id": "40", "indoor": False, "perf_date": date(2021, 5, 1),
             "mark": 11.2, "wind": 1.0},
            # unsupported event -> dropped
            {"pid": 1, "event_id": "90", "indoor": False, "perf_date": date(2020, 5, 1),
             "mark": 120.0, "wind": None},
            # pid 2 dropped (no DOB)
            {"pid": 2, "event_id": "70", "indoor": False, "perf_date": date(2020, 6, 1),
             "mark": 51.0, "wind": None},
        ]
    )


def test_best_per_season_and_filters():
    sb = build_season_bests(_perf(), _athletes())
    # only pid 1, supported events, wind-legal
    assert set(sb["pid"]) == {1}
    assert set(sb["event_id"]) <= {"40", "50", "70"}
    # 400m 2020 best is the min mark
    row = sb[(sb["event_id"] == "70") & (sb["season"] == 2020)].iloc[0]
    assert row["mark"] == 52.0
    # wind-aided 100m (2020) dropped; legal 100m (2021) kept
    e40 = sb[sb["event_id"] == "40"]
    assert list(e40["season"]) == [2021]


def test_age_is_computed():
    sb = build_season_bests(_perf(), _athletes())
    row = sb[(sb["event_id"] == "70") & (sb["season"] == 2020)].iloc[0]
    assert 20.0 < row["age"] < 21.0  # born 2000, perf mid-2020
