from datetime import date

import pytest
from pydantic import ValidationError

from peakpredict.common.schemas import (
    PeakPrediction,
    PerformanceRow,
    UploadedAthlete,
    UploadedResult,
)


def test_uploaded_athlete_valid():
    a = UploadedAthlete(
        sex=1,
        event_id="40",
        results=[UploadedResult(date_or_age="2021", mark=10.1)],
    )
    assert a.event_id == "40"
    assert a.results[0].mark == 10.1


def test_uploaded_athlete_rejects_bad_sex():
    with pytest.raises(ValidationError):
        UploadedAthlete(
            sex=3, event_id="40", results=[UploadedResult(date_or_age="2021", mark=10.1)]
        )


def test_uploaded_athlete_rejects_unsupported_event():
    with pytest.raises(ValidationError):
        UploadedAthlete(
            sex=1, event_id="60", results=[UploadedResult(date_or_age="2021", mark=10.1)]
        )


def test_uploaded_result_requires_positive_mark():
    with pytest.raises(ValidationError):
        UploadedResult(date_or_age="2021", mark=0)


def test_uploaded_athlete_requires_at_least_one_result():
    with pytest.raises(ValidationError):
        UploadedAthlete(sex=1, event_id="40", results=[])


def test_peak_prediction_confidence_literal():
    p = PeakPrediction(
        peak_age=25.3, interval_lo=24.0, interval_hi=26.5,
        peak_score=2.1, window_lo=24.0, window_hi=27.0, confidence="ok",
    )
    assert p.confidence == "ok"
    with pytest.raises(ValidationError):
        PeakPrediction(
            peak_age=25.3, interval_lo=24.0, interval_hi=26.5,
            peak_score=2.1, window_lo=24.0, window_hi=27.0, confidence="maybe",
        )


def test_performance_row_parses_date():
    r = PerformanceRow(pid=45032, event_id="40", perf_date=date(2009, 8, 16), mark=9.58, wind=0.9)
    assert r.perf_date.year == 2009
    assert r.wind == 0.9
