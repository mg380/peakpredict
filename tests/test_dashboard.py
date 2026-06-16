import plotly.graph_objects as go
import pytest
from tests.test_publish import _make_processed

from peakpredict.common.schemas import UploadedAthlete, UploadedResult
from peakpredict.dashboard.charting import hero_chart
from peakpredict.dashboard.service import (
    IncompatibleArtifactError,
    athlete_series,
    check_compatible,
    load_bundle,
    population_overlay,
    predict_uploaded,
    upload_to_series,
)
from peakpredict.pipeline.publish import publish


@pytest.fixture
def bundle(tmp_path):
    processed = _make_processed(tmp_path)
    out, _, _ = publish(processed, tmp_path / "artifacts", version="vtest")
    return load_bundle(out)


def test_bundle_loads_and_is_compatible(bundle):
    check_compatible(bundle)
    assert bundle.manifest["version"] == "vtest"


def test_incompatible_bundle_is_refused(bundle):
    bundle.feature_schema["schema_version"] = "999"
    with pytest.raises(IncompatibleArtifactError):
        check_compatible(bundle)


def test_predict_uploaded_ok(bundle):
    marks = [(19, 53.5), (20, 52.8), (21, 52.0), (22, 52.3)]
    results = [UploadedResult(age=a, mark=m) for a, m in marks]
    pred, series = predict_uploaded(bundle, UploadedAthlete(sex=2, event_id="70", results=results))
    assert len(series) == 4
    assert pred.confidence in ("ok", "low", "out_of_distribution")


def test_predict_insufficient(bundle):
    results = [UploadedResult(age=19, mark=53.0), UploadedResult(age=20, mark=52.0)]
    pred, _ = predict_uploaded(bundle, UploadedAthlete(sex=2, event_id="70", results=results))
    assert pred.confidence == "insufficient"


def test_predict_event_absent_from_bundle_flagged(bundle):
    # event 40 is a valid v1 event but the synthetic bundle only has 70 data
    results = [UploadedResult(age=a, mark=10.0 + a / 100) for a in (18, 19, 20)]
    pred, _ = predict_uploaded(bundle, UploadedAthlete(sex=2, event_id="40", results=results))
    assert pred.confidence == "unsupported_event"


def test_all_wind_aided_is_insufficient_not_unsupported(bundle):
    # supported event (70) but every result is wind-aided -> dropped -> insufficient
    results = [UploadedResult(age=a, mark=52.0, wind=3.5) for a in (19, 20, 21, 22)]
    pred, series = predict_uploaded(bundle, UploadedAthlete(sex=2, event_id="70", results=results))
    assert pred.confidence == "insufficient"
    assert series.empty


def test_window_undefined_without_interior_peak(bundle):
    import math

    # monotonically improving (decreasing 400m times) -> no interior max -> no peak window
    marks = [(18, 54.0), (19, 53.0), (20, 52.0), (21, 51.0)]
    results = [UploadedResult(age=a, mark=m) for a, m in marks]
    pred, _ = predict_uploaded(bundle, UploadedAthlete(sex=2, event_id="70", results=results))
    if pred.confidence in ("ok", "low", "out_of_distribution"):
        assert math.isnan(pred.window_lo) and math.isnan(pred.window_hi)


def test_normalization_parity(bundle):
    athlete = UploadedAthlete(sex=2, event_id="70", results=[UploadedResult(age=20, mark=52.0)])
    series = upload_to_series(bundle, athlete)
    expected = bundle.normalizer.transform(52.0, "70", 2)
    assert series["score"].iloc[0] == pytest.approx(expected)


def test_explore_series_and_overlay(bundle):
    assert not athlete_series(bundle, 1, "70", 2).empty
    assert not population_overlay(bundle, "70", 2).empty


def test_hero_chart_builds(bundle):
    series = athlete_series(bundle, 1, "70", 2)
    fig = hero_chart(series, None, population_overlay(bundle, "70", 2), title="t")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
