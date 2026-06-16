import pandas as pd
import pytest

from peakpredict.common.normalization import WorldAthleticsNormalizer, ZScoreNormalizer


def _fitted():
    df = pd.DataFrame(
        {
            "event_id": ["40"] * 5,
            "sex": [1] * 5,
            "mark": [10.0, 10.1, 10.2, 9.9, 9.8],
        }
    )
    return ZScoreNormalizer().fit(df)


def test_faster_time_scores_higher():
    n = _fitted()
    # 100m: lower mark is better -> must map to a HIGHER score
    assert n.transform(9.8, "40", 1) > n.transform(10.2, "40", 1)


def test_inverse_round_trip():
    n = _fitted()
    score = n.transform(10.0, "40", 1)
    assert n.inverse(score, "40", 1) == pytest.approx(10.0, abs=1e-9)


def test_missing_params_raise():
    n = _fitted()
    with pytest.raises(KeyError):
        n.transform(20.0, "50", 1)  # event 50 not fitted


def test_persistence_round_trip():
    n = _fitted()
    restored = ZScoreNormalizer.from_dict(n.to_dict())
    assert restored.transform(9.8, "40", 1) == pytest.approx(n.transform(9.8, "40", 1))


def test_fit_requires_columns():
    with pytest.raises(ValueError):
        ZScoreNormalizer().fit(pd.DataFrame({"event_id": ["40"], "mark": [10.0]}))


def test_wa_normalizer_is_interface_stub():
    with pytest.raises(NotImplementedError):
        WorldAthleticsNormalizer().transform(9.8, "40", 1)
