import numpy as np
import pytest

from peakpredict.pipeline.trajectory import fit_trajectory


def test_recovers_known_vertex():
    # inverted-U peaking at age 25
    ages = np.arange(20, 31)
    scores = -0.02 * (ages - 25.0) ** 2 + 3.0
    fit = fit_trajectory(ages, scores)
    assert fit is not None and fit.has_interior_max
    assert fit.peak_age == pytest.approx(25.0, abs=0.1)
    assert fit.a < 0
    assert fit.window_lo < fit.peak_age < fit.window_hi


def test_too_few_points_returns_none():
    assert fit_trajectory([20, 21], [1.0, 2.0]) is None


def test_monotonic_rising_has_no_interior_max():
    ages = np.arange(18, 24)
    scores = 0.3 * ages  # still improving, no peak observed
    fit = fit_trajectory(ages, scores)
    assert fit is not None
    assert not fit.has_interior_max


def test_peak_outside_range_not_interior():
    # vertex would sit beyond the observed ages
    ages = np.array([20.0, 21.0, 22.0])
    scores = np.array([1.0, 1.8, 2.4])  # rising, vertex (if any) beyond range
    fit = fit_trajectory(ages, scores)
    assert fit is not None and not fit.has_interior_max
