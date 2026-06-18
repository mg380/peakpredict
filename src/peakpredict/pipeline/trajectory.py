"""B3 — fit a performance-vs-age trajectory and derive the peak.

Scores are higher-is-better, so a career trajectory is an inverted-U; the peak
is the parabola's maximum (requires a < 0). The peak window is the age interval
whose fitted score is within ``tau`` of the peak. Validated in the spike: Bolt's
100m trajectory vertex ~25.3y.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DEFAULT_TAU = 0.25  # score units (z) defining the near-peak window
MIN_FIT_POINTS = 3


@dataclass
class TrajectoryFit:
    a: float
    b: float
    c: float
    n_points: int
    span_years: float
    age_min: float
    age_max: float
    has_interior_max: bool
    peak_age: float | None = None
    peak_score: float | None = None
    window_lo: float | None = None
    window_hi: float | None = None


def fit_trajectory(ages, scores, tau: float = DEFAULT_TAU) -> TrajectoryFit | None:
    """Fit ``score = a*age^2 + b*age + c``; return None if fewer than 3 points.

    ``has_interior_max`` is True only when the parabola opens downward (a < 0)
    AND its vertex lies strictly within the observed age range (peak observed,
    not extrapolated).
    """
    ages = np.asarray(ages, dtype=float)
    scores = np.asarray(scores, dtype=float)
    n = len(ages)
    if n < MIN_FIT_POINTS:
        return None
    age_min, age_max = float(ages.min()), float(ages.max())
    a, b, c = (float(x) for x in np.polyfit(ages, scores, 2))
    base = TrajectoryFit(
        a=a, b=b, c=c, n_points=n, span_years=age_max - age_min,
        age_min=age_min, age_max=age_max, has_interior_max=False,
    )
    if a >= 0:  # opens upward -> no performance peak
        return base
    vertex = -b / (2 * a)
    base.peak_age = vertex
    base.peak_score = a * vertex * vertex + b * vertex + c
    # near-peak window = ages whose fitted score is within tau of the peak. A
    # nearly-flat parabola (|a| -> 0) makes this half-width explode, so clamp it
    # to the observed range: we can't assert "near peak" at ages we never saw.
    half = math.sqrt(tau / (-a))
    base.window_lo = max(age_min, vertex - half)
    base.window_hi = min(age_max, vertex + half)
    base.has_interior_max = age_min < vertex < age_max
    return base
