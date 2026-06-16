"""B4 — build training labels (descriptive peak) from complete-enough careers.

A career contributes a label only if it has enough season-bests over enough
years AND shows an interior performance maximum (observed both rising to and
declining from the peak). Athletes still ascending are excluded from labels —
they are the inference population, not the training population.
"""

from __future__ import annotations

import pandas as pd

from .trajectory import fit_trajectory

MIN_LABEL_POINTS = 5
MIN_LABEL_SPAN = 4.0


def build_labels(scored_season_bests: pd.DataFrame) -> pd.DataFrame:
    """Return one label row per qualifying (pid, event_id, sex) career.

    Columns: pid, event_id, sex, peak_age, peak_score, window_lo, window_hi,
    n_points, span_years.
    """
    rows: list[dict] = []
    for (pid, event_id, sex), g in scored_season_bests.groupby(["pid", "event_id", "sex"]):
        g = g.sort_values("age")
        fit = fit_trajectory(g["age"].to_numpy(), g["score"].to_numpy())
        if fit is None or not fit.has_interior_max:
            continue
        if fit.n_points < MIN_LABEL_POINTS or fit.span_years < MIN_LABEL_SPAN:
            continue
        rows.append(
            {
                "pid": int(pid),
                "event_id": event_id,
                "sex": int(sex),
                "peak_age": fit.peak_age,
                "peak_score": fit.peak_score,
                "window_lo": fit.window_lo,
                "window_hi": fit.window_hi,
                "n_points": fit.n_points,
                "span_years": fit.span_years,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "pid", "event_id", "sex", "peak_age", "peak_score",
            "window_lo", "window_hi", "n_points", "span_years",
        ],
    )
