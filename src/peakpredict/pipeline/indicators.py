"""B8 — indicators: which features correlate with peak age.

Correlates each engineered feature with the eventual peak age, using one row per
athlete (the largest available cutoff) to avoid pseudo-replication. Reports
Pearson r, p-value, and n — the analyst-facing multivariate signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import NUMERIC, TARGET

LITERATURE_NOTE = "Sprint peak age is ~23-26y in the literature (event/sex dependent)."


def _one_row_per_athlete(features: pd.DataFrame) -> pd.DataFrame:
    idx = features.groupby(["pid", "event_id", "sex"])["cutoff_k"].idxmax()
    return features.loc[idx]


def compute_indicators(features: pd.DataFrame) -> dict:
    """Return ranked feature-vs-peak-age correlations with statistical support."""
    from scipy.stats import pearsonr

    one = _one_row_per_athlete(features)
    out: list[dict] = []
    for feature in NUMERIC:
        x = one[feature].to_numpy(dtype=float)
        y = one[TARGET].to_numpy(dtype=float)
        if len(x) < 3 or np.std(x) == 0:
            continue
        r, p = pearsonr(x, y)
        out.append(
            {"feature": feature, "pearson_r": float(r), "p_value": float(p), "n": int(len(one))}
        )
    out.sort(key=lambda d: -abs(d["pearson_r"]))
    return {"target": TARGET, "indicators": out, "literature_note": LITERATURE_NOTE}
