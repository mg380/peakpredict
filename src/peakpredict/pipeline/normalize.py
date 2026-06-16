"""B1 — fit the shared normalizer and add a higher-is-better score column.

Uses ``common.normalization`` (the single shared implementation) so the score a
season-best gets here is identical to the score a dashboard upload will get.
"""

from __future__ import annotations

import pandas as pd

from ..common.normalization import ZScoreNormalizer


def fit_normalizer(season_bests: pd.DataFrame) -> ZScoreNormalizer:
    """Fit a ZScoreNormalizer on season-best marks (per event + sex)."""
    return ZScoreNormalizer().fit(season_bests[["event_id", "sex", "mark"]])


def add_scores(season_bests: pd.DataFrame, normalizer: ZScoreNormalizer) -> pd.DataFrame:
    """Return season_bests with a normalized, higher-is-better ``score`` column."""
    out = season_bests.copy()
    out["score"] = [
        normalizer.transform(m, e, int(s))
        for m, e, s in zip(out["mark"], out["event_id"], out["sex"], strict=False)
    ]
    return out
