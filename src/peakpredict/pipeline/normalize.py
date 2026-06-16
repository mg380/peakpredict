"""B1 — fit the shared normalizer and add a higher-is-better score column.

Uses ``common.normalization`` (the single shared implementation) so the score a
season-best gets here is identical to the score a dashboard upload will get.
"""

from __future__ import annotations

import pandas as pd

from ..common.logging import get_logger
from ..common.normalization import ZScoreNormalizer

log = get_logger("pipeline.normalize")


def fit_normalizer(season_bests: pd.DataFrame) -> ZScoreNormalizer:
    """Fit a ZScoreNormalizer on season-best marks (per event + sex)."""
    return ZScoreNormalizer().fit(season_bests[["event_id", "sex", "mark"]])


def add_scores(season_bests: pd.DataFrame, normalizer: ZScoreNormalizer) -> pd.DataFrame:
    """Return season_bests with a normalized, higher-is-better ``score`` column.

    Rows in an (event, sex) group the normalizer could not fit (a single
    season-best, or zero variance) cannot be scored and are dropped with a
    warning rather than crashing the build.
    """
    out = season_bests.copy()
    scores = []
    for m, e, s in zip(out["mark"], out["event_id"], out["sex"], strict=False):
        try:
            scores.append(normalizer.transform(m, e, int(s)))
        except KeyError:
            scores.append(float("nan"))
    out["score"] = scores
    dropped = int(out["score"].isna().sum())
    if dropped:
        log.warning("dropped %d season-best(s) with no normalization params", dropped)
    return out[out["score"].notna()].reset_index(drop=True)
