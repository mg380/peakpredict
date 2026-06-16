"""B9 — reference-population aggregates and the similarity index.

Per (event, sex, integer-age) score percentiles give the dashboard its
population overlay (average trajectory + bands). The similarity index is one
feature row per athlete (largest cutoff) for the dashboard's "similar athletes".
"""

from __future__ import annotations

import pandas as pd


def _percentiles(s: pd.Series) -> pd.Series:
    return pd.Series(
        {
            "p10": s.quantile(0.10),
            "p25": s.quantile(0.25),
            "p50": s.median(),
            "p75": s.quantile(0.75),
            "p90": s.quantile(0.90),
            "mean": s.mean(),
            "count": int(s.count()),
        }
    )


def build_population_aggregates(scored_season_bests: pd.DataFrame) -> pd.DataFrame:
    """Per (event_id, sex, age_bin) score percentiles for population overlays."""
    df = scored_season_bests.copy()
    df["age_bin"] = df["age"].round().astype(int)
    grouped = df.groupby(["event_id", "sex", "age_bin"])["score"]
    agg = grouped.apply(_percentiles).unstack().reset_index()
    return agg.sort_values(["event_id", "sex", "age_bin"]).reset_index(drop=True)


def build_similarity_index(features: pd.DataFrame) -> pd.DataFrame:
    """One feature row per athlete (largest cutoff) for nearest-neighbour lookup."""
    idx = features.groupby(["pid", "event_id", "sex"])["cutoff_k"].idxmax()
    return features.loc[idx].reset_index(drop=True)
