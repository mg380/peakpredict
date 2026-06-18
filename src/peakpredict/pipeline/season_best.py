"""B2 — reduce raw performances to per-season bests for the v1 events.

Filters to supported events, drops wind-aided marks (> +2.0 m/s), joins DOB to
compute age, and keeps the single best (direction-aware) mark per athlete /
event / season. Season-bests are the clean signal the trajectory fit uses.
"""

from __future__ import annotations

import pandas as pd

from ..common.event_maps import SUPPORTED_V1_EVENTS, is_lower_better

LEGAL_WIND_MAX = 2.0
# Plausible competitive age. Marks computed outside this band come from a bad
# dob or perf_date (the source has stray far-future/typo dates, e.g. a 1950s
# athlete with a 2022 record). Such rows wreck the trajectory fit, so drop them.
AGE_BOUNDS = (8.0, 55.0)


def build_season_bests(perf: pd.DataFrame, athletes: pd.DataFrame) -> pd.DataFrame:
    """Return per-(pid, event_id, sex, season) best marks with age and raw mark.

    Columns: pid, event_id, sex, season, age, mark, wind.
    """
    df = perf[perf["event_id"].isin(SUPPORTED_V1_EVENTS)].copy()
    df = df[df["mark"].notna() & df["perf_date"].notna()]
    # wind-legal: keep rows with no wind recorded (e.g. 400m) or wind <= +2.0
    df = df[df["wind"].isna() | (df["wind"] <= LEGAL_WIND_MAX)]

    ath = athletes[["pid", "sex", "dob"]].dropna(subset=["dob"])
    df = df.merge(ath, on="pid", how="inner")

    df["perf_date"] = pd.to_datetime(df["perf_date"])
    df["dob"] = pd.to_datetime(df["dob"])
    df["season"] = df["perf_date"].dt.year
    df["age"] = (df["perf_date"] - df["dob"]).dt.days / 365.25
    df = df[df["age"].between(*AGE_BOUNDS)]  # guard against bad dob/perf_date

    # direction-aware ordering: lower-better -> ascending; pick the best per group
    df["_ord"] = [
        m if is_lower_better(e) else -m
        for m, e in zip(df["mark"], df["event_id"], strict=False)
    ]
    idx = df.groupby(["pid", "event_id", "sex", "season"])["_ord"].idxmin()
    cols = ["pid", "event_id", "sex", "season", "age", "mark", "wind"]
    return df.loc[idx, cols].sort_values(["pid", "event_id", "season"]).reset_index(drop=True)
