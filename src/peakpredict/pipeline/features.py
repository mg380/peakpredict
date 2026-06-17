"""B5 — early-career features (leakage-safe) and the published feature schema.

Features describe what is known about an athlete from their first *k* observed
seasons only — never the peak or anything after it. The same ``compute_features``
runs on a user-uploaded athlete in the dashboard, so training and inference
features are produced by one implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..common.schemas import FeatureSchema, FieldSpec

FEATURE_SCHEMA_VERSION = "2"  # v2 adds static physical features (height/weight)
DEFAULT_CUTOFFS = (3, 5, 7)

# Engineered model features (computed from the first-k season-bests).
_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("n_seasons", "int", "number of observed seasons"),
    ("debut_age", "float", "age at first observed season"),
    ("debut_score", "float", "normalized score at debut"),
    ("current_age", "float", "age at the latest observed season"),
    ("current_best_score", "float", "best normalized score observed so far"),
    ("span_observed", "float", "years between first and latest observed season"),
    ("progression_rate", "float", "slope of score vs age over observed seasons"),
    ("recent_slope", "float", "slope of score vs age over the last up-to-3 seasons"),
    ("mean_score", "float", "mean observed score"),
    ("score_std", "float", "std of observed score (consistency)"),
)

# Static physical features (per athlete; often missing -> imputed by the model).
_PHYSICAL_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("height_cm", "float", "athlete height in cm (static; may be missing)"),
    ("weight_kg", "float", "athlete weight in kg (static; may be missing)"),
)

FEATURE_NAMES: tuple[str, ...] = tuple(n for n, _, _ in _FIELDS)
PHYSICAL_NAMES: tuple[str, ...] = tuple(n for n, _, _ in _PHYSICAL_FIELDS)


def feature_schema() -> FeatureSchema:
    """The versioned schema describing all model input features (engineered + physical)."""
    return FeatureSchema(
        schema_version=FEATURE_SCHEMA_VERSION,
        fields=[
            FieldSpec(name=n, dtype=d, description=desc)
            for n, d, desc in (*_FIELDS, *_PHYSICAL_FIELDS)
        ],
    )


def _slope(ages: np.ndarray, scores: np.ndarray) -> float:
    if len(ages) < 2:
        return 0.0
    return float(np.polyfit(ages, scores, 1)[0])


def compute_features(obs: pd.DataFrame) -> dict:
    """Engineered features from a frame of observed (age, score) rows.

    ``obs`` must be sorted by age and contain only the data available at the
    prediction cutoff (the leakage guard lives at the call site that slices it).
    """
    ages = obs["age"].to_numpy(dtype=float)
    scores = obs["score"].to_numpy(dtype=float)
    recent = obs.tail(3)
    return {
        "n_seasons": int(len(obs)),
        "debut_age": float(ages[0]),
        "debut_score": float(scores[0]),
        "current_age": float(ages[-1]),
        "current_best_score": float(scores.max()),
        "span_observed": float(ages[-1] - ages[0]),
        "progression_rate": _slope(ages, scores),
        "recent_slope": _slope(recent["age"].to_numpy(float), recent["score"].to_numpy(float)),
        "mean_score": float(scores.mean()),
        "score_std": float(scores.std(ddof=0)),
    }


def build_features(
    scored_season_bests: pd.DataFrame,
    labels: pd.DataFrame,
    physical: pd.DataFrame | None = None,
    cutoffs: tuple[int, ...] = DEFAULT_CUTOFFS,
) -> pd.DataFrame:
    """Training table: leakage-safe features at each cutoff, joined to the label.

    For every labelled career and each cutoff ``k`` (when the athlete has >= k
    seasons), compute features from the first ``k`` season-bests, attach the
    full-career ``peak_age`` label, and the athlete's static physical attributes
    (``physical`` = DataFrame with pid/height_cm/weight_kg; NaN where unknown).
    """
    lab = labels.set_index(["pid", "event_id", "sex"])["peak_age"]
    rows: list[dict] = []
    for (pid, event_id, sex), g in scored_season_bests.groupby(["pid", "event_id", "sex"]):
        key = (int(pid), event_id, int(sex))
        if key not in lab.index:
            continue
        g = g.sort_values("age").reset_index(drop=True)
        for k in cutoffs:
            if len(g) < k:
                continue
            obs = g.iloc[:k]  # leakage guard: first k seasons only
            feats = compute_features(obs)
            feats.update(
                {
                    "pid": int(pid),
                    "event_id": event_id,
                    "sex": int(sex),
                    "cutoff_k": k,
                    "cutoff_age": float(obs["age"].iloc[-1]),
                    "peak_age": float(lab.loc[key]),
                }
            )
            rows.append(feats)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if physical is not None:
        df = df.merge(physical[["pid", *PHYSICAL_NAMES]], on="pid", how="left")
    else:
        for col in PHYSICAL_NAMES:
            df[col] = np.nan
    return df
