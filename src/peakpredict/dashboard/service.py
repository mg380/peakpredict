"""C1 — artifact loading + the non-UI inference/explore logic.

Keeps everything the dashboard does (load a bundle, refuse incompatible ones,
predict an uploaded athlete, assemble an existing athlete's view, find peers)
out of the Streamlit layer so it can be unit-tested. Uploads are scored with the
bundle's own normalizer and the pipeline's ``compute_features`` — identical to
training, so there is no train/inference drift.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..common.io import read_parquet
from ..common.normalization import ZScoreNormalizer
from ..common.schemas import PeakPrediction, UploadedAthlete
from ..pipeline.features import (
    DEFAULT_CUTOFFS,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    compute_features,
)
from ..pipeline.season_best import LEGAL_WIND_MAX
from ..pipeline.trajectory import fit_trajectory

MIN_POINTS = 3
MAX_TRAINED_SEASONS = max(DEFAULT_CUTOFFS)  # model is trained on first-k features, k <= this
PLAUSIBLE_PEAK_AGE = (14.0, 42.0)
WIDE_INTERVAL = 8.0
_EMPTY_SERIES_COLS = ["age", "score", "mark"]


class IncompatibleArtifactError(RuntimeError):
    """Raised when a bundle's feature schema is not what the dashboard expects."""


@dataclass
class Artifacts:
    manifest: dict
    predictor: dict
    normalizer: ZScoreNormalizer
    feature_schema: dict
    aggregates: pd.DataFrame
    similar_index: pd.DataFrame
    indicators: dict
    validation: dict
    season_bests: pd.DataFrame
    labels: pd.DataFrame
    athletes: pd.DataFrame


def load_bundle(path: str | Path) -> Artifacts:
    p = Path(path)
    return Artifacts(
        manifest=json.loads((p / "manifest.json").read_text()),
        predictor=joblib.load(p / "predictor.pkl"),
        normalizer=ZScoreNormalizer.from_dict(json.loads((p / "normalization.json").read_text())),
        feature_schema=json.loads((p / "feature_schema.json").read_text()),
        aggregates=read_parquet(p / "aggregates.parquet"),
        similar_index=read_parquet(p / "similar_index.parquet"),
        indicators=json.loads((p / "indicators.json").read_text()),
        validation=json.loads((p / "validation.json").read_text()),
        season_bests=read_parquet(p / "season_bests.parquet"),
        labels=read_parquet(p / "labels.parquet"),
        athletes=read_parquet(p / "athletes.parquet"),
    )


def check_compatible(art: Artifacts, expected: str = FEATURE_SCHEMA_VERSION) -> None:
    """Refuse to serve a bundle whose feature schema the dashboard can't consume."""
    got = art.feature_schema.get("schema_version")
    if got != expected:
        raise IncompatibleArtifactError(
            f"dashboard expects feature schema '{expected}', bundle has '{got}'"
        )


def find_latest_bundle(root: str | Path) -> Path | None:
    versions = sorted(p for p in Path(root).glob("*") if (p / "manifest.json").exists())
    return versions[-1] if versions else None


def _flag(confidence: str) -> PeakPrediction:
    nan = float("nan")
    return PeakPrediction(
        peak_age=nan, interval_lo=nan, interval_hi=nan,
        peak_score=nan, window_lo=nan, window_hi=nan, confidence=confidence,
    )


def _confidence(peak_age: float, lo: float, hi: float, n_seasons: int) -> str:
    if not (PLAUSIBLE_PEAK_AGE[0] <= peak_age <= PLAUSIBLE_PEAK_AGE[1]):
        return "out_of_distribution"
    # more observed seasons than the model was trained on -> extrapolation
    if n_seasons > MAX_TRAINED_SEASONS or (hi - lo) > WIDE_INTERVAL:
        return "low"
    return "ok"


def upload_to_series(art: Artifacts, athlete: UploadedAthlete) -> pd.DataFrame:
    """Normalize an upload's wind-legal results into an (age, score) frame.

    Returns an empty frame (with the expected columns) if no legal results
    remain. Assumes the (event, sex) is present in the normalizer.
    """
    rows = []
    for r in athlete.results:
        if r.wind is not None and r.wind > LEGAL_WIND_MAX:
            continue
        score = art.normalizer.transform(r.mark, athlete.event_id, int(athlete.sex))
        rows.append({"age": float(r.age), "score": float(score), "mark": r.mark})
    if not rows:
        return pd.DataFrame(columns=_EMPTY_SERIES_COLS)
    return pd.DataFrame(rows).sort_values("age").reset_index(drop=True)


def predict_uploaded(
    art: Artifacts, athlete: UploadedAthlete
) -> tuple[PeakPrediction, pd.DataFrame]:
    """Predict an uploaded athlete's peak; returns (prediction, normalized series)."""
    # event/sex absent from the bundle -> cannot score (distinct from too-few points)
    if not art.normalizer.has(athlete.event_id, int(athlete.sex)):
        return _flag("unsupported_event"), pd.DataFrame(columns=_EMPTY_SERIES_COLS)
    series = upload_to_series(art, athlete)
    if len(series) < MIN_POINTS:
        return _flag("insufficient"), series
    feats = compute_features(series)
    peak_age, lo, hi = art.predictor["model"].predict_one(feats, athlete.event_id, int(athlete.sex))
    # the peak window is curvature-derived; only defined when the fit has an interior max,
    # otherwise leave it undefined rather than conflating it with the prediction interval
    fit = fit_trajectory(series["age"].to_numpy(), series["score"].to_numpy())
    if fit is not None and fit.has_interior_max:
        window_lo, window_hi = float(fit.window_lo), float(fit.window_hi)
    else:
        window_lo = window_hi = float("nan")
    pred = PeakPrediction(
        peak_age=peak_age, interval_lo=lo, interval_hi=hi,
        peak_score=float(series["score"].max()),
        window_lo=window_lo, window_hi=window_hi,
        confidence=_confidence(peak_age, lo, hi, len(series)),
    )
    return pred, series


# -- Explore helpers ------------------------------------------------------
# (label shown in the UI -> (column, ascending)) for browsing the athlete directory
DIRECTORY_SORTS: dict[str, tuple[str, bool]] = {
    "Name (A–Z)": ("name", True),
    "Best performance": ("best_score", False),
    "Most seasons": ("seasons", False),
    "Earliest peak": ("peak_age", True),
    "Latest peak": ("peak_age", False),
}


def athlete_directory(
    art: Artifacts, event_id: str, sex: int, sort_by: str = "Name (A–Z)"
) -> pd.DataFrame:
    """All athletes for an (event, sex) with summary stats, sorted for browsing.

    Columns: name, country, seasons, best_score, peak_age, pid.
    """
    sb = art.season_bests
    g = sb[(sb["event_id"] == event_id) & (sb["sex"] == sex)]
    if g.empty:
        return pd.DataFrame(columns=["name", "country", "seasons", "best_score", "peak_age", "pid"])
    agg = (
        g.groupby("pid").agg(seasons=("season", "count"), best_score=("score", "max")).reset_index()
    )
    agg = agg.merge(art.athletes[["pid", "name", "country"]], on="pid", how="left")
    lab = art.labels
    lab = lab[(lab["event_id"] == event_id) & (lab["sex"] == sex)][["pid", "peak_age"]]
    agg = agg.merge(lab, on="pid", how="left")
    agg["best_score"] = agg["best_score"].round(2)
    agg["peak_age"] = agg["peak_age"].round(1)
    col, ascending = DIRECTORY_SORTS.get(sort_by, ("name", True))
    agg = agg.sort_values(col, ascending=ascending, na_position="last").reset_index(drop=True)
    return agg[["name", "country", "seasons", "best_score", "peak_age", "pid"]]


def apply_directory_filters(
    df: pd.DataFrame,
    *,
    countries: list[str] | None = None,
    seasons: tuple[float, float] | None = None,
    best_score: tuple[float, float] | None = None,
    peak_age: tuple[float, float] | None = None,
    include_no_peak: bool = True,
) -> pd.DataFrame:
    """Filter an athlete directory by column values (all filters optional, AND-ed).

    ``peak_age`` filtering keeps athletes whose peak is in range; athletes with no
    peak estimate (NaN) are kept only when ``include_no_peak`` is True.
    """
    out = df
    if countries:
        out = out[out["country"].isin(countries)]
    if seasons is not None:
        out = out[out["seasons"].between(seasons[0], seasons[1])]
    if best_score is not None:
        out = out[out["best_score"].between(best_score[0], best_score[1])]
    if peak_age is not None:
        in_range = out["peak_age"].between(peak_age[0], peak_age[1])
        out = out[in_range | out["peak_age"].isna()] if include_no_peak else out[in_range]
    elif not include_no_peak:
        out = out[out["peak_age"].notna()]
    return out.reset_index(drop=True)


def athlete_series(art: Artifacts, pid: int, event_id: str, sex: int) -> pd.DataFrame:
    sb = art.season_bests
    g = sb[(sb["pid"] == pid) & (sb["event_id"] == event_id) & (sb["sex"] == sex)]
    return g.sort_values("age")[["age", "score", "season", "mark"]].reset_index(drop=True)


def population_overlay(art: Artifacts, event_id: str, sex: int) -> pd.DataFrame:
    a = art.aggregates
    return a[(a["event_id"] == event_id) & (a["sex"] == sex)].sort_values("age_bin")


def similar_athletes(
    art: Artifacts, feats: dict, event_id: str, sex: int, k: int = 5
) -> pd.DataFrame:
    pool = art.similar_index
    pool = pool[(pool["event_id"] == event_id) & (pool["sex"] == sex)].copy()
    if pool.empty:
        return pool
    x = pool[list(FEATURE_NAMES)].to_numpy(dtype=float)
    mu, sd = x.mean(0), x.std(0)
    sd[sd == 0] = 1.0
    q = np.array([feats[f] for f in FEATURE_NAMES], dtype=float)
    dist = np.sqrt((((x - mu) / sd - (q - mu) / sd) ** 2).sum(axis=1))
    pool["distance"] = dist
    return pool.nsmallest(k, "distance")
