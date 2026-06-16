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

from ..common.normalization import ZScoreNormalizer
from ..common.schemas import PeakPrediction, UploadedAthlete
from ..pipeline.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, compute_features
from ..pipeline.season_best import LEGAL_WIND_MAX
from ..pipeline.trajectory import fit_trajectory

MIN_POINTS = 3
PLAUSIBLE_PEAK_AGE = (14.0, 42.0)
WIDE_INTERVAL = 8.0


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
        aggregates=pd.read_parquet(p / "aggregates.parquet"),
        similar_index=pd.read_parquet(p / "similar_index.parquet"),
        indicators=json.loads((p / "indicators.json").read_text()),
        validation=json.loads((p / "validation.json").read_text()),
        season_bests=pd.read_parquet(p / "season_bests.parquet"),
        labels=pd.read_parquet(p / "labels.parquet"),
        athletes=pd.read_parquet(p / "athletes.parquet"),
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


def _confidence(peak_age: float, lo: float, hi: float) -> str:
    if not (PLAUSIBLE_PEAK_AGE[0] <= peak_age <= PLAUSIBLE_PEAK_AGE[1]):
        return "out_of_distribution"
    if (hi - lo) > WIDE_INTERVAL:
        return "low"
    return "ok"


def upload_to_series(art: Artifacts, athlete: UploadedAthlete) -> pd.DataFrame:
    """Normalize an upload's wind-legal results into an (age, score) frame."""
    rows = []
    for r in athlete.results:
        if r.wind is not None and r.wind > LEGAL_WIND_MAX:
            continue
        score = art.normalizer.transform(r.mark, athlete.event_id, int(athlete.sex))
        rows.append({"age": float(r.age), "score": float(score), "mark": r.mark})
    return pd.DataFrame(rows).sort_values("age").reset_index(drop=True)


def predict_uploaded(
    art: Artifacts, athlete: UploadedAthlete
) -> tuple[PeakPrediction, pd.DataFrame]:
    """Predict an uploaded athlete's peak; returns (prediction, normalized series)."""
    try:
        series = upload_to_series(art, athlete)
    except KeyError:
        # the bundle has no data for this (event, sex) -> cannot score
        return _flag("unsupported_event"), pd.DataFrame(columns=["age", "score", "mark"])
    if len(series) < MIN_POINTS:
        return _flag("insufficient"), series
    feats = compute_features(series)
    peak_age, lo, hi = art.predictor["model"].predict_one(feats, athlete.event_id, int(athlete.sex))
    fit = fit_trajectory(series["age"].to_numpy(), series["score"].to_numpy())
    window_lo = fit.window_lo if fit and fit.window_lo is not None else lo
    window_hi = fit.window_hi if fit and fit.window_hi is not None else hi
    pred = PeakPrediction(
        peak_age=peak_age, interval_lo=lo, interval_hi=hi,
        peak_score=float(series["score"].max()),
        window_lo=float(window_lo), window_hi=float(window_hi),
        confidence=_confidence(peak_age, lo, hi),
    )
    return pred, series


# -- Explore helpers ------------------------------------------------------
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
