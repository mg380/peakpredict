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

from ..common.event_maps import is_lower_better
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


def _model_peak(
    art: Artifacts, series: pd.DataFrame, event_id: str, sex: int, height_cm, weight_kg
) -> tuple[float, float, float]:
    """Forward peak-age prediction (peak, lo, hi) from whichever model the bundle ships."""
    model = art.predictor["model"]
    if art.predictor.get("primary") == "rnn":
        # sequence model scores the observed (age, score) series directly
        return model.predict_series(series, event_id, int(sex), height_cm, weight_kg)
    feats = compute_features(series)
    feats["height_cm"] = height_cm  # static physical inputs (None -> imputed)
    feats["weight_kg"] = weight_kg
    return model.predict_one(feats, event_id, int(sex))


def peak_for_series(
    art: Artifacts, series: pd.DataFrame, event_id: str, sex: int,
    height_cm=None, weight_kg=None,
) -> PeakPrediction:
    """Resolve an athlete's peak from their normalized (age, score) series.

    If the observed trajectory already turned over (an interior maximum), the
    peak is *history* — report the observed peak (``kind="actual"``). Otherwise
    the athlete is still ascending, so report the model's forward projection
    (``kind="predicted"``). Callers must ensure ``len(series) >= MIN_POINTS``.
    """
    best_score = float(series["score"].max())
    fit = fit_trajectory(series["age"].to_numpy(), series["score"].to_numpy())
    if fit is not None and fit.has_interior_max:
        # already peaked: the peak age is observed, the window is its near-peak band
        return PeakPrediction(
            peak_age=float(fit.peak_age),
            interval_lo=float(fit.window_lo), interval_hi=float(fit.window_hi),
            peak_score=best_score,
            window_lo=float(fit.window_lo), window_hi=float(fit.window_hi),
            confidence="ok", kind="actual",
        )
    # not yet peaked: project forward with the model; no observed window to draw
    peak_age, lo, hi = _model_peak(art, series, event_id, sex, height_cm, weight_kg)
    return PeakPrediction(
        peak_age=peak_age, interval_lo=lo, interval_hi=hi, peak_score=best_score,
        window_lo=float("nan"), window_hi=float("nan"),
        confidence=_confidence(peak_age, lo, hi, len(series)), kind="predicted",
    )


def predict_uploaded(
    art: Artifacts, athlete: UploadedAthlete
) -> tuple[PeakPrediction, pd.DataFrame]:
    """Resolve an uploaded athlete's peak; returns (prediction, normalized series)."""
    # event/sex absent from the bundle -> cannot score (distinct from too-few points)
    if not art.normalizer.has(athlete.event_id, int(athlete.sex)):
        return _flag("unsupported_event"), pd.DataFrame(columns=_EMPTY_SERIES_COLS)
    series = upload_to_series(art, athlete)
    if len(series) < MIN_POINTS:
        return _flag("insufficient"), series
    pred = peak_for_series(
        art, series, athlete.event_id, int(athlete.sex), athlete.height_cm, athlete.weight_kg
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


# numeric peak_age drives sort/filter; peak_age_display is the bracketed render
_DIRECTORY_COLS = [
    "name", "country", "seasons", "best_score", "best_time",
    "peak_age_display", "peak_age", "peak_kind", "pid",
]


def predicted_directory_peaks(art: Artifacts, event_id: str, sex: int) -> dict[int, float]:
    """Model-projected peak age for (event, sex) careers that have no measured peak.

    These are athletes still ascending (no observed interior maximum, hence no
    label). Returns ``{pid: predicted_peak_age}``. Batched so the whole roster is
    one forward pass; callers should cache per (bundle, event, sex).
    """
    g = art.season_bests
    g = g[(g["event_id"] == event_id) & (g["sex"] == sex)]
    if g.empty:
        return {}
    lab = art.labels
    labelled = set(lab[(lab["event_id"] == event_id) & (lab["sex"] == sex)]["pid"])
    items, pids = [], []
    for pid, career in g.groupby("pid"):
        if pid in labelled or len(career) < MIN_POINTS:
            continue
        items.append((career.sort_values("age")[["age", "score"]], event_id, int(sex), None, None))
        pids.append(int(pid))
    if not items:
        return {}
    model = art.predictor["model"]
    if art.predictor.get("primary") == "rnn":
        ages = model.predict_series_batch(items)
    else:  # tabular predictor: one feature-row prediction per career
        ages = [_model_peak(art, it[0], event_id, sex, None, None)[0] for it in items]
    return {pid: float(a) for pid, a in zip(pids, ages, strict=True)}


def athlete_directory(
    art: Artifacts, event_id: str, sex: int, sort_by: str = "Name (A–Z)",
    predicted: dict[int, float] | None = None,
) -> pd.DataFrame:
    """All athletes for an (event, sex) with summary stats, sorted for browsing.

    ``best_time`` is the athlete's fastest raw mark (seconds). ``peak_age`` is the
    measured peak where observed, else the model projection from ``predicted``;
    ``peak_kind`` ("actual"/"predicted") and ``peak_age_display`` (projections
    shown in brackets) let the UI distinguish the two.
    """
    sb = art.season_bests
    g = sb[(sb["event_id"] == event_id) & (sb["sex"] == sex)]
    if g.empty:
        return pd.DataFrame(columns=_DIRECTORY_COLS)
    # best raw time is the fastest mark for time events (lowest), slowest otherwise
    best_mark = ("mark", "min") if is_lower_better(event_id) else ("mark", "max")
    agg = (
        g.groupby("pid")
        .agg(seasons=("season", "count"), best_score=("score", "max"), best_time=best_mark)
        .reset_index()
    )
    agg = agg.merge(art.athletes[["pid", "name", "country"]], on="pid", how="left")
    lab = art.labels
    lab = lab[(lab["event_id"] == event_id) & (lab["sex"] == sex)][["pid", "peak_age"]]
    agg = agg.merge(lab, on="pid", how="left")
    agg["best_score"] = agg["best_score"].round(2)
    agg["best_time"] = agg["best_time"].round(2)
    agg["peak_age"] = agg["peak_age"].round(1)
    # measured peak -> "actual"; fill the rest from the model projection -> "predicted"
    agg["peak_kind"] = np.where(agg["peak_age"].notna(), "actual", "")
    if predicted:
        proj = agg["pid"].map(predicted).round(1)
        fill = agg["peak_age"].isna() & proj.notna()
        agg.loc[fill, "peak_age"] = proj[fill]
        agg.loc[fill, "peak_kind"] = "predicted"
    agg["peak_age_display"] = [
        "" if pd.isna(v) else (f"{v:.1f}" if k == "actual" else f"({v:.1f})")
        for v, k in zip(agg["peak_age"], agg["peak_kind"], strict=True)
    ]
    col, ascending = DIRECTORY_SORTS.get(sort_by, ("name", True))
    agg = agg.sort_values(col, ascending=ascending, na_position="last").reset_index(drop=True)
    return agg[_DIRECTORY_COLS]


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
