"""Streamlit dashboard entry point (Component C).

Credential-gated app over a published artifact bundle: Explore existing
athletes, Upload & Predict a new athlete, and review Indicators. All data logic
lives in ``service``; charts in ``charting``. Run:

    streamlit run src/peakpredict/dashboard/app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from pydantic import ValidationError

# Absolute imports: Streamlit runs this file as a standalone script, so it is not
# loaded as part of the package and cannot use relative imports.
from peakpredict.common.event_maps import SUPPORTED_V1_EVENTS, event_name
from peakpredict.common.schemas import PeakPrediction, UploadedAthlete, UploadedResult
from peakpredict.dashboard import service, theme
from peakpredict.dashboard.auth import require_auth
from peakpredict.dashboard.charting import hero_chart
from peakpredict.pipeline.features import compute_features
from peakpredict.pipeline.trajectory import fit_trajectory

EVENTS = sorted(SUPPORTED_V1_EVENTS)
SEXES = {"Men": 1, "Women": 2}


def _resolve_bundle() -> Path | None:
    override = os.environ.get("PEAKPREDICT_BUNDLE")
    if override:
        return Path(override)
    for root in ("data/artifacts", "data/artifacts_dev"):
        found = service.find_latest_bundle(root)
        if found:
            return found
    return None


@st.cache_resource
def _load(bundle_path: str) -> service.Artifacts:
    art = service.load_bundle(bundle_path)
    service.check_compatible(art)
    return art


def _event_label(event_id: str) -> str:
    return f"{event_name(event_id)} ({event_id})"


def _descriptive_prediction(series: pd.DataFrame) -> PeakPrediction | None:
    fit = fit_trajectory(series["age"].to_numpy(), series["score"].to_numpy())
    if fit is None or not fit.has_interior_max:
        return None
    return PeakPrediction(
        peak_age=fit.peak_age, interval_lo=fit.window_lo, interval_hi=fit.window_hi,
        peak_score=fit.peak_score, window_lo=fit.window_lo, window_hi=fit.window_hi,
        confidence="ok",
    )


def _filter_roster(df: pd.DataFrame) -> pd.DataFrame:
    """Render per-column filter controls and return the filtered directory."""
    if df.empty:
        return df
    f: dict = {}
    with st.expander("Filter roster"):
        countries = st.multiselect("Country", sorted(df["country"].dropna().unique()))
        f["countries"] = countries or None
        smin, smax = int(df["seasons"].min()), int(df["seasons"].max())
        if smin < smax:
            f["seasons"] = st.slider("Seasons", smin, smax, (smin, smax))
        bmin = round(float(df["best_score"].min()), 2)
        bmax = round(float(df["best_score"].max()), 2)
        if bmin < bmax:
            f["best_score"] = st.slider("Best score", bmin, bmax, (bmin, bmax))
        peaks = df["peak_age"].dropna()
        if not peaks.empty:
            f["include_no_peak"] = st.checkbox(
                "Include athletes without a peak estimate", value=True
            )
            pmin, pmax = round(float(peaks.min()), 1), round(float(peaks.max()), 1)
            if pmin < pmax:
                f["peak_age"] = st.slider("Peak age", pmin, pmax, (pmin, pmax))
    return service.apply_directory_filters(df, **f)


def page_explore(art: service.Artifacts) -> None:
    st.header("Explore athletes")
    c1, c2, c3, c4 = st.columns(4)
    event = c1.selectbox("Event", EVENTS, format_func=_event_label)
    sex = SEXES[c2.selectbox("Sex", list(SEXES))]
    sort_by = c3.selectbox("Sort by", list(service.DIRECTORY_SORTS))
    query = c4.text_input("Search name")

    directory = service.athlete_directory(art, event, sex, sort_by)
    if query:
        directory = directory[directory["name"].str.contains(query, case=False, na=False)]
    directory = _filter_roster(directory)
    if directory.empty:
        st.info("No athletes match the current filters.")
        return

    # browse the full roster: click a row to view that athlete
    theme.eyebrow("roster")
    st.caption(f"{len(directory)} athletes — click a row to view")
    selection = st.dataframe(
        directory,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        column_config={"pid": None},  # hide the internal id column
    )
    rows = selection.selection.rows if selection and selection.selection else []
    chosen = directory.iloc[rows[0] if rows else 0]
    pid = int(chosen["pid"])

    series = service.athlete_series(art, pid, event, sex)
    if series.empty:
        st.warning("No performances for this athlete in the selected event.")
        return
    pred = _descriptive_prediction(series) if len(series) >= service.MIN_POINTS else None
    overlay = service.population_overlay(art, event, sex)

    left, right = st.columns([3, 2], gap="large")
    with left:
        theme.eyebrow(f"performance trajectory · {chosen['name']}")
        st.plotly_chart(hero_chart(series, pred, overlay, title=""), width="stretch")
    with right:
        theme.eyebrow("summary")
        st.metric("Seasons", len(series))
        st.metric("Age range", f"{series['age'].min():.0f}-{series['age'].max():.0f}")
        st.metric("Best score", f"{series['score'].max():.2f}")
        if pred:
            st.metric("Est. peak age", f"{pred.peak_age:.1f}y")
        elif len(series) < service.MIN_POINTS:
            st.info("Too few seasons to fit a reliable peak.")
        else:
            st.info("No clear interior peak observed yet.")

    if len(series) >= service.MIN_POINTS:
        feats = compute_features(series)
        peers = service.similar_athletes(art, feats, event, sex, k=6)  # +1 to drop self
        peers = peers[peers["pid"] != pid].merge(
            art.athletes[["pid", "name"]], on="pid", how="left"
        )
        if not peers.empty:
            theme.eyebrow("similar athletes")
            st.dataframe(peers[["name", "pid", "distance"]].head(5), hide_index=True)


def page_upload(art: service.Artifacts) -> None:
    st.header("Upload & Predict")
    c1, c2 = st.columns(2)
    event = c1.selectbox("Event", EVENTS, format_func=_event_label, key="up_event")
    sex = SEXES[c2.selectbox("Sex", list(SEXES), key="up_sex")]
    theme.eyebrow("enter results")
    st.caption("Age in years, mark, optional wind. Add a row per season.")
    starter = pd.DataFrame(
        {"age": [18.0, 19.0, 20.0], "mark": [0.0, 0.0, 0.0], "wind": [None, None, None]}
    )
    edited = st.data_editor(starter, num_rows="dynamic", width="stretch")

    if not st.button("Predict peak"):
        return
    try:
        results = [
            UploadedResult(
                age=float(r.age), mark=float(r.mark),
                wind=(None if pd.isna(r.wind) else float(r.wind)),
            )
            for r in edited.itertuples()
            if float(r.mark) > 0
        ]
        athlete = UploadedAthlete(sex=sex, event_id=event, results=results)
    except (ValidationError, ValueError) as exc:
        st.error(f"Please fix the input: {exc}")
        return

    pred, series = service.predict_uploaded(art, athlete)
    if pred.confidence == "unsupported_event":
        st.error("This event/sex is not available in the current model bundle.")
        return
    if pred.confidence == "insufficient":
        st.warning(f"Need at least {service.MIN_POINTS} valid results to predict.")
        st.dataframe(series, hide_index=True)
        return
    if pred.confidence == "out_of_distribution":
        st.warning("Inputs fall outside the model's training range — shown with low confidence.")

    overlay = service.population_overlay(art, event, sex)
    theme.eyebrow("projection")
    st.plotly_chart(hero_chart(series, pred, overlay, title=""), width="stretch")
    cols = st.columns(3)
    cols[0].metric("Predicted peak age", f"{pred.peak_age:.1f}y")
    cols[1].metric("80% interval", f"{pred.interval_lo:.1f}-{pred.interval_hi:.1f}")
    cols[2].metric("Confidence", pred.confidence)


def page_indicators(art: service.Artifacts) -> None:
    st.header("Peak-performance indicators")
    val = art.validation
    if isinstance(val.get("ridge"), dict):
        theme.eyebrow("model validation")
        c = st.columns(3)
        c[0].metric("Model MAE (years)", f"{val['ridge']['mae']:.2f}")
        c[1].metric("Baseline MAE", f"{val['baseline']['mae']:.2f}")
        c[2].metric("80% coverage", f"{val['ridge']['interval_coverage']:.0%}")
    ind = pd.DataFrame(art.indicators.get("indicators", []))
    if not ind.empty:
        theme.eyebrow("feature correlations with peak age")
        st.dataframe(ind, hide_index=True, width="stretch")
    st.caption(art.indicators.get("literature_note", ""))


def main() -> None:
    st.set_page_config(page_title="PeakPredictor", layout="wide", page_icon="◆")
    theme.inject()
    require_auth()
    bundle = _resolve_bundle()
    if bundle is None:
        st.error("No artifact bundle found. Run the pipeline to publish one (data/artifacts/).")
        st.stop()
    try:
        art = _load(str(bundle))
    except service.IncompatibleArtifactError as exc:
        st.error(f"Incompatible artifact bundle — refusing to serve predictions.\n\n{exc}")
        st.stop()

    version = art.manifest.get("version", "?")
    theme.sidebar_brand()
    with st.sidebar:
        theme.eyebrow("navigate")
    page = st.sidebar.radio("Page", ["Explore", "Upload & Predict", "Indicators"],
                            label_visibility="collapsed")
    with st.sidebar:
        theme.eyebrow("model bundle")
        st.sidebar.caption(f"version · {version}")

    theme.header(version)
    if page == "Explore":
        page_explore(art)
    elif page == "Upload & Predict":
        page_upload(art)
    else:
        page_indicators(art)


main()
