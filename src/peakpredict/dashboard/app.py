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
from peakpredict.dashboard import service
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


def page_explore(art: service.Artifacts) -> None:
    st.header("Explore athletes")
    c1, c2, c3 = st.columns(3)
    event = c1.selectbox("Event", EVENTS, format_func=_event_label)
    sex_label = c2.selectbox("Sex", list(SEXES))
    sex = SEXES[sex_label]
    query = c3.text_input("Search name")

    present = art.season_bests
    present = present[(present["event_id"] == event) & (present["sex"] == sex)]["pid"].unique()
    roster = art.athletes[art.athletes["pid"].isin(present)][["pid", "name"]]
    if query:
        roster = roster[roster["name"].str.contains(query, case=False, na=False)]
    if roster.empty:
        st.info("No athletes match. Try another event/sex or clear the search.")
        return

    label = roster["name"] + "  (#" + roster["pid"].astype(str) + ")"
    choice = st.selectbox("Athlete", label.tolist())
    pid = int(choice.split("#")[-1].rstrip(")"))

    series = service.athlete_series(art, pid, event, sex)
    if series.empty:
        st.warning("No performances for this athlete in the selected event.")
        return
    pred = _descriptive_prediction(series) if len(series) >= service.MIN_POINTS else None
    overlay = service.population_overlay(art, event, sex)

    left, right = st.columns([3, 2])
    left.plotly_chart(
        hero_chart(series, pred, overlay, title=choice), width="stretch"
    )
    with right:
        st.subheader("Summary")
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
            st.subheader("Similar athletes")
            st.dataframe(peers[["name", "pid", "distance"]].head(5), hide_index=True)


def page_upload(art: service.Artifacts) -> None:
    st.header("Upload & Predict")
    c1, c2 = st.columns(2)
    event = c1.selectbox("Event", EVENTS, format_func=_event_label, key="up_event")
    sex = SEXES[c2.selectbox("Sex", list(SEXES), key="up_sex")]
    st.caption("Enter the athlete's season results (age in years, mark, optional wind).")
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
    st.plotly_chart(
        hero_chart(series, pred, overlay, title="Predicted peak"), width="stretch"
    )
    cols = st.columns(3)
    cols[0].metric("Predicted peak age", f"{pred.peak_age:.1f}y")
    cols[1].metric("80% interval", f"{pred.interval_lo:.1f}-{pred.interval_hi:.1f}")
    cols[2].metric("Confidence", pred.confidence)


def page_indicators(art: service.Artifacts) -> None:
    st.header("Peak-performance indicators")
    val = art.validation
    if isinstance(val.get("ridge"), dict):
        c = st.columns(3)
        c[0].metric("Model MAE (years)", f"{val['ridge']['mae']:.2f}")
        c[1].metric("Baseline MAE", f"{val['baseline']['mae']:.2f}")
        c[2].metric("80% coverage", f"{val['ridge']['interval_coverage']:.0%}")
    ind = pd.DataFrame(art.indicators.get("indicators", []))
    if not ind.empty:
        st.subheader("Feature correlations with peak age")
        st.dataframe(ind, hide_index=True, width="stretch")
    st.caption(art.indicators.get("literature_note", ""))


def main() -> None:
    st.set_page_config(page_title="PeakPredictor", layout="wide")
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

    st.sidebar.title("PeakPredictor")
    st.sidebar.caption(f"Data version: {art.manifest.get('version', '?')}")
    page = st.sidebar.radio("Page", ["Explore", "Upload & Predict", "Indicators"])
    if page == "Explore":
        page_explore(art)
    elif page == "Upload & Predict":
        page_upload(art)
    else:
        page_indicators(art)


main()
