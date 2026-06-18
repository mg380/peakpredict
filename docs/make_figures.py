"""Regenerate the documentation figures from a published bundle.

The three chart PNGs under ``docs/images/`` are produced here — not hand-made — by
rendering the dashboard's own ``hero_chart`` (and a small Plotly bar) to PNG with
``kaleido``. Run after publishing a bundle:

    pip install -e ".[dashboard,dev]"     # dev extra includes kaleido
    python docs/make_figures.py

Requires a bundle in ``data/artifacts/`` (git-ignored, regenerable via
``peakpredict-build`` + ``peakpredict-publish``). Example athletes are chosen
deterministically by trajectory shape, so reruns are stable for a given bundle.
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from peakpredict.dashboard.charting import hero_chart
from peakpredict.dashboard.service import (
    athlete_series,
    find_latest_bundle,
    load_bundle,
    peak_for_series,
    population_overlay,
)
from peakpredict.pipeline.trajectory import fit_trajectory

IMAGES = Path(__file__).parent / "images"
PAPER, PLOT, INK = "#F4F2EC", "#FCFBF7", "#14161B"


def _name(art, pid: int) -> str:
    n = art.athletes[art.athletes["pid"] == pid]["name"]
    return n.iloc[0] if len(n) else str(pid)


def _save(fig: go.Figure, path: Path, title: str, yrange=None) -> None:
    fig.update_layout(
        paper_bgcolor=PAPER, plot_bgcolor=PLOT, width=900, height=480,
        title={"text": title, "x": 0.02, "font": {"size": 17, "color": INK}},
        margin={"l": 60, "r": 30, "t": 54, "b": 60},
    )
    if yrange:
        fig.update_yaxes(range=yrange)
    fig.write_image(str(path), scale=2)
    print("wrote", path.relative_to(path.parents[2]))


def _pick(art, event, sex, want_peaked: bool, *, min_n, max_n, max_age=None, min_rise=None):
    """First athlete in (event, sex) matching a trajectory shape (deterministic)."""
    sb = art.season_bests
    for pid, g in sb[(sb["event_id"] == event) & (sb["sex"] == sex)].groupby("pid"):
        if not (min_n <= len(g) <= max_n):
            continue
        g = g.sort_values("age")
        if max_age and g["age"].max() >= max_age:
            continue
        fit = fit_trajectory(g["age"].to_numpy(), g["score"].to_numpy())
        if fit is None or fit.has_interior_max != want_peaked:
            continue
        if min_rise and (g["score"].iloc[-1] - g["score"].iloc[0]) < min_rise:
            continue
        return int(pid)
    raise SystemExit(f"no athlete matched in ({event}, {sex})")


def main() -> int:
    IMAGES.mkdir(parents=True, exist_ok=True)
    art = load_bundle(find_latest_bundle("data/artifacts"))

    # 1. observed peak (already peaked) — a clean inverted-U 400m woman
    pid = _pick(art, "70", 2, want_peaked=True, min_n=9, max_n=20)
    s = athlete_series(art, pid, "70", 2)
    pred = peak_for_series(art, s[["age", "score"]], "70", 2)
    _save(hero_chart(s, pred, population_overlay(art, "70", 2)),
          IMAGES / "chart-actual-peak.png",
          f"{_name(art, pid)} · 400m · observed peak (already peaked)")

    # 2. projected peak (still ascending) — a rising young 400m woman
    pid = _pick(art, "70", 2, want_peaked=False, min_n=5, max_n=7, max_age=24, min_rise=0.6)
    s = athlete_series(art, pid, "70", 2)
    pred = peak_for_series(art, s[["age", "score"]], "70", 2)
    _save(hero_chart(s, pred, population_overlay(art, "70", 2)),
          IMAGES / "chart-predicted-peak.png",
          f"{_name(art, pid)} · 400m · projected peak (still ascending)", yrange=[-2.6, 1.7])

    # 3. model comparison bar
    r = art.validation
    bars = [("Group-mean\nbaseline", "baseline", "#B7B0A2"),
            ("Pooled\nRidge", "ridge", "#16263A"),
            ("Bidirectional\nRNN", "rnn", "#FF4D17")]
    fig = go.Figure(go.Bar(
        x=[b[0] for b in bars], y=[r[b[1]]["mae"] for b in bars],
        marker_color=[b[2] for b in bars],
        text=[f"{r[b[1]]['mae']:.2f}y" for b in bars], textposition="outside",
        textfont={"size": 15, "family": "IBM Plex Mono"}))
    fig.update_layout(
        paper_bgcolor=PAPER, plot_bgcolor=PLOT, width=760, height=460, showlegend=False,
        title={"text": "Held-out peak-age error (MAE, years · lower is better)",
               "x": 0.02, "font": {"size": 16}},
        yaxis={"title": "MAE (years)", "range": [0, 2.6], "gridcolor": "#E8E3D9"},
        xaxis={"tickfont": {"size": 13}}, margin={"l": 60, "r": 30, "t": 56, "b": 50})
    fig.write_image(str(IMAGES / "model-comparison.png"), scale=2)
    print("wrote", (IMAGES / "model-comparison.png").relative_to(IMAGES.parents[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
