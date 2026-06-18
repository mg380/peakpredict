"""C5 — the shared Plotly hero chart used identically by Explore and Upload.

Layers (back to front): population percentile band -> population average ->
athlete's observed scores + fitted trajectory -> predicted peak window + peak
line. Colour roles are fixed so the two pages render identically; the palette
mirrors dashboard.theme (kept here as plain values so charting stays a pure
Plotly module with no Streamlit dependency).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..common.schemas import PeakPrediction
from ..pipeline.trajectory import fit_trajectory

COL_ATHLETE = "#16263A"
COL_PEAK = "#FF4D17"
COL_REFERENCE = "#B7B0A2"
COL_BAND = "rgba(183,176,162,0.22)"
COL_PEAK_WINDOW = "rgba(255,77,23,0.10)"
_INK = "#14161B"
_MUTED = "#60656F"
_GRID = "#E8E3D9"
_FONT = "IBM Plex Sans, system-ui, sans-serif"
_MONO = "IBM Plex Mono, monospace"

_DRAWN_CONFIDENCES = {"ok", "low", "out_of_distribution"}


def hero_chart(
    series: pd.DataFrame,
    prediction: PeakPrediction | None,
    overlay: pd.DataFrame | None = None,
    title: str = "",
) -> go.Figure:
    """Build the performance-over-time figure (athlete + population + peak)."""
    fig = go.Figure()

    if overlay is not None and len(overlay):
        fig.add_trace(go.Scatter(
            x=overlay["age_bin"], y=overlay["p90"], mode="lines",
            line={"width": 0}, hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=overlay["age_bin"], y=overlay["p10"], mode="lines", fill="tonexty",
            fillcolor=COL_BAND, line={"width": 0}, name="population 10–90%",
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=overlay["age_bin"], y=overlay["p50"], mode="lines",
            line={"color": COL_REFERENCE, "dash": "dash", "width": 1.5}, name="population avg",
        ))

    # athlete observed scores + fitted trajectory
    if len(series) >= 3 and series["age"].nunique() >= 3:
        fit = fit_trajectory(series["age"].to_numpy(), series["score"].to_numpy())
        if fit is not None:
            xs = np.linspace(series["age"].min(), series["age"].max(), 60)
            ys = fit.a * xs**2 + fit.b * xs + fit.c
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", line={"color": COL_ATHLETE, "width": 2.5},
                name="fitted trajectory", hoverinfo="skip",
            ))
    fig.add_trace(go.Scatter(
        x=series["age"], y=series["score"], mode="markers",
        marker={"color": COL_ATHLETE, "size": 10, "line": {"color": "#FFFFFF", "width": 1.5}},
        name="athlete",
    ))

    if prediction is not None and prediction.confidence in _DRAWN_CONFIDENCES:
        if math.isfinite(prediction.window_lo) and math.isfinite(prediction.window_hi):
            fig.add_vrect(
                x0=prediction.window_lo, x1=prediction.window_hi,
                fillcolor=COL_PEAK_WINDOW, line_width=0,
            )
        if math.isfinite(prediction.peak_age):
            # observed peak = solid "PEAK"; forward projection = dashed "PROJECTED PEAK"
            actual = getattr(prediction, "kind", "predicted") == "actual"
            label = "PEAK" if actual else "PROJECTED PEAK"
            fig.add_vline(
                x=prediction.peak_age,
                line={"color": COL_PEAK, "width": 2, "dash": "solid" if actual else "dash"},
                annotation_text=f"{label} ~{prediction.peak_age:.1f}y",
                annotation_position="top",
                annotation_font_color=COL_PEAK,
                annotation_font_size=12,
                annotation_font_family=_MONO,
            )

    # frame the age axis to actual content (observed ages, population overlay, and
    # any finite predicted-peak markers) so an outlier value can't stretch it
    x_vals = [float(a) for a in series["age"]]
    if overlay is not None and len(overlay):
        x_vals += [float(a) for a in overlay["age_bin"]]
    if prediction is not None:
        x_vals += [v for v in (prediction.peak_age, prediction.window_lo, prediction.window_hi)
                   if v is not None and math.isfinite(v)]
    x_range = None
    if x_vals:
        lo, hi = min(x_vals), max(x_vals)
        pad = max(0.5, 0.04 * (hi - lo))
        x_range = [lo - pad, hi + pad]

    xaxis = {
        "title": {"text": "AGE (YEARS)",
                  "font": {"size": 11, "color": _MUTED, "family": _MONO}},
        "gridcolor": _GRID, "zeroline": False, "showline": True, "linecolor": _GRID,
        "ticks": "outside", "tickcolor": _GRID,
    }
    if x_range is not None:
        xaxis["range"] = x_range
    layout = {
        "template": "plotly_white",
        "font": {"family": _FONT, "color": _INK, "size": 13},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "xaxis": xaxis,
        "yaxis": {
            "title": {
                "text": "PERFORMANCE SCORE · HIGHER = BETTER",
                "font": {"size": 11, "color": _MUTED, "family": _MONO},
            },
            "gridcolor": _GRID, "zeroline": False,
        },
        "legend": {"orientation": "h", "y": -0.24, "x": 0, "font": {"size": 11},
                   "bgcolor": "rgba(0,0,0,0)"},
        "hovermode": "x unified",
        "margin": {"l": 56, "r": 24, "t": 44 if title else 14, "b": 52},
    }
    if title:
        layout["title"] = {"text": title, "font": {"family": "Bricolage Grotesque, " + _FONT,
                                                    "size": 18, "color": _INK}}
    fig.update_layout(**layout)
    return fig
