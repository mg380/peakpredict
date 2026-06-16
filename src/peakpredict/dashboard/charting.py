"""C5 — the shared Plotly hero chart used identically by Explore and Upload.

Layers (back to front): population percentile band -> population average ->
athlete's observed scores -> predicted peak window + peak line. Colour roles are
fixed (design spec) so the two pages render identically.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..common.schemas import PeakPrediction
from ..pipeline.trajectory import fit_trajectory

COL_ATHLETE = "#1F6FEB"
COL_PEAK = "#E8590C"
COL_REFERENCE = "#9AA5B1"
COL_BAND = "rgba(154,165,177,0.20)"
COL_PEAK_WINDOW = "rgba(232,89,12,0.12)"

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
            fillcolor=COL_BAND, line={"width": 0}, name="population 10-90%",
        ))
        fig.add_trace(go.Scatter(
            x=overlay["age_bin"], y=overlay["p50"], mode="lines",
            line={"color": COL_REFERENCE, "dash": "dash"}, name="population avg",
        ))

    # athlete observed scores + fitted trajectory
    fig.add_trace(go.Scatter(
        x=series["age"], y=series["score"], mode="markers",
        marker={"color": COL_ATHLETE, "size": 9}, name="athlete",
    ))
    # draw the same trajectory the pipeline fits (not an independent polyfit),
    # and only when ages vary enough to fit a parabola
    if len(series) >= 3 and series["age"].nunique() >= 3:
        fit = fit_trajectory(series["age"].to_numpy(), series["score"].to_numpy())
        if fit is not None:
            xs = np.linspace(series["age"].min(), series["age"].max(), 50)
            ys = fit.a * xs**2 + fit.b * xs + fit.c
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line={"color": COL_ATHLETE}, name="fitted trajectory",
            ))

    if prediction is not None and prediction.confidence in _DRAWN_CONFIDENCES:
        if math.isfinite(prediction.window_lo) and math.isfinite(prediction.window_hi):
            fig.add_vrect(
                x0=prediction.window_lo, x1=prediction.window_hi,
                fillcolor=COL_PEAK_WINDOW, line_width=0,
            )
        if math.isfinite(prediction.peak_age):
            fig.add_vline(
                x=prediction.peak_age, line={"color": COL_PEAK, "width": 2},
                annotation_text=f"peak ~{prediction.peak_age:.1f}y",
            )

    fig.update_layout(
        title=title, template="plotly_white",
        xaxis_title="age (years)", yaxis_title="performance score (higher = better)",
        legend={"orientation": "h", "y": -0.2},
    )
    return fig
