# Screenshot capture guide

This folder holds the images used across the docs. Three of them are **already generated**
from the dashboard's own charting code on real bundle data; four are **full-page UI captures
you take** by running the app.

## Already generated (real, from code — no action needed)

| File | What it is |
|---|---|
| `chart-actual-peak.png` | Hero trajectory chart for an athlete who has **already peaked** (observed peak, solid line) |
| `chart-predicted-peak.png` | Hero trajectory chart for a **still-ascending** athlete (projected peak, dashed line) |
| `model-comparison.png` | Held-out MAE bar chart: baseline 2.24 → ridge 1.60 → RNN 1.38 |

These were produced by rendering `dashboard/charting.hero_chart` (and a small Plotly bar)
to PNG via `kaleido`. They reflect the exact code and palette the live app uses.

## To capture (run the app, screenshot, save here)

```bash
streamlit run src/peakpredict/dashboard/app.py
```

Capture each view at a wide browser width (≈1400 px) for crisp docs, then save with the
**exact filename** below so the embeds in [`../dashboard.md`](../dashboard.md) resolve.

| Save as | Page | What to show |
|---|---|---|
| `dashboard-explore.png` | **Explore** | The roster table — make sure the **Best time (s)** column and a mix of plain and **(bracketed)** peak ages are visible; the event/sex/sort controls at top. |
| `dashboard-athlete.png` | **Explore** (athlete selected) | An athlete clicked open, showing the hero chart + the summary panel (best score, peak age). Pick one that has already peaked for a clean inverted-U. |
| `dashboard-upload.png` | **Upload & Predict** | The results editor with a few seasons entered and a projection shown — peak age, 80% interval, and the chart with the dashed projected-peak line. |
| `dashboard-indicators.png` | **Indicators** | The feature-correlation table and the model validation metrics. |

> Tip: the login gate appears first (credentials from `.secrets` / `st.secrets`). Capture
> after logging in. If you'd rather not show real athlete names, the Upload page produces a
> fully synthetic example.

## Optional extras

- A capture of the **login screen** (`dashboard-login.png`) can lead the dashboard doc to
  show the credential gate.
- An animated GIF of selecting an athlete and watching the chart update reads very well in a
  portfolio README, if you want to add one as `dashboard-demo.gif`.
