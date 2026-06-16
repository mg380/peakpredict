# Peak Performance Predictor — Design Specification

> Status: DRAFT v1 (Step 3 of newbuild workflow). Scope: the dashboard (Component C). Components A (scraping) and B (analysis) are non-UI and excluded except for the artifact contract they expose. Builds on business-requirements.md and user-journeys.md. Stack: **Streamlit**, charts = **native Streamlit charts where they suffice + Plotly for the core layered chart**.

## 1. Design System

**Aesthetic:** Precision-instrument / track-timing. Warm paper ground, near-black ink, a single hot vermilion accent, and a dark "instrument-panel" sidebar. Editorial structure via monospace eyebrow labels (a stopwatch/scoreboard cue). Data-ink first; the accent is reserved almost entirely for the peak.

**Framework / theming:** Streamlit `.streamlit/config.toml` for base colours + a single injected CSS block (`dashboard/theme.py`) for fonts, variables, and component styling. Fonts loaded via Google Fonts `@import`.

```toml
# .streamlit/config.toml
[theme]
base = "light"
primaryColor = "#FF4D17"               # vermilion accent (peak / interactive)
backgroundColor = "#F4F2EC"            # warm paper
secondaryBackgroundColor = "#FFFFFF"   # cards / surfaces
textColor = "#14161B"
```

**Typography:** **Bricolage Grotesque** (display — page headings, wordmark), **IBM Plex Sans** (body/UI), **IBM Plex Mono** (data values, metric labels, eyebrow section labels, chart axis titles). The mono numerals deliberately evoke a stopwatch/scoreboard.

**Color palette (semantic):**
| Token | Hex | Use |
|-------|-----|-----|
| `paper` | `#F4F2EC` | page background (warm off-white) |
| `surface` | `#FFFFFF` | metric cards, tables |
| `ink` | `#14161B` | primary text, sidebar ground |
| `ink-soft` | `#60656F` | secondary text, axis/eyebrow labels |
| `line` | `#E5E0D6` | hairline borders |
| `accent` | `#FF4D17` | predicted peak line/marker, selected nav, button hover |
| `athlete` | `#16263A` | the subject athlete's observed points + fitted trajectory |
| `reference` | `#B7B0A2` | population average trajectory (dashed) |
| `band` | `rgba(183,176,162,0.22)` | percentile band fill |
| `peak-window` | `rgba(255,77,23,0.10)` | shaded vertical span for the peak window |

**Layout signatures:** dark sidebar with the `PEAK`**`PREDICTOR`** wordmark + nav; a slim top context bar (tagline + `DATA · <version>` pill) with a 2px ink rule; mono **eyebrow** labels above each section; bordered mono metric cards; ink buttons that flip to accent on hover.

**Spacing:** Streamlit default vertical rhythm; group related controls in `st.container`/cards on `surface`. Use `st.columns` for side-by-side layout, not custom CSS grids.

## 2. Layout

- **Global shell:** Streamlit **multipage app**. Left **sidebar** = persistent navigation + global controls; main panel = active page.
- **Sidebar contents:** app title; nav (Explore · Upload & Predict · Indicators); a small "Data version" caption showing the loaded artifact version + date (supports the operator/version-mismatch story); credential/user indicator.
- **Main panel:** page-specific. Standard pattern = a top context row (selected athlete / inputs summary) then a two-column body: **left = the performance chart** (the hero), **right = summary diagrams + prediction callout**. Tables/exports below.
- **Responsive:** desktop-first (≥1200px ideal). On narrower widths, the two-column body stacks (chart on top, summaries below) via `st.columns` reflow. Not optimized for phones.

## 3. Components

For each: purpose + states (default / loading / empty / error, plus hover/active where interactive).

### 3.1 Athlete search & filter (Explore)
- **Purpose:** find/select an existing athlete (name search + filters: event, sex, country).
- **States:** default (search box + filter widgets); loading (spinner while artifacts/index load); **empty** (no matches → message + "check spelling / adjust filters"); error (artifacts unavailable → see 3.9).

### 3.2 Athlete context header
- **Purpose:** show who/what is currently in view (name or "Uploaded athlete", event, sex, country, # seasons, age range, PB).
- **States:** default; sparse (badge "limited data" when `< MIN_POINTS`).

### 3.3 Core performance-over-time chart (Plotly) — the hero component
- **Purpose:** the central visual for both Explore and Upload. **Identical rendering in both** (single shared chart function).
- **Layers (z-order back→front):**
  1. Percentile **bands** for the event/sex population (`band`), nested (e.g., 10–90 then 25–75).
  2. Population **average trajectory** — dashed `reference` line.
  3. (Toggle) **similar-athlete** traces — thin `peer` lines.
  4. Athlete **observed results** — `athlete` markers.
  5. Athlete **fitted trajectory** — solid `athlete` line.
  6. **Peak window** — shaded vertical span (`peak-window`).
  7. **Predicted/estimated peak** — vertical `peak` line + marker + text label (age + level).
- **Axes:** X = age (years); Y = normalized performance score (higher = better, regardless of event direction). Secondary readout may show raw mark on hover.
- **Interactions:** hover tooltips (age, raw mark, normalized score, competition/wind if present); zoom/pan; legend toggles per layer (bands, peers, population).
- **States:** default; **loading** (skeleton/spinner); **insufficient-data** (`< MIN_POINTS` → render observed points only, no trajectory/peak, with an inline "insufficient data to fit a reliable peak" note); **low-confidence** (wide uncertainty or out-of-distribution → peak line dimmed + `warning` annotation); error.

### 3.4 Summary diagrams (native Streamlit charts)
- **Purpose:** athlete-parameter summaries (e.g., results-per-season bar, distribution of marks, progression rate, age histogram vs population).
- **Implementation:** `st.bar_chart` / `st.scatter_chart` / `st.area_chart`. No overlays/annotations needed here.
- **States:** default; empty (no parameters → hide the specific diagram, don't show an empty axis).

### 3.5 Prediction & uncertainty callout
- **Purpose:** state the headline result in words + numbers: predicted age of peak, peak window, and an explicit **uncertainty** (e.g., ± years / interval) + a one-line methodology/limitations link.
- **States:** default (confident); warning (low-confidence/extrapolation — `warning` styling, prominent caveat); blocked (insufficient data — show "cannot predict: need ≥ MIN_POINTS results").

### 3.6 Manual-entry form (Upload & Predict)
- **Purpose:** enter athlete results by hand (FR-010, FR-014).
- **Structure:** static fields (sex, event — constrained to supported v1 sprint events) + a dynamic results table (one row per result: date-or-age, mark, optional wind/competition) with add/remove-row controls; submit button.
- **States:** default; **inline field validation** (per-row errors in `error`; bad date/mark/units/out-of-range mark); **block-on-submit** when invalid or `< MIN_POINTS` rows; success (valid → triggers chart + callout); unsupported-event (event picker disables out-of-scope events with "not supported in this version").

### 3.7 Similar-athletes panel
- **Purpose:** list comparable athletes; each navigates to that athlete's Explore view; each also toggleable as a `peer` trace on the chart.
- **States:** default (list with key stats); empty (none found → hide panel with a small note); loading.

### 3.8 Indicators view components
- **Purpose:** present B's correlation/indicator report — ranked indicators with effect size + statistical support, a correlation matrix / feature-importance chart, and a literature-comparison note.
- **States:** default; empty/stale (no report in artifacts → empty state pointing to pipeline status); low-power (findings flagged low-confidence, not hidden).

### 3.9 Global empty / error states
- **No artifacts loaded:** full-page empty state explaining the dataset isn't available yet (maps to Journey 4 preconditions).
- **Incompatible artifact version:** blocking `error` banner — "dashboard expects feature schema vX; loaded artifact is vY" — refuse to render predictions (FR-013, AC-4.4). Never mispredict silently.
- **Out-of-group athlete:** show parameters with an "event not yet supported in this version" notice.

## 4. Screen-by-Screen Specs

### Screen: Explore  (Journey 1)
```
┌── sidebar ──┬───────────────────────── main ──────────────────────────┐
│ PeakPredict │  Search [__________]  Event[▾] Sex[▾] Country[▾]         │
│             │  ── Athlete: Jane Doe · 200m · F · GBR · 7 seasons ──    │
│ ● Explore   │ ┌─ performance over time ─────────┐ ┌ prediction ──────┐│
│ ○ Upload    │ │        ░░band░░                  │ │ Peak age: 25.4   ││
│ ○ Indicators│ │   ····athlete pts····  ╱‾●‾╲     │ │  ± 1.3 y         ││
│             │ │  ╱fitted╱        peak ▎  window  │ │ Window: 24–27    ││
│ data v2026. │ │ ─ ─ population avg ─ ─           │ │ ▸ methodology    ││
│ 06.14       │ └─────────────────────────────────┘ └──────────────────┘│
│ user: ●     │ ┌ summary diagrams ───────┐ ┌ similar athletes ────────┐│
│             │ │ [bar] results/season    │ │ • A. Smith  → open       ││
│             │ │ [hist] marks dist.      │ │ • L. Nguyen → open       ││
│             │ └─────────────────────────┘ └──────────────────────────┘│
└─────────────┴──────────────────────────── [Export CSV] ───────────────┘
```
- Key interactions: search/filter → select; hover/zoom/toggle on chart; click a peer to re-run; export.
- Responsive: right column (prediction/summaries) stacks under the chart below ~1000px.

### Screen: Upload & Predict  (Journey 2)
```
┌── sidebar ──┬───────────────────────── main ──────────────────────────┐
│ ○ Explore   │  Sex [F ▾]   Event [200m ▾]   (supported events only)   │
│ ● Upload    │ ┌ results (add ⊕ / remove ⊖) ───────────────────────┐   │
│ ○ Indicators│ │  date/age │ mark   │ wind │ comp     │            │   │
│             │ │  2021     │ 24.10  │ +0.8 │ regional │ ⊖          │   │
│             │ │  2022     │ 23.65  │ -0.2 │ national │ ⊖          │   │
│             │ │  …                                  ⊕ add row     │   │
│             │ └───────────────────────────────────────────────────┘   │
│             │  [ Predict ]   (disabled until ≥ MIN_POINTS valid rows)  │
│             │ ┌ same hero chart + prediction callout as Explore ─────┐ │
│             │ └──────────────────────────────────────────────────────┘ │
└─────────────┴───────────────────────────────────────────────────────────┘
```
- Validation: per-row inline errors; submit blocked on invalid/too-few; out-of-distribution → warning-styled prediction.
- The result visualization is the **same hero chart component** as Explore (3.3) for consistency.

### Screen: Indicators  (Journey 3)
```
┌── sidebar ──┬───────────────────────── main ──────────────────────────┐
│ ○ Explore   │  Peak-performance indicators (event group: sprints)      │
│ ○ Upload    │ ┌ ranked indicators ──────┐ ┌ correlation matrix ──────┐│
│ ● Indicators│ │ feature   effect  supp. │ │ [heatmap]                ││
│             │ │ start age  +0.41   ***  │ │                          ││
│             │ │ progress.  +0.33   **   │ └──────────────────────────┘│
│             │ └─────────────────────────┘  Literature: sprint peak    │
│             │  ▸ methodology & limitations   ~25–26y (consistent)     │
└─────────────┴──────────────────────────── [Export CSV] ───────────────┘
```

## 5. Visual References

- Wireframes above are the v1 reference (ASCII; agent-implementable layout intent).
- The **chart color semantics in §1 are normative** — the hero chart must use exactly these roles so Explore and Upload are visually identical and peak/population/peer layers are unambiguous.
- No external brand assets. If higher-fidelity mockups are wanted later, generate them from these wireframes (e.g., a design tool) — not required for implementation.
- Out of scope for v1 visuals: dark mode, custom fonts/logos, mobile-optimized layouts, animations.
```
