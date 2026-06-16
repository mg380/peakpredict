# Peak Performance Predictor — Phase 0 Feasibility Spike (Findings)

> Status: COMPLETE. Hands-on spike run during planning to gate the rest of the build. Evidence files in `spike/` (`spike_scrape.py`, `athlete_45032.html`, `athlete_45032_parsed.json`). Outcome: **GATE PASSED** — the build may proceed; data-dependent specs below are now grounded in real data, not assumptions.

## Method
- Analyzed existing `Data/` HDF5 (cross-sectional) + ran a single-athlete live scrape of the source athlete page (`db/at.php?ID=...`) after authenticated login, porting the proven notebook-2 parsing logic. Subject: PID 45032 (Usain Bolt).

## Findings

### F1 — Existing data is cross-sectional, NOT longitudinal
- `top_performance_*.h5` files contain **exactly one row per athlete** (their single all-time best mark). Confirmed: 7948 rows / 7948 unique PIDs in 100m male.
- Therefore the existing data **cannot** support per-athlete trajectory/peak modeling. The per-athlete career scrape is the **critical path** (this is the unfinished notebook-2 work).
- Existing data IS usable for: (a) the **roster** (PID → name, country, sex, URL, event participation), (b) **DOB** per athlete, (c) a cross-sectional age-at-best sanity signal.

### F2 — Date / DOB quality is good
- `Date of Event` parses 100% (`%d %b %Y`). `Athlete Date of Birth` (`%d %b %y`) missing only **2–12%** depending on event (2-digit year needs century correction: if DOB > event date, subtract 100y).
- Cross-sectional age-at-personal-best medians: 100m M 22.4, 200m M 22.4, 400m M 22.6, 100m F 23.1, 200m F 23.1, 400m F 24.5. These run *younger* than the literature peak (~25–26) because age-at-single-best is a **biased proxy**, not the trajectory vertex — reinforcing the need for longitudinal modeling.

### F3 — Per-athlete career time-series IS obtainable (the gate)
- Login succeeded; athlete page returned 122 KB; **270 dated performance rows** parsed for Bolt spanning **2001→2017 (17 distinct years)** across events 39,40,48,49,50,60,70,560,580.
- Event ID map (from `Events.h5`): **40 = 100m, 50 = 200m, 60 = 300m, 70 = 400m** (v1 sprint events present and dense).
- Row structure: section header rows carry the **year**; data rows carry `[mark(+wind/record), round/position, competition, location, day-month]`. Full date = header-year + row day/month.

### F4 — Peak is detectable and labelable, and matches reality
- Bolt 100m: **85 dated marks**, ages 20.9–31.0. Season-best trajectory: `10.03 → 9.58 (age 22) → ~9.6–9.8 plateau → 9.95`, a textbook ∪.
- Quadratic fit `mark = a·age² + b·age + c` gave a true minimum (a>0), **vertex = 25.3y**; single fastest mark (9.58 WR) at age 23.0.
- The **23.0 (instantaneous) vs 25.3 (vertex) gap** is a genuine finding: the peak *definition* materially changes the label. The literature uses the trajectory vertex / sustained window; we adopt that, explicitly.

### F5 — Primary source is sufficient for v1
- The athlete page supplies marks, dates, wind, record flags, competition, location; DOB joins from the roster by PID. **No supplementary sources are required for sprint trajectory modeling in v1.** (IS-7 / A-6 remain available but are not triggered.)

## Data-cleaning requirements surfaced (inputs to the ML & technical specs)
1. **Mark parsing:** split the mark cell on whitespace/`\xa0` into `mark | wind | record-flag`; strip trailing `+` (hand-time/annotation); coerce to float.
2. **Wind legality:** wind is available per row — **filter or flag wind-aided marks (> +2.0 m/s)** for fair trajectory fitting.
3. **Season-best aggregation:** model on **per-season-best** (or top-k mean) rather than all rounds/heats — removes heat/semi noise and yields a far cleaner trajectory.
4. **Date assembly:** combine section-header year with row day/month.
5. **Round/position codes** (`1r1`,`1h1`,`1s2`): normalize/drop for modeling.
6. **DOB:** join from roster by PID; handle 2–12% missing and 2-digit-year century correction.

## Decisions locked by the spike
- D1: v1 models **season-best, wind-legal** performances per athlete per sprint event.
- D2: **Peak = vertex of the fitted performance-vs-age trajectory** (plus a peak *window*), NOT the single best mark. (Refined further in the ML Modeling Spec.)
- D3: Scraper (A) target = athlete career pages keyed by PID from the roster; primary source only for v1.
- D4: No supplementary data sources for v1.

## Residual risks (carried forward)
- R-a: Lower-tier / younger athletes will have far fewer points than Bolt → trajectory fits unstable; sets the `MIN_POINTS` threshold question for the ML spec.
- R-b: Scraping ~thousands of sprint athletes is slow/credentialed → A must be throttled, resumable, monitored (already in NFR-001/004).
- R-c: Survivorship bias — **accepted by design**. Intended subject population is semi-pro/pro athletes (matches the DB population); the tool projects an up-and-coming pro's peak from historical pro data. Stated as an applicability boundary, not a defect.
