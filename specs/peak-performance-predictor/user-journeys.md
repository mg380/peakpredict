# Peak Performance Predictor — User Journeys

> Status: DRAFT v1 (Step 2 of newbuild workflow). Builds on business-requirements.md. References FR/NFR IDs from that document. Journeys are written for agent implementation: numbered steps, explicit empty/error states, testable acceptance criteria.

## Actors

- **Analyst (sports scientist)** — High technical proficiency. Goals: explore correlations, inspect methodology, compare athletes against the reference population, export data, judge whether a prediction is trustworthy. Wants transparency and statistical detail.
- **Coach / talent scout** — Moderate technical proficiency. Goals: enter a developing athlete's results by hand and get an actionable, interpretable projection of when they will peak. Tolerant of sparse data; wants clarity over depth.
- **Operator (project owner)** — High technical proficiency. Goals: run the scraping (A) and analysis (B) components on a schedule, publish fresh artifacts to the dashboard (C), and keep the system healthy. Not a dashboard end-user.

Shared notes:
- The dashboard (C) is **credential-gated** (FR-015): all dashboard journeys assume an authenticated, authorized user.
- v1 covers the **sprints** event group (100/200/400m, both sexes). Athletes/events outside this group are "out-of-scope" states the UI must handle gracefully.
- `MIN_POINTS` = the minimum number of dated results required to attempt a prediction. Exact value is set by the ML Modeling Spec (Step 4.5); journeys treat it as a defined threshold.

---

## Journey 1: Explore an existing athlete

**Actor:** Analyst or Coach
**Goal:** View an existing athlete's parameters, statistics, and performance-over-time with the fitted trajectory, predicted peak, and population context.
**Preconditions:** Artifacts published by B are loaded in C (FR-013); the athlete exists in the processed dataset.

### Happy Path
1. User opens the dashboard and authenticates.
2. User lands on the Explore view with a search/filter control (by name; filters for event, sex, country).
3. User searches for an athlete and selects a result.
4. System displays the athlete's **summary diagrams** (parameters: sex, event(s), country, age, number of seasons, personal best, etc.) (FR-011).
5. System displays a **performance-over-time graph**: the athlete's normalized results as points, the **fitted trajectory curve**, and the **estimated peak** (age + level) marked (FR-009, implements FR-005).
6. System overlays the **reference population**: the event/sex average trajectory and percentile bands (implements the population-overlay requirement).
7. System shows a **"similar athletes"** panel listing comparable athletes from the dataset, each linkable to their own Explore view.
8. System displays the prediction's **uncertainty** and a short methodology/limitations note (FR-012, NFR-006).

### Alternate Paths
- **Filter-first browse:** If the user applies filters without a name search, the system lists matching athletes ranked by a sensible default (e.g., data completeness or peak level) for selection.
- **Follow a peer:** Selecting a "similar athlete" (step 7) re-runs the journey for that athlete.
- **Export:** Analyst exports the displayed athlete's processed data and fitted-trajectory values (CSV).

### Unhappy Paths / Edge Cases
- **No match:** Search returns nothing → empty state with suggestions ("check spelling; try filters").
- **Sparse athlete:** Athlete has `< MIN_POINTS` results → show available points and parameters, but display "insufficient data to fit a reliable trajectory/peak" instead of a curve.
- **Out-of-group athlete:** Athlete exists in raw data but outside the v1 sprint group → show parameters with an "event not yet supported in this version" notice.
- **First load / empty artifacts:** No artifacts loaded yet → global empty state explaining the dataset is not yet available (see Journey 4).

### Acceptance Criteria
- AC-1.1: Given a valid athlete, the performance graph renders points + fitted trajectory + marked peak.
- AC-1.2: The reference population average trajectory and percentile bands are visibly overlaid for the athlete's event/sex.
- AC-1.3: A "similar athletes" panel lists ≥1 comparable athlete (when any exist) and each is navigable.
- AC-1.4: For an athlete with `< MIN_POINTS` results, no trajectory/peak is drawn and an explicit insufficient-data message is shown.
- AC-1.5: Prediction uncertainty is displayed alongside any peak estimate.

---

## Journey 2: Upload & Predict (manual form entry)

**Actor:** Coach / talent scout (also Analyst)
**Goal:** Manually enter a developing athlete's results and receive a visualized prediction of when their peak performance will occur.
**Preconditions:** Authenticated; artifacts (model + feature schema + reference population) loaded (FR-008, FR-013).

### Happy Path
1. User opens the **Upload & Predict** view.
2. User selects static attributes: **sex** and **event** (constrained to the supported v1 sprint events).
3. User enters the athlete's **results history** via a form, **one row per result**: date (or age) + mark; optional wind/competition. Rows can be added/removed.
4. User submits the form.
5. System **validates** the input (FR-014): parses marks/dates, checks units, checks `≥ MIN_POINTS` rows, flags impossible values.
6. System **normalizes** the entered marks using the same scoring method as B (consistency requirement) and fits the athlete's trajectory.
7. System runs the **published model** to predict **age of peak** and the **peak window**, with uncertainty.
8. System renders the **performance-over-time graph**: entered points, fitted trajectory, and a clearly-labeled **predicted peak line/marker**, overlaid with the event/sex reference population (FR-010, FR-012).
9. System renders **summary diagrams** of the entered athlete's parameters (FR-011).
10. (Optional) User adjusts a row and re-submits to see the prediction update.

### Alternate Paths
- **Prefill from existing athlete:** User starts from an existing athlete (Journey 1) as a template, then edits — useful for "what-if" exploration.
- **Multiple events:** If the athlete has results in more than one supported sprint event, the user may enter each; system predicts per event (v1 keeps events independent).

### Unhappy Paths / Edge Cases
- **Too few points:** `< MIN_POINTS` valid rows → block prediction; explain the minimum and show the entered points only.
- **Malformed entry:** Bad date/mark/units → inline field-level errors; submission blocked until resolved.
- **Out-of-range mark:** Mark implausible for the event (e.g., 4s 100m) → warning prompting correction.
- **Unsupported event:** User picks/needs an event outside v1 → disabled with "not supported in this version" messaging (no silent wrong prediction).
- **Out-of-distribution athlete:** Inputs far outside training distribution → prediction shown with a prominent low-confidence / extrapolation warning.
- **Flat/declining-only history:** No improvement trend → system still fits but flags that a peak may be unidentifiable from the data.

### Acceptance Criteria
- AC-2.1: The form accepts ≥1 result row, supports add/remove, and enforces supported sex+event.
- AC-2.2: Submitting `< MIN_POINTS` valid rows yields a clear blocking message, not a prediction.
- AC-2.3: Invalid rows produce field-level validation errors (FR-014) and block submission.
- AC-2.4: A valid submission renders the predicted-peak visualization with the reference-population overlay and an uncertainty indication (FR-010, FR-012).
- AC-2.5: Entered marks are normalized by the same method B uses (no divergence between explore and upload normalization).
- AC-2.6: Out-of-distribution / unsupported-event inputs are flagged, never silently predicted.

---

## Journey 3: Review peak-performance indicators

**Actor:** Analyst
**Goal:** Inspect which characteristics/indicators correlate with peak timing and peak level, with their statistical support.
**Preconditions:** Authenticated; B has published the indicator/correlation report (implements FR-006).

### Happy Path
1. User opens the **Indicators** view.
2. System displays the identified indicators of peak performance (e.g., correlations of features with age-of-peak and peak level), each with effect size and statistical support.
3. System shows supporting visualizations (e.g., correlation matrix / ranked feature importance).
4. System notes how findings compare to literature benchmarks (e.g., sprint peak ~25–26y) (NFR-006).
5. (Optional) Analyst exports the indicator table.

### Unhappy Paths / Edge Cases
- **Stale/empty report:** No indicator report in current artifacts → empty state pointing to pipeline run status.
- **Low-power findings:** Indicators computed on insufficient data are labeled low-confidence rather than omitted silently.

### Acceptance Criteria
- AC-3.1: The view lists indicators with effect size and a measure of statistical support.
- AC-3.2: At least one supporting visualization renders.
- AC-3.3: Findings are accompanied by a methodology/limitations note.

---

## Journey 4: Operator pipeline run (A → B → publish)

**Actor:** Operator
**Goal:** Refresh data and model so the dashboard serves up-to-date artifacts. The three components run **independently** but in output order.
**Preconditions:** Operator has authorized source access and runs A and B locally (C-1); the components are runnable on demand and on a schedule.

### Happy Path
1. Operator runs **A (Scraping)**: it scrapes new/updated source data for the in-scope events and writes/extends the **raw data store** (FR-001, FR-002). Long runs are resumable (NFR-004).
2. Operator (or scheduler) runs **B (Analysis)**: it reads the raw store, normalizes, runs analysis, fits trajectories, (re)trains the model, and **publishes versioned artifacts** + feature schema + reference-population aggregates + indicator report (FR-004–FR-008, NFR-005).
3. Operator publishes/points **C (Dashboard)** at the new artifact version.
4. Dashboard users now see refreshed data, model, and indicators.

### Alternate Paths
- **B-only re-run:** Operator re-runs B against the existing raw store (e.g., to change normalization or model config) without re-scraping — valid because processes are independent.
- **A scope extension:** If B reports required features missing, operator extends A's collection scope/sources (IS-7, A-6), re-runs A, then B.

### Unhappy Paths / Edge Cases
- **Scrape interrupted:** A fails/crashes mid-run → resumes without corrupting the raw store; partial data is consistent (NFR-004).
- **Source structure changed:** Selectors/auth break → A fails loudly with diagnostics; B is not run on incomplete data.
- **Artifact version mismatch:** C points at an artifact whose feature schema differs from what C expects → C refuses to load that version with a clear error rather than mispredicting (FR-013).
- **Empty raw store:** B run before any successful A run → B exits with a clear "no input data" error.

### Acceptance Criteria
- AC-4.1: A can be re-run and resumes/extends the raw store idempotently without full re-scrape.
- AC-4.2: B produces a **versioned** artifact set traceable to the input data snapshot and code (NFR-005).
- AC-4.3: B can run standalone against an existing raw store with no scraping.
- AC-4.4: C loads only a compatible artifact version and rejects incompatible ones with an explicit error.
- AC-4.5: A failure in A or B does not leave the raw store or artifacts in a corrupt/partial state that C would serve.
```
