# Peak Performance Predictor — Technical Specification (SRS + SAD)

> Status: v1. Implement against this document. Builds on business-requirements.md, user-journeys.md, design.md, ml-modeling-spec.md, feasibility-spike.md. References FR/NFR IDs from the BRD. Schemas are normative.

## 1. Tech Stack

| Layer | Choice | Justification |
|------|--------|---------------|
| Language | **Python 3.13** | Matches existing venv; ecosystem for scraping + DS + Streamlit. |
| Scraping | **Selenium** (login/JS) + **BeautifulSoup4** + **requests** | Proven in notebooks/spike; login is JS-driven (needs Selenium); page parse via bs4; requests where pages are static post-auth. |
| Storage | **DuckDB + Parquet** | Embedded analytical SQL over columnar files; fast trajectory/group-by queries; scales to 70k+ athletes; no server; native pandas/pyarrow. |
| Dataframes | **pandas + pyarrow** | Interop with DuckDB/Parquet and modeling libs. |
| Trajectory fit | **numpy / statsmodels** | Quadratic + robust/weighted fits (spike-validated). |
| Predictor (primary) | **statsmodels MixedLM** (hierarchical/partial-pooling) | Field-standard for sparse longitudinal data; native to the manual-entry reality; lightweight. |
| Predictor (optional) | **scikit-learn / LightGBM** (B3), **PyMC or Bambi** (full-Bayesian B2), **MAPIE** (conformal intervals) | Flexible alternatives + calibrated uncertainty; added only if they beat MixedLM on the temporal split. |
| Dashboard | **Streamlit** (multipage) + **Plotly** (hero chart) + native st charts (summaries) | Per design.md; free credential-gated hosting; fastest path for analysts/coaches. |
| Dashboard auth | **streamlit-authenticator** (or Streamlit Community Cloud viewer allow-list) | Credential-gates C (FR-015) without building an auth backend. |
| Config / secrets | **python-dotenv** (local `.env`), **st.secrets** (dashboard) | Keep source + auth secrets out of code/VCS. |
| Scheduling | **cron / launchd** invoking CLI entrypoints | A and B run locally on a schedule; no heavyweight orchestrator needed at v1 scale. |
| Packaging | **uv or pip + pyproject**, monorepo with editable installs | Reproducible env; shared `common` package. |
| Testing | **pytest** | Unit + contract tests per component. |

## 2. Architecture Overview

**Pattern:** three loosely-coupled components in one **monorepo**, communicating only through **data contracts** (DuckDB/Parquet stores + a versioned artifact bundle). Each component is independently runnable (FR-002, FR-007, AC-4.3). Coupling is by data, not execution; B's feature needs define A's collection scope (requirements flow B⇠A).

```
repo/
  packages/
    common/        # shared: schemas, normalization, config, io (DuckDB/Parquet helpers), event-id maps
    scraper/       # Component A — acquire raw data -> raw store
    pipeline/      # Component B — normalize, analyze, fit, train, publish artifacts
    dashboard/     # Component C — Streamlit app, consumes artifact bundle
  data/
    raw/           # DuckDB db + Parquet (A's output, B's input)
    processed/     # Parquet (B intermediate: season_bests, features, labels)
    artifacts/     # versioned artifact bundles (B's output, C's input)
  tests/
  pyproject.toml
```

**Data flow:**
`source site → [A] → data/raw (DuckDB+Parquet) → [B] → data/artifacts/<version>/ → [C] Streamlit`

**Why monorepo + shared `common`:** the **normalization function and feature schema must be identical** in B (training) and C (upload inference) — user-journeys AC-2.5. Putting them in `common` guarantees one implementation, no drift.

## 3. Module Definitions

### Component A — scraper
- **A1 `session`** — Responsibility: authenticated Selenium/requests session to the source. Public API: `login() -> Session`, `get_page(url, params) -> html`, `close()`. Deps: Selenium, requests, dotenv. Data: holds credentials in memory (from `.env`); owns no persistent data. Salvage: `sports_data_platform/core/session_manager.py` retry/backoff + cookie-handoff (strip all httpbin mock branches).
- **A2 `roster`** — Responsibility: scrape/refresh the athlete roster (PID, name, country, sex, URL, DOB, event participation) for in-scope events. Public API: `scrape_roster(events, sexes) -> rows`. Deps: A1, common.schemas. Data: writes `raw.athlete`.
- **A3 `career`** — Responsibility: scrape one athlete's career page (`db/at.php?Sex&ID`) → dated performances per event (the spike's logic). Public API: `scrape_career(pid, sex) -> list[PerformanceRow]`. Parsing rules: section-header year + row day/month → date; split mark cell on whitespace/`\xa0` → `mark|wind|record`; capture round/position, competition, location. Deps: A1, common. Data: writes `raw.performance`.
- **A4 `runner`** — Responsibility: orchestrate resumable, throttled, monitored bulk scrape; CLI. Public API: `run(--events --sexes --resume --limit --throttle)`. Resumability: a `raw.scrape_state` table tracks per-PID status (pending/done/failed); reruns skip `done` (FR-002, NFR-004). Throttle/backoff between athlete fetches (NFR-001).

### Component B — pipeline
- **B1 `normalize`** — Responsibility: raw marks → normalized performance score (WA points primary; within-(event,sex) z-score fallback). **Imported from `common.normalization`** so C reuses it. Public API: `normalize(mark, event, sex) -> score`, `inverse(...)`. Data: reads `raw.performance`.
- **B2 `season_best`** — Responsibility: filter wind-legal (≤ +2.0), aggregate to per-athlete-per-event-per-season best; join DOB; compute age. Public API: `build_season_bests() -> processed/season_bests.parquet`.
- **B3 `trajectory`** — Responsibility: per-athlete trajectory fit (quadratic/robust) → vertex (peak age), peak window, peak level. Public API: `fit(series) -> TrajectoryFit`. Used for labels and per-athlete description.
- **B4 `labels`** — Responsibility: apply label-validity filters (ml-spec §2) → training labels. Public API: `build_labels() -> processed/labels.parquet`.
- **B5 `features`** — Responsibility: early-career feature construction with **leakage guard** (features at cutoff use only data ≤ cutoff). Public API: `build_features(cutoff_policy) -> processed/features.parquet`. Owns the **feature schema** (the upload contract).
- **B6 `model`** — Responsibility: train the predictor ladder (B0 baseline → MixedLM → optional LightGBM/Bayesian), select simplest-within-noise, fit uncertainty. Public API: `train() -> Predictor`, `predict(features) -> PeakPrediction`.
- **B7 `evaluate`** — Responsibility: temporal forward-prediction validation (split-by-athlete, truncate horizons), metrics, calibration, baseline skill → validation report. Public API: `evaluate() -> ValidationReport`.
- **B8 `indicators`** — Responsibility: multivariate/correlation analysis of features vs peak age/level → indicator report with effect sizes + support (FR-006).
- **B9 `aggregates`** — Responsibility: per-(event,sex) reference population: average trajectory + percentile bands; similarity basis for "similar athletes".
- **B10 `publish`** — Responsibility: assemble a **versioned artifact bundle** (model, normalization, feature schema, aggregates, similarity, indicator report, validation report, manifest) → `data/artifacts/<version>/`; CLI orchestrates B1–B10 (FR-007, NFR-005).

### Component C — dashboard
- **C1 `artifact_loader`** — Responsibility: load an artifact bundle; verify **feature-schema compatibility**; refuse incompatible versions with explicit error (FR-013, AC-4.4). Public API: `load(version) -> Artifacts`. Shows loaded version in sidebar.
- **C2 `explore`** — Page: search/filter → athlete view (summary diagrams, hero chart with trajectory+peak+population overlay+peers, prediction callout, export). Implements Journey 1.
- **C3 `predict`** — Page: manual-entry form (sex/event + results rows), validation (FR-014), normalize via `common.normalization`, run predictor, render same hero chart. Implements Journey 2.
- **C4 `indicators`** — Page: render indicator + validation reports. Implements Journey 3.
- **C5 `charting`** — Responsibility: the **shared hero-chart function** (Plotly) used by C2 and C3 — identical rendering (design.md §3.3). Public API: `hero_chart(athlete_series, fit, prediction, population, peers) -> fig`.
- **C6 `auth`** — Responsibility: credential gate (streamlit-authenticator/st.secrets). Implements FR-015/NFR-002.

### Shared — common
- `schemas` (dataclasses/pydantic for raw rows, feature schema, prediction, manifest), `normalization`, `config`, `io` (DuckDB/Parquet open/read/write helpers), `event_maps` (event-id ↔ name; spike: 40=100m,50=200m,70=400m), `logging`.

## 4. Data Model

### 4.1 Raw store (DuckDB; A owns)
```sql
CREATE TABLE raw.athlete (
  pid INTEGER PRIMARY KEY, name TEXT, country TEXT, sex SMALLINT,    -- 1=M,2=F
  dob DATE, url TEXT, scraped_at TIMESTAMP
);
CREATE TABLE raw.event (event_id TEXT PRIMARY KEY, name TEXT, indoor BOOLEAN);
CREATE TABLE raw.performance (
  perf_id BIGINT PRIMARY KEY, pid INTEGER REFERENCES raw.athlete(pid),
  event_id TEXT, indoor BOOLEAN, perf_date DATE,
  mark_raw TEXT, mark DOUBLE, wind DOUBLE, record_flag TEXT,        -- parsed from mark cell
  round_pos TEXT, competition TEXT, location TEXT, scraped_at TIMESTAMP
);
CREATE TABLE raw.scrape_state (                                      -- resumability (NFR-004)
  pid INTEGER PRIMARY KEY, status TEXT, attempts INTEGER, last_error TEXT, updated_at TIMESTAMP
);
```

### 4.2 Processed (Parquet; B owns)
- `season_bests.parquet`: `pid, event_id, sex, season(year), age, mark, wind, score(normalized)`; wind-legal only.
- `labels.parquet`: `pid, event_id, sex, peak_age, peak_window_lo, peak_window_hi, peak_score, n_points, span_years`.
- `features.parquet`: `pid, event_id, sex, cutoff_age, <feature columns per ml-spec §5>`. (Multiple cutoff rows per athlete for the forward-prediction protocol.)

### 4.3 Artifact bundle (B→C contract; `data/artifacts/<version>/`)
```
manifest.json          # version, created_at, code_commit, data_snapshot, schema_version, event group, metrics summary
predictor.pkl          # trained model(s) + trajectory fitter
normalization.json     # tables/params for common.normalization (identical to B's)
feature_schema.json    # exact upload fields/units/constraints (the C3 contract)
aggregates.parquet     # per (event,sex): avg trajectory + percentile bands
similar_index.parquet  # similarity basis
indicators.json        # FR-006 report
validation.json        # metrics, calibration, baseline skill
```

### 4.4 Feature/Upload schema (C3 manual-entry contract; from `common.schemas`)
```python
@dataclass
class UploadedResult:   # one manually-entered row
    date_or_age: str    # ISO date or integer age
    mark: float
    wind: float | None = None
    competition: str | None = None
@dataclass
class UploadedAthlete:
    sex: int            # 1|2
    event_id: str       # must be in supported v1 set {40,50,70}
    results: list[UploadedResult]   # len >= MIN_POINTS after validation
@dataclass
class PeakPrediction:
    peak_age: float; interval_lo: float; interval_hi: float
    peak_score: float; window_lo: float; window_hi: float
    confidence: str     # ok|low|insufficient|out_of_distribution|unsupported_event
```

## 5. Interface Contracts (CLI + internal)

- **A CLI:** `scraper run --events 40,50,70 --sexes 1,2 [--resume] [--limit N] [--throttle S]` → writes raw store; idempotent; resumable.
- **B CLI:** `pipeline build [--from raw] [--publish]` → runs B1–B10, writes processed + artifact bundle; can run standalone on existing raw store (AC-4.3).
- **C entry:** `streamlit run packages/dashboard/app.py` → loads latest compatible artifact (C1); auth-gated.
- **A→B contract:** raw DuckDB schema (§4.1). **B→C contract:** artifact bundle (§4.3) + `schema_version`. C refuses bundles whose `schema_version` it doesn't support (AC-4.4) — no silent misprediction.
- **Error/confidence semantics** (PeakPrediction.confidence): `insufficient` (< MIN_POINTS), `unsupported_event`, `out_of_distribution`, `low` (wide interval/extrapolation), `ok`.

## 6. Integration Points

- **Source site (tilastopaja.info)** — login `login.php` (fields `user`,`password`, button value `Login`); roster `db/alltfull.php?Ind&Event&Sex&area=&All=0&Age=99`; athlete `db/at.php?Sex&ID`. Failure modes: auth change, selector/structure change, rate-limit/block → A fails loudly with diagnostics; B is **not** run on incomplete data (AC-4.5). Etiquette: throttle, backoff, resumable (NFR-001).
- **Streamlit hosting (Community Cloud)** — deploys C from a Git repo; HTTPS; viewer auth/allow-list. Artifact transfer (local B → cloud C): publish the versioned bundle to the dashboard's deploy repo (committed `data/artifacts/<version>/`) **or** an object store the app reads at startup. v1 default: committed bundle in the deploy repo (simplest; bundle is non-sensitive).

## 7. Security Architecture

- **Trust boundaries:** A holds **source credentials** (sensitive) — local `.env` only, **gitignored**, never in code. C holds **dashboard auth credentials** — `st.secrets`, never in repo. The scraped data is **non-sensitive public results** (BRD C-2), so no encryption-at-rest requirement.
- **Secrets management:** `.env` (A/B local), `st.secrets` (C). **Cleanup required (security debt in existing assets):** the working notebooks hardcode a plaintext source password and `sports_data_platform/.env` is committed — these must be removed/rotated and `.env` gitignored before any push.
- **In transit:** HTTPS to source and to the Streamlit app.
- **AuthN/Z:** C is credential-gated (FR-015); single role (authorized viewer) in v1; no write-back of uploaded athletes (OS-5).
- **Data flow sensitivity:** only credentials are sensitive; they never leave the local machine or `st.secrets`.

## 8. Non-Functional Requirements (Technical)

- **NFR-001 (etiquette):** A throttles (configurable delay) + exponential backoff; one career fetch at a time by default.
- **NFR-003 (scale):** DuckDB/Parquet + per-(event,sex) partitioning scale to 70k athletes / 72 events with no schema change; v1 limited to event set {40,50,70} by config only.
- **NFR-004 (reliability):** `raw.scrape_state` + idempotent upserts → resumable, no partial-corruption; B refuses empty/incomplete raw input.
- **NFR-005 (reproducibility):** every artifact bundle records `code_commit` + `data_snapshot` + `schema_version`; same inputs reproduce outputs.
- **NFR-006 (transparency):** validation + indicator reports shipped in the bundle and surfaced in C4.
- **NFR-007 (portability):** components depend only on data contracts, not each other's runtime → A/B can move to an always-on host and C between hosts without interface change.
- **Performance:** B full rebuild on the v1 sprint set should complete in minutes on a laptop; C interactive actions < ~1s against loaded artifacts (artifacts pre-aggregated, no live training).
- **Monitoring/logging:** structured logging per component; A logs per-PID progress + failures; B logs stage timings + metric summary.

## 9. Environment & Deployment

- **Dev:** monorepo, single virtualenv, editable installs of all packages; `.env` for source creds; pytest.
- **A + B (local/prod):** run via CLI; scheduled by cron/launchd; outputs to `data/`.
- **C (prod):** Streamlit Community Cloud deploying from the dashboard repo with the committed artifact bundle + `st.secrets` auth.
- **CI (lightweight):** lint + pytest on push (no heavy infra at v1).
- **Artifact promotion:** B publishes `<version>/`; promoting = pointing/committing that version to the dashboard deploy repo; C auto-loads the latest compatible version.

## 10. Constraints & Decisions (ADR-style)

- **D-1 Monorepo + shared `common`** — guarantees identical normalization/feature schema across B and C (kills drift). Trade-off: one repo to deploy three things; mitigated by component-scoped CLIs.
- **D-2 DuckDB + Parquet** over SQLite/HDF5 — analytical speed + scale + pandas interop.
- **D-3 Build fresh, salvage patterns** — notebooks/spike are the real-site source of truth; reuse session/retry structure from `sports_data_platform`, discard mock/httpbin code and divergent URLs/selectors.
- **D-4 MixedLM as primary predictor; RNN exploratory** — matches sparse manual-entry inputs; advanced models must beat it on the temporal split (ml-spec §6).
- **D-5 Events modeled independently in v1**; event-group set is config (IS-6) → expansion without redesign.
- **D-6 Committed artifact bundle for hybrid transfer** — simplest local-B → cloud-C handoff; data is non-sensitive so committing is acceptable. Revisit (object store) if bundles grow large.
- **D-7 Naming/patterns:** snake_case modules; one public entrypoint per module; dataclasses/pydantic for all cross-boundary data (no loose dicts across contracts); abstract base scraper + concrete per-page scrapers (Template Method, salvaged).
```
