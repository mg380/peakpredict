# Peak Performance Predictor — Implementation Plan

> Status: v1. The agent's marching orders. Builds on srs.md (module IDs A1–A4, B1–B10, C1–C6, common), ml-modeling-spec.md, user-journeys.md (AC IDs), design.md, business-requirements.md (FR/NFR IDs). Working branch: **develop**. Phase 0 (feasibility spike) is COMPLETE (feasibility-spike.md) — gate passed.

## 1. Principles

- All code is **modular** per srs.md §3 — one responsibility per module, cross-boundary data as dataclasses/pydantic (no loose dicts).
- Development proceeds in **discrete, independently testable phases** — no "later phase tests cover earlier code."
- The **three components stay decoupled**; they touch only through the data contracts (raw DuckDB store; versioned artifact bundle).
- The two **shared contracts that must never drift** — `common.normalization` and the **feature schema** — are owned by `common` and imported by B and C. Any change is a `schema_version` bump.
- **Security/foundation first:** secrets handling and the shared schemas/IO are Phase 1, before any component.
- Each phase ends with its **audit checklist** and a `/sf-commit` on `develop`, then a pause for manual testing.
- **File-size guideline:** target ≤ ~400 lines per module file; split if exceeded.

## 2. Phase Overview

| Phase | Focus | Modules | Depends On |
|-------|-------|---------|------------|
| 0 | Feasibility spike ✅ DONE | — | — |
| 1 | Foundation: monorepo, `common`, secrets, CI | common.{schemas, normalization, config, io, event_maps, logging}; repo scaffold | — |
| 2 | Component A — scraper → raw store | A1 session, A2 roster, A3 career, A4 runner | Phase 1 |
| 3 | Component B (data) — normalize→features | B1 normalize, B2 season_best, B3 trajectory, B4 labels, B5 features | Phase 1, 2 |
| 4 | Component B (model) — train→publish artifact | B6 model, B7 evaluate, B8 indicators, B9 aggregates, B10 publish | Phase 3 |
| 5 | Component C — dashboard | C1 loader, C6 auth, C5 charting, C2 explore, C3 predict, C4 indicators | Phase 1, 4 |
| 6 | Integration + end-to-end + manual test plan | all | Phases 2–5 |
| 7 | Final deep audit (`/sf-deepaudit`) | all | all |

## 3. Phase Details

### Phase 1: Foundation
**Objective:** A working monorepo skeleton with the shared contracts and secret handling that everything else imports — so components can be built independently without drift.

**Modules:**
- `pyproject.toml` + monorepo layout (`packages/{common,scraper,pipeline,dashboard}`, `data/{raw,processed,artifacts}`, `tests/`); editable installs.
- `common.config` — load secrets from gitignored `.secrets`/env (`TILASTOPAJA_USER/PASS`); never log secret values.
- `common.schemas` — dataclasses/pydantic for `PerformanceRow`, `UploadedAthlete/UploadedResult`, `PeakPrediction`, artifact `manifest`, `feature_schema` (srs.md §4.3–4.4).
- `common.normalization` — WA-points implementation + interface; pluggable z-score fallback (fitted params supplied later by B via the artifact). One implementation, imported by B and C.
- `common.io` — DuckDB connect + Parquet read/write helpers; raw-store DDL (srs.md §4.1).
- `common.event_maps` — event-id ↔ name (40=100m, 50=200m, 70=400m), supported-v1 set.
- `common.logging` — structured logging.
- CI: lint (ruff) + `pytest` on push. Refresh root `CLAUDE.md` to describe the new monorepo.

**Agent Team:** Coordinator + (a) schemas/normalization dev, (b) io/storage+DDL dev, (c) config/secrets+CI+scaffold dev. **Team Type:** small Agent Team (UAT) — schema shape is shared, brief coordination needed.

**Tests:** unit tests for normalization (round-trip mark↔score; monotonicity; higher=better), schema validation, io read/write round-trip, secret loader (missing-secret error path).

**Deliverables:** installable packages; `common` fully tested; CI green; CLAUDE.md updated.

**Audit Checklist:** [ ] ruff + type-check pass · [ ] no hardcoded secrets; secret loader never logs values · [ ] normalization is the single shared impl (no duplicate in B/C) · [ ] unit tests for all `common` modules · [ ] file-size ≤ threshold · [ ] CLAUDE.md matches reality.

### Phase 2: Component A — Scraper
**Objective:** A resumable, throttled scraper that fills the raw DuckDB store for the v1 sprint events, porting the **spike-proven** parsing.

**Modules:** A1 `session` (salvage retry/backoff + cookie-handoff; **no** httpbin mock code), A2 `roster`, A3 `career` (header-year + day/month → date; split mark cell on `\xa0` → mark/wind/record; round/pos/competition/location), A4 `runner` (CLI; `raw.scrape_state` resumability; throttle/backoff).

**Agent Team:** Coordinator + session/auth dev, roster dev, career-parser dev, runner/resumability dev. **Team Type:** Agent Team (UAT) — share the session interface and `raw.performance` schema.

**Tests:** parser unit tests against **saved spike HTML** (`spike/athlete_45032.html`) — assert ≥ N rows, correct date assembly, mark/wind split, event mapping; resumability test (rerun skips `done`); throttle/backoff unit test. (Live-site calls are manual, not in CI.)

**Deliverables:** `scraper run --events 40,50,70 --sexes 1,2 --resume` fills `raw.*`; idempotent.

**Audit Checklist:** [ ] parser tests pass on spike fixture · [ ] no mock/httpbin remnants, no divergent URLs · [ ] resumable + idempotent (NFR-004) · [ ] throttle/backoff present (NFR-001) · [ ] no secrets in logs · [ ] spec fidelity to srs.md A1–A4 · [ ] file-size.

### Phase 3: Component B (data) — normalize → features
**Objective:** Turn the raw store into clean, leakage-safe modeling tables + the published feature schema.

**Modules:** B1 normalize (uses `common.normalization`), B2 season_best (wind-legal ≤ +2.0; per-season best; join DOB w/ century-correct), B3 trajectory (quadratic/robust → vertex/window/level), B4 labels (validity filters, ml-spec §2), B5 features (early-career features; **leakage guard**; owns feature schema; multi-cutoff rows for forward validation).

**Agent Team:** Coordinator + normalize/season-best dev, trajectory/labels dev, features dev. **Team Type:** Agent Team (UAT) — feature schema must be agreed and will be consumed by C (Phase 5).

**Tests:** wind-legal filtering; season-best aggregation; trajectory vertex on a **synthetic ∪ and the Bolt fixture** (vertex ≈ spike's 25.3); label validity filters; **leakage test** (no feature uses data > cutoff age); DOB century-correction.

**Deliverables:** `processed/{season_bests,labels,features}.parquet`; documented feature schema in `common.schemas`.

**Audit Checklist:** [ ] leakage test passes (critical) · [ ] wind/DOB cleaning per spike reqs · [ ] trajectory reproduces spike sanity numbers · [ ] feature schema versioned · [ ] unit tests all modules · [ ] spec fidelity ml-spec §2–5 · [ ] file-size.

### Phase 4: Component B (model) — train → publish
**Objective:** Train the baseline-first predictor ladder, validate temporally, and publish the versioned artifact bundle (the B→C contract).

**Modules:** B6 model (B0 mean → B1 naive → **B2 MixedLM primary** → optional B3 LightGBM/Bayesian; select simplest-within-noise), B7 evaluate (split-by-athlete, forward-prediction truncation at 3/5/7 seasons; MAE/bias/interval calibration/baseline skill), B8 indicators (FR-006), B9 aggregates (population avg trajectory + percentile bands + similarity basis), B10 publish (assemble versioned bundle, srs.md §4.3).

**Agent Team:** Coordinator + model/ladder dev, evaluation/temporal-validation dev, indicators/aggregates dev, publish/artifact dev. **Team Type:** Agent Team (UAT).

**Tests:** baseline B0 reproduces population mean; **temporal split has no athlete leakage** (athlete in train XOR test); metric computation on a toy set; interval calibration sanity; artifact bundle round-trips (publish→load); population peak-age sanity vs literature (~25–26).

**Deliverables:** `data/artifacts/<version>/` complete bundle incl. validation + indicator reports; model selection documented.

**Audit Checklist:** [ ] no athlete leakage across split (critical) · [ ] baselines implemented and beaten-or-reported honestly · [ ] uncertainty calibrated + OOD/insufficient flags (AC-2.6) · [ ] artifact bundle matches §4.3 + `schema_version` · [ ] reproducibility metadata (NFR-005) · [ ] spec fidelity ml-spec §6–10 · [ ] file-size.

### Phase 5: Component C — Dashboard
**Objective:** The credential-gated Streamlit app delivering Explore, Upload & Predict, and Indicators against the published artifact.

**Modules:** C1 artifact_loader (verify schema compat; refuse incompatible — AC-4.4), C6 auth (credential gate, FR-015), C5 charting (**shared Plotly hero chart** — design.md §3.3; identical in Explore/Upload), C2 explore (Journey 1), C3 predict (manual form; validate FR-014; normalize via `common.normalization`; Journey 2), C4 indicators (Journey 3).

**Agent Team:** Coordinator + loader/auth dev, charting dev, explore dev, predict dev, indicators dev. **Team Type:** Agent Team (UAT) — all share the hero-chart component + feature schema.

**Tests:** loader rejects incompatible `schema_version`; upload validation (too-few-points blocks per AC-2.2/2.3; unsupported event; OOD flag AC-2.6); **normalization parity** test — C3 produces identical scores to B for the same mark (AC-2.5); hero-chart renders required layers (design.md §3.3); empty/error states.

**Deliverables:** `streamlit run packages/dashboard/app.py` serving all three pages, auth-gated, reading the artifact bundle.

**Audit Checklist:** [ ] normalization parity B↔C (critical) · [ ] schema-mismatch refusal works (AC-4.4) · [ ] all component states present (loading/empty/insufficient/low-confidence/error) · [ ] auth gate enforced · [ ] design-spec color semantics on hero chart · [ ] spec fidelity journeys+design · [ ] file-size.

### Phase 6: Integration + End-to-End + Manual Test Plan
**Objective:** Prove the full pipeline A→B→C end-to-end on the v1 sprint set and document a manual walkthrough.

**Work:** wire CLIs; run A (small `--limit`) → B → publish → C loads it; an automated E2E smoke test over a tiny fixture dataset; produce `manual-test-plan.md` (end-user walkthrough mapping to journeys/ACs); update `CLAUDE.md` with run instructions.

**Agent Team:** Coordinator + integration/E2E dev + manual-test-plan author. **Team Type:** subagents (self-contained).

**Audit Checklist:** [ ] E2E smoke test green on fixture · [ ] versioned artifact flows A→B→C · [ ] manual-test-plan covers every journey + unhappy paths · [ ] CLAUDE.md run instructions accurate.

### Phase 7: Final Deep Audit
**Objective:** Comprehensive review of the whole implementation against all specs. Run **`/sf-deepaudit`**. Block on critical findings (leakage, normalization drift, secret exposure, schema-compat); log warnings as follow-up issues.

## 4. Agent Team Strategy

- **UAT (coordinated)** for Phases 1–5: teammates share a contract (schema, session interface, hero chart, artifact bundle) and must negotiate it mid-build. **Subagents** for Phase 6 (self-contained).
- **Parallelism:** within a phase, module devs work in parallel against agreed interfaces; the coordinator owns the shared contract (schema/DDL/bundle/chart API) and integration. Across phases, build order is sequential by dependency (table §2), except Phase 5 (C) can begin its UI shell against a **mock artifact bundle** while Phase 4 finalizes the real one.
- **Team size:** 3–5 teammates per phase, ~5–6 tasks each, per workflow guidance.
- **Coordinator responsibilities:** enforce the no-drift contracts, run the per-phase audit checklist, gate the `/sf-commit`, and surface cross-phase risks (esp. leakage guards and normalization parity).

## 5. Cross-Cutting Definition of Done (every phase)
1. Modules match their srs.md responsibility + API. 2. Unit tests pass; phase is independently testable. 3. Audit checklist fully checked. 4. No secrets, no dead code, file-size respected. 5. Committed on `develop` via `/sf-commit`. 6. User manual-test confirmation before the next phase.
