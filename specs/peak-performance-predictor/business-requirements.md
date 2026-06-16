# Peak Performance Predictor — Business Requirements

> Status: DRAFT v1 (Step 1 of newbuild workflow). Anchor document — all later artifacts trace back here. No technology/architecture decisions are binding in this document except where explicitly recorded as a business constraint.

## 1. Purpose

Build a system that (a) collects athletics performance data, (b) identifies which athlete characteristics and career signals correlate with peak performance, (c) predicts **when** an athlete's peak performance will occur, and (d) presents this through an interactive dashboard. The goal is a methodologically sound, working end-to-end tool that helps practitioners and researchers reason about athlete development trajectories.

The system exists because age-of-peak-performance is well-studied in sports science in aggregate (event/sex averages), but there is no accessible tool that takes an **individual** athlete's partial career history and projects their personal peak timing against a large empirical reference population.

## 2. Stakeholders

| ID | Stakeholder | Role / Interest |
|----|-------------|-----------------|
| ST-1 | Sports scientists / analysts | **Primary user.** Need methodological transparency, statistical detail, exportable data, defensible predictions. |
| ST-2 | Coaches / talent scouts | **Primary user.** Need actionable, decision-oriented output; tolerant of sparse input data for a prospect. |
| ST-3 | Operator (project owner) | Runs and maintains the data pipeline; holds the data-source subscription; owns deployment. |
| ST-4 | Data source(s) (tilastopaja.info, possibly others) | Provider(s) of source data. Operator **has permission** to scrape and use/display the data; data is public athletics results freely available elsewhere. A dependency and an etiquette constraint, not a privacy constraint. |

Decision authority: ST-3 (project owner).

## 3. Business Objectives

- BO-1: Produce, for a chosen event group, a model that predicts an individual athlete's **age of peak performance** (and peak-performance window) from their partial career history.
- BO-2: Surface the **characteristics/indicators** most correlated with peak timing and peak level, grounded in both literature and our own data analysis.
- BO-3: Deliver a working end-to-end product: periodic data pipeline → trained model artifact → interactive dashboard, demonstrably functioning on v1.
- BO-4: Report predictive performance **honestly** (accuracy, uncertainty, limitations) rather than overclaiming.

Success is defined by **methodological soundness + a working end-to-end system with transparent accuracy reporting**, NOT by hitting a fixed accuracy threshold (see §8).

## 4. Scope

### 4.1 System structure (three semi-independent components)
The system is **three loosely-coupled components** connected only by **data contracts** (defined schemas / published artifacts). Each component's *process* runs independently; each *consumes the output* of the previous one. Coupling is by data, not by execution.

- **A — Scraping** (batch, scheduled, local): acquires raw athlete/event/performance data from the source(s) and writes it to a defined **raw data store**. Does not analyze or model.
- **B — Analysis Pipeline** (batch, scheduled): consumes the raw data store → clean/normalize → multivariate/correlation analysis → fit trajectories → (re)train and update model weights → publish the **artifact interface** (trained model + processed datasets + feature schema + indicator report).
- **C — Dashboard** (interactive): consumes the published artifact interface only. Never scrapes, never trains.

**Contract directions:**
- A → B: the raw data store (schema = contract).
- B → C: the artifact interface (model + processed data + feature schema = contract).
- B ⇠ A (requirements, not execution): the analysis defines **what** A must collect; if the current source is insufficient for the required features, A's collection scope (and possibly its sources) expands to satisfy B's needs.

### 4.2 Dashboard capabilities
- DC-1 **Explore**: view an existing athlete's performance indicators, statistics, parameters, and performance-over-time, including the fitted trajectory and peak estimate.
- DC-2 **Upload & Predict**: a user uploads their own athlete's information; the system visualizes that athlete's **predicted peak performance** (performance-over-time graph with a predicted peak line), plus summary diagrams of the athlete's parameters.

### In Scope (v1)
- IS-1: **One event group** (default: sprints — 100m / 200m / 400m, both sexes), with per-event performance normalization.
- IS-2: Scraping of per-athlete **career time-series** (the data needed for modeling), which does not yet exist.
- IS-3: Multivariate/correlation analysis to identify indicators of peak performance.
- IS-4: A predictive model for age-of-peak (baseline-first; advanced models must justify themselves — see ML Modeling Spec, Step 4.5).
- IS-5: Dashboard with the two capabilities above (DC-1, DC-2).
- IS-6: Architecture that is **event-group-agnostic**: adding further event groups later is configuration/data, not a rewrite.
- IS-7: Scraper (A) designed so its collection scope can expand (more fields, and if needed **additional sources**) to satisfy the analysis component's (B) feature requirements; whether supplementary sources are actually needed is decided after the feasibility spike.

### Out of Scope (v1)
- OS-1: All 72 events / full population at launch (deferred; enabled by IS-6).
- OS-2: Field-event cross-unit normalization beyond the chosen v1 group (deferred until a field group is added).
- OS-3: An open, un-credentialed public data portal. Data display happens inside the **credential-gated** dashboard only (not a privacy requirement — a scope boundary).
- OS-4: Real-time / on-demand scraping triggered by dashboard users (pipeline is scheduled and separate).
- OS-5: Multi-user accounts, role management, or write-back of user-uploaded athletes into the shared dataset.
- OS-6: Injury, physiological, or training-load data not available from the source (unless trivially derivable).

## 5. Requirements

### Functional Requirements

**Component A — Scraping**
- FR-001: A shall scrape athlete rosters, event metadata, and per-athlete career performance histories for the in-scope event group, writing them to the raw data store under a defined schema.
- FR-002: A shall run periodically to fetch new/updated data and incrementally extend the raw store without full re-scrape where avoidable (resumable, idempotent).
- FR-003: A's collection scope shall be configurable (fields, events, and source endpoints) so it can be extended to meet B's feature requirements without redesign.

**Component B — Analysis Pipeline**
- FR-004: B shall normalize raw performances into a cross-event-comparable performance score per the adopted scoring method.
- FR-005: B shall compute, per athlete, a fitted performance-vs-age trajectory and derived peak estimates.
- FR-006: B shall run a correlation/multivariate analysis and publish the identified indicators of peak performance.
- FR-007: B shall (re)train the predictive model on the updated dataset and publish a **versioned** model artifact plus its **feature schema** and processed datasets (the artifact interface for C).
- FR-008: B shall define and document the feature schema that user-uploaded athletes (in C) must conform to.

**Component C — Dashboard**
- FR-009: C shall let a user browse and select an existing athlete and view their indicators, statistics, parameters, and performance-over-time with fitted trajectory and peak estimate (DC-1).
- FR-010: C shall accept a user-uploaded athlete record conforming to the published feature schema and return a predicted peak-performance visualization (DC-2).
- FR-011: C shall present athlete-parameter **summary diagrams**.
- FR-012: C shall display predictions with their **uncertainty** and a clear statement of model limitations.
- FR-013: C shall load only published artifacts and shall function without access to A, B, or the source site.
- FR-014: C shall validate and give clear feedback on malformed or insufficient user-uploaded athlete data (e.g., too few seasons to predict).
- FR-015: C shall be access-controlled (credential-gated) while remaining reachable by authorized users over the web.

### Non-Functional Requirements
- NFR-001 (Etiquette): A (scraping) shall operate under the operator's authorized access and shall respect rate limits / scraping etiquette (throttled, backoff, resumable). See §6.
- NFR-002 (Access control): C (dashboard) shall be credential-gated so only authorized users can access it; this is an access requirement, not a data-privacy requirement (the data is public athletics results the operator is permitted to display).
- NFR-003 (Scale): The data model, scraper, and analysis pipeline shall be designed to scale from the v1 event group toward ~70k athletes / 72 events without architectural change.
- NFR-004 (Reliability): Long scrapes shall be resumable and fault-tolerant; a failure mid-run shall not corrupt stored data.
- NFR-005 (Reproducibility): Model artifacts shall be versioned and traceable to the data snapshot and code that produced them; results shall be reproducible.
- NFR-006 (Transparency): The methodology (peak definition, normalization, validation) shall be documented and surfaced to the analyst user.
- NFR-007 (Portability): The hybrid deployment shall allow the pipeline to move from a local machine to an always-on host, and the dashboard to move hosts, without changing the artifact interface.
- NFR-008 (Cost): v1 shall use free or low-cost tooling/hosting.

## 6. Constraints

- C-1 (Deployment): **Hybrid.** A (scraping) and B (analysis) run **locally** (residential IP — required because the source blocks/distrusts datacenter IPs and to avoid account-flagging). C (dashboard) is hosted on **free/low-cost Streamlit-class hosting**, credential-gated. Components communicate only via their data contracts (raw store; artifact interface).
- C-2 (Legal/usage): Operator **has permission** to scrape and use/display the data. Data is **public athletics results** freely available elsewhere — not sensitive/personal-private. Governing constraint is **access control on the dashboard + scraping etiquette**, not data privacy.
- C-3 (Data dependency): The per-athlete career time-series required for modeling does not yet exist and must be scraped; the current source may also be **incomplete** for some required features. v1 cannot proceed past feasibility until the needed data is demonstrated obtainable and clean (Phase 0 spike gate), which may require extending A's scope or sources.
- C-4 (Budget): Minimal; free-tier / low-cost services preferred.
- C-5 (Timeline): None fixed / open-ended (to be set during implementation planning).
- C-6 (Existing assets): Working scraper *notebooks* (proven against the real site) are the source of truth for site structure; the `sports_data_platform/` OOP rewrite is mock-based and must be reconciled against the notebooks, not trusted as-is.

## 7. Assumptions

- A-1: The source site exposes per-athlete career history (year-by-year results per event) reachable from the athlete page (`db/at.php?...`), as suggested by the unfinished notebook 2.
- A-2: A recognized cross-event scoring method (e.g., World Athletics scoring tables, or %-of-personal-best / within-event z-score) is adequate to make performances comparable for the v1 group.
- A-3: For the v1 event group, enough athletes have enough career data points to fit individual trajectories and train a population model.
- A-4: "Peak performance" can be operationalized via the established literature method (trajectory fit; age-of-peak at the curve maximum; peak window = contiguous ages within X% of peak), refined by our own data analysis.
- A-5: The operator can run A (scraping) locally on a residential connection with authorized credentials.
- A-6: If the primary source lacks features the analysis needs, equivalent public data is obtainable from additional sources the operator may scrape (to be confirmed in the feasibility spike).
- A-7: **Intended subject population = semi-pro / professional athletes** (the population in the database). The tool projects an up-and-coming pro athlete's peak from historical pro data; it is **not** intended for the general public. Survivorship bias is therefore **accepted by design** as an applicability boundary, not a defect — predictions are valid for athletes already at semi-pro/pro level. (Distinct from the dashboard *users* in §2, who are analysts/coaches.)

## 8. Success Criteria

- SC-1: End-to-end demonstrability — pipeline produces a versioned model+data artifact, and the dashboard consumes it to deliver DC-1 and DC-2 for the v1 event group.
- SC-2: The model is benchmarked against a transparent **baseline** (population/event mean peak age and a simple per-athlete quadratic fit); the chosen model's error and uncertainty are reported on a held-out, **temporally valid** split.
- SC-3: Identified indicators of peak performance are reported with their statistical support and are consistent-with or reconciled-against the literature.
- SC-4: Methodology (peak definition, normalization, validation strategy, limitations) is documented and visible to the analyst user.
- SC-5: Honest reporting: known limitations and failure modes (e.g., sparse-input athletes) are explicitly surfaced, not hidden.

## 9. Risks

| ID | Risk | Mitigation |
|----|------|-----------|
| R-1 | **Data feasibility / completeness** — per-athlete career time-series not obtainable/clean enough, or primary source missing required features. | Phase 0 hands-on feasibility spike **gates** the rest of the plan; spike explicitly tests completeness and identifies any supplementary sources A must add (IS-7, A-6). |
| R-2 | **Etiquette / access** — over-aggressive scraping harms the source relationship; unauthorized dashboard access. | Operator is permitted to scrape (C-2); throttle/backoff and respect the site (NFR-001); credential-gate the dashboard (NFR-002, FR-015). |
| R-3 | **Model feasibility** — peak timing may be weakly predictable from available features. | Define success around soundness + honest reporting, not a fixed accuracy bar (§8); baseline-first modeling; report uncertainty. |
| R-4 | **Cross-event normalization wrong** — bad scoring distorts trajectories. | Adopt a recognized scoring method; validate normalized peaks against literature benchmarks (e.g., sprints ~25–26y). |
| R-5 | **Cold-start / sparse input** — a young uploaded athlete has few data points. | Define minimum-data thresholds; communicate prediction uncertainty; degrade gracefully (FR-012, FR-010). |
| R-6 | **Scraper fragility** — site structure/auth changes break the pipeline. | Port proven notebook logic; centralize selectors/URLs; resumable, monitored runs (NFR-004). |
| R-7 | **Deployment IP blocking** — running the scrape off-residential IP gets blocked/banned. | Pipeline stays local on residential IP (C-1); decoupled dashboard hosting avoids the issue. |
| R-8 | **Scope creep** across 72 events. | v1 is one event group; expansion is deliberate, config-driven (IS-1, IS-6). |
```
