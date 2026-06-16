# Peak Performance Predictor — ML Modeling Specification

> Status: v1, grounded in the Phase 0 feasibility spike (see feasibility-spike.md). Governs Component B's modeling. Defines the prediction target, peak definition, normalization, dataset construction, model ladder, and validation. Numbers herein are design targets; final hyper-parameters/thresholds are set during modeling experiments in execution. Scope: v1 = sprints (100m=event40, 200m=event50, 400m=event70), both sexes, events modeled independently.

## 1. The prediction task (precise framing)

There are TWO distinct quantities; conflating them is the classic error:

- **Descriptive peak (label generation):** given an athlete's (near-)complete career, the age at which their performance trajectory was maximal.
- **Predictive peak (the product task):** given an athlete's **partial, still-developing** career (the coach's "up-and-coming athlete", entered by hand), predict the age at which their peak **will** occur.

The dashboard's Upload & Predict flow is the **predictive** task. Therefore:
- **Labels** come from athletes with sufficiently complete careers (descriptive peak).
- **Features** are computed only from the **early/partial** portion of a career that would be available at prediction time.
- The model maps *early-career features → future peak*. **No post-peak information may enter the features** (leakage guard).

Primary output: **predicted age of peak performance** (continuous, years).
Secondary outputs: **peak window** (age interval of near-peak), **predicted peak level** (normalized score at peak), and a calibrated **uncertainty/prediction interval** on the peak age.

## 2. Definition of "peak performance" (label)

Adopted from the sports-science standard, validated on real data in the spike (Bolt: vertex 25.3y vs single-best 23.0y — the gap is why we do NOT use single-best):

- Per athlete, per (event, sex): take **season-best, wind-legal** performances → series of (age, normalized score).
- Fit performance-vs-age (see §6 for model of the trajectory). **Age of peak = the trajectory maximum** (vertex), not the single best mark.
- **Peak window** = contiguous ages whose fitted score is within **τ** of the peak (τ default = a small % of peak, e.g. within noise/measurement band; exact τ set in experiments). Reported as [age_lo, age_hi].
- **Label validity filter** — an athlete contributes a training label only if: (a) ≥ `MIN_LABEL_POINTS` season-bests, (b) spanning ≥ `MIN_LABEL_SPAN` years, (c) the fit has an interior maximum (a genuine ∩ in score space), and (d) the career is plausibly complete (observed decline past the peak, or age sufficiently beyond the population peak). Athletes still ascending are excluded from **labels** (but are exactly the **inference** population).

## 3. Performance normalization

Requirement: a monotone, higher-is-better, athlete-comparable score so trajectories are on a common scale (times are lower-is-better; field events higher-is-better; v1 is all track but this generalizes for expansion).
- **Primary:** **World Athletics scoring tables** (mark → points; higher = better; comparable across events) — future-proofs cross-event expansion and is interpretable.
- **Acceptable v1 simplification:** within-(event, sex) standardized score (z-score or percentile vs the event/sex distribution), since v1 models events independently.
- The **same normalization function must be reused verbatim by the dashboard's Upload flow** (no divergence — see user-journeys AC-2.5). It is part of B's published artifact.

## 4. Dataset construction

Unit of analysis = one athlete × (event, sex) career.
1. Source: per-athlete career scrape (Component A) → season-best, wind-legal mark per year per event.
2. Join DOB (roster by PID; century-correct 2-digit years; drop/flag missing DOB).
3. Compute age at each season-best; normalize marks (§3).
4. Produce per-athlete sequence: [(age₁, score₁), …]. Attach static attributes.
5. **Label set:** athletes passing §2 validity filter, with descriptive peak age + window + peak level.
6. **Feature set (per athlete, from partial career only):** see §5.

## 5. Features (from partial/early career only)

Static: sex, event, country/region (optional), age at first recorded competition, debut performance level.
Trajectory (computed over the observed-so-far window): number of seasons observed, current age, current best score, recent **progression rate** (slope of recent season-bests), curvature/decel signal, consistency (variance), score relative to the population reference at the same age (percentile-vs-population by age).
Sequence form (for sequence models): the ordered (age, score) pairs themselves.

Leakage guard: every feature must be computable at the simulated "prediction time" (i.e., using only data up to a cutoff age), never using the peak or post-peak observations.

## 6. Model ladder (baseline-first; advanced models must beat simpler ones)

Two modeling layers:
**(i) Trajectory model** (how the curve is fit, for label generation and per-athlete description):
- Quadratic in age (validated in spike). Robust/weighted variants to reduce outlier and wind influence. Higher-order/spline only if quadratic underfits diagnostics.

**(ii) Predictor** (early-career features → future peak age) — evaluated as a ladder:
- **B0 — Population baseline:** predict the (event, sex) mean peak age (the literature constant). Floor to beat.
- **B1 — Naive individual extrapolation:** fit a quadratic to the athlete's observed-so-far points, take its vertex. Uses the individual but unstable when sparse.
- **B2 — Hierarchical / mixed-effects (PRIMARY CANDIDATE):** partial-pooling model that shrinks each athlete's trajectory toward the population (by event/sex), giving stable estimates for **sparse, short, manually-entered** sequences and **native uncertainty**. This is the field standard and the best fit for our manual-entry reality.
- **B3 — ML regressor:** gradient-boosted trees on the §5 engineered features. Flexible; good when interactions matter; uncertainty via quantile regression / conformal.
- **B4 — Sequence model (RNN/Transformer) — EXPLORATORY ONLY:** justified only if B2/B3 leave material unexplained signal AND data volume supports it. Expected to be weak here because dashboard inputs are short, sparse, hand-entered sequences (3–10 points). Must beat B2/B3 on the same temporal split to be adopted; otherwise documented as not-justified.

Selection rule: adopt the **simplest model within noise of the best** measured performance. The RNN named in the original brief is **not assumed**; it must earn its place.

## 7. Validation (temporal / career-aware — non-negotiable)

- **Split by athlete**, never by row (no athlete in both train and test).
- **Forward-prediction protocol:** for each held-out athlete, **truncate** their career at one or more cutoff ages/horizons (e.g., first 3, 5, 7 seasons), compute features from the truncated data, predict peak age, and compare to their **realized descriptive peak** (from their full career). This simulates the real product use (predicting from partial data).
- Evaluate across cutoff horizons (earlier cutoff = harder = more useful).
- **No leakage:** features at a cutoff use only data ≤ cutoff.
- Optional temporal holdout by era to check drift over decades.

### Metrics
- **Peak-age MAE / RMSE** (years) — primary.
- **Peak-age bias** (systematic early/late).
- **Peak-level error** (normalized score).
- **Prediction-interval coverage & sharpness** (e.g., does the 80% interval contain truth ~80%? — calibration matters for honest reporting).
- **Skill vs baselines:** % MAE reduction over B0 (population mean) and B1 (naive extrapolation). A model that can't beat B0 by a clear margin is reported as such.

### Sanity checks
- Population peak-age estimates must land near literature (sprints ~25–26; spike season-bests consistent). Large divergence ⇒ normalization/label bug.

## 8. Uncertainty & honesty requirements

- Every prediction ships with a **calibrated interval** (B2 native posterior; B3 conformal/quantile).
- **Out-of-distribution** inputs (features far from training support) and **insufficient data** (< `MIN_POINTS`) are flagged, never silently predicted (user-journeys AC-2.6).
- **Survivorship bias is accepted by design** — the tool's intended subject population is **semi-pro / professional athletes**, which matches the database population. Predictions are therefore valid *within that population* (the intended use: projecting an up-and-coming pro's peak from historical pro data). This is stated as an **applicability boundary** ("valid for athletes already at semi-pro/pro level"), not a defect, and is surfaced to the user as scope rather than as a warning.

## 9. Thresholds & cold-start

- `MIN_POINTS` (dashboard inference minimum): provisional **≥ 3** season-bests spanning **≥ 2** years; finalized from the validation curve of error-vs-#points. Below it, no prediction (show points + population context only).
- `MIN_LABEL_POINTS` / `MIN_LABEL_SPAN` (training-label validity): provisional ≥ 5 season-bests / ≥ 4 years with an interior maximum; finalized in experiments.

## 10. Outputs (B → C artifact contract; the modeling half)

B publishes, versioned, with feature-schema compatibility (user-journeys AC-4.4):
1. Trained **predictor** + the **trajectory fitter**.
2. The **normalization function/tables** (must match dashboard Upload).
3. **Feature schema** (exact fields/units a user upload must provide).
4. **Reference-population aggregates** per (event, sex): average trajectory + percentile bands (for the dashboard overlays).
5. **Similarity basis** for "similar athletes".
6. **Indicator/correlation report** (the multivariate analysis: which features correlate with peak age/level + effect sizes + statistical support) — feeds the Indicators view (FR-006).
7. **Validation report** (metrics, calibration, baseline comparison) — surfaced for the analyst (honest reporting).

## 11. Risks / limitations (modeling)
- Sparse/young athletes → unstable individual fits; mitigated by hierarchical pooling (B2) and `MIN_POINTS`.
- Wind-aided and altitude marks distort trajectories → filtered/flagged (§3, spike cleaning reqs).
- Multi-event athletes (e.g., 100/200) modeled independently in v1; joint modeling deferred.
- Era/track-technology drift over decades → optional era control.
- "Peak window" τ and label-completeness criteria are judgment calls → fixed in experiments and documented.
```
