# Peak Performance Predictor — Manual Test Plan (v1)

End-user walkthrough to validate the system by hand. Each test maps to a user
journey and its acceptance criteria. Run from the repo root with the venv active.

## Prerequisites
- `python3 -m venv .venv && source .venv/bin/activate`
- `pip install -e ".[scraper,pipeline,dashboard,dev]"`
- `.secrets` contains a valid `TILASTOPAJA_USER` / `TILASTOPAJA_PASS` (rotated).

---

## T1 — Operator pipeline run (Journey 4)
**Goal:** produce a fresh artifact bundle from scraped data.
1. Scrape a slice: `peakpredict-scrape --events 70 --sexes 2 --limit 50 --throttle 1 --db data/raw/peakpredict.duckdb`
   - Expect: log shows `authenticated`, `roster ... -> N athletes`, then per-athlete `-> K performances`.
2. Re-run the same command.
   - Expect (AC-4.1): already-done athletes are skipped (far fewer career fetches).
3. Build processed data: `peakpredict-build --db data/raw/peakpredict.duckdb`
   - Expect: `build complete: {season_bests, labels, features}` with non-zero counts.
4. Publish: `peakpredict-publish`
   - Expect: `published v… (primary=…)`; a new dir under `data/artifacts/<version>/` with
     `manifest.json, predictor.pkl, normalization.json, feature_schema.json, aggregates.parquet,
     similar_index.parquet, indicators.json, validation.json, season_bests.parquet,
     labels.parquet, athletes.parquet` (AC-4.2).

## T2 — Launch the dashboard
1. `streamlit run src/peakpredict/dashboard/app.py`
2. Expect: sidebar with **Data version** = the published version, and pages Explore / Upload & Predict / Indicators.
3. (If `dashboard_password` is set in Streamlit secrets) Expect a sign-in prompt first (FR-015).

## T3 — Explore an athlete (Journey 1)
1. On **Explore**, pick Event = 400m, Sex = Women.
2. Type part of a name in **Search**.
   - Expect: the athlete dropdown filters; empty search shows all for that event/sex.
   - Empty/no-match → an informational empty state (AC-1: empty state).
3. Select an athlete.
   - Expect: performance-over-time chart with the athlete's points, fitted curve, and a marked
     peak; the **population average + 10–90% band** overlaid (AC-1.1, AC-1.2).
   - Summary metrics (seasons, age range, best score, est. peak age).
   - A **Similar athletes** list (AC-1.3).
4. Select an athlete with very few seasons.
   - Expect: "Too few seasons to fit a reliable peak" and no peak line (AC-1.4).

## T4 — Upload & Predict (Journey 2)
1. On **Upload & Predict**, pick Event + Sex.
2. Enter ≥3 result rows (age, mark); click **Predict peak**.
   - Expect: predicted-peak chart (same hero chart as Explore) + predicted age, 80% interval,
     and a **confidence** value (AC-2.4).
3. Enter only 1–2 rows; predict.
   - Expect: blocked with "Need at least 3 valid results" (AC-2.2).
4. Enter an implausible mark or leave marks at 0.
   - Expect: rows with mark ≤ 0 are ignored / a validation error is shown (AC-2.3).
5. Confirm only supported events (100m/200m/400m) are selectable (AC-2.6: no silent wrong predictions).

## T5 — Indicators (Journey 3)
1. Open **Indicators**.
   - Expect: model MAE vs baseline MAE and 80% coverage; a table of feature↔peak-age correlations
     with r and p; the literature note (AC-3.1–3.3).

## T6 — Incompatible-artifact refusal (AC-4.4)
1. Edit a bundle's `feature_schema.json` `schema_version` to a different value; point the app at it
   (`PEAKPREDICT_BUNDLE=<that dir> streamlit run …`).
2. Expect: the app shows "Incompatible artifact bundle — refusing to serve predictions" and does
   not render predictions (never mispredicts silently).

---

### Pass criteria
All expected results observed; no Python tracebacks in the Streamlit console; predictions always
shown with uncertainty and a confidence flag; unsupported/insufficient inputs are blocked, not
silently predicted.
