# Data pipeline

> [← back to README](../README.md) · Data-engineering & data-science deep dive

This is the path from a credentialed website to a clean, labelled, leakage-safe training
table. Every step is a small, tested module in `src/peakpredict/pipeline/` (plus the
scraper in `src/peakpredict/scraper/`).

```
raw results → season-bests → normalized scores → trajectory fit → peak labels → features
  1.9 M           162 k            (z-scores)        (per career)     10 k         27 k rows
```

## 1. Acquisition (Component A)

The source is a credentialed athletics database whose career pages are rendered behind a
JavaScript login, so acquisition uses Selenium. Three engineering concerns dominate:

- **Be a good citizen.** Each run logs in **once** and reuses the authenticated session's
  cookies; the browser is recreated periodically (reusing cookies) to bound memory without
  re-authenticating. Requests are throttled. Logging in repeatedly trips the source's
  rate-limiter — so the design minimizes logins to exactly one per run.
- **Be resumable.** A `scrape_state` table tracks each athlete as `pending`/`done`/`failed`.
  A full sprint scrape is ~12 hours and ~14 k athletes; it must survive interruption and
  resume from where it stopped. (The real v1 scrape ran 11 h 47 m, one login, zero errors.)
- **Be resilient.** Browser and connection-level death are detected and the session is
  transparently re-established, distinguishing a genuinely dead session from a transient
  hiccup.

The output is the DuckDB **raw store**: `athlete` (with static physical attributes),
`event`, `performance` (one row per result — the career time-series the model needs), and
`scrape_state`.

**Result:** 15,981 athletes and **1,896,089 individual performances**.

## 2. Season-bests — reducing noise to signal

`season_best.py` reduces the 1.9 M raw results to one best mark per **(athlete, event, sex,
season)**. Along the way it:

- filters to the supported v1 events (100/200/400 m);
- drops wind-aided marks (> +2.0 m/s) — illegal for record purposes;
- joins date-of-birth to compute **age** at each performance;
- keeps the *direction-aware* best per group (fastest time, or longest distance).

It also enforces a **plausibility guard** on age (`AGE_BOUNDS = 8–55`). This guard is a
data-science story worth telling: it was added *because of what the predictions revealed*
(see [Data quality](#5-data-quality--a-bug-found-by-looking-at-the-output)).

**Result:** 162,469 season-bests.

## 3. Normalization — one comparable scale

Raw marks are not comparable: a 100 m time of 10.1 s and a 400 m time of 45 s live in
different units and different directions (lower is better for time). The model needs a
single scale where **higher is always better** and an athlete's 100 m and 400 m
trajectories can be reasoned about together.

`normalization.py` solves this with a within-(event, sex), direction-aware **z-score**:

```
score = sign · (mark − mean) / std        sign = −1 for lower-is-better events
```

So a score is the number of standard deviations an athlete is above (or below) the average
mark for *their* event and sex — `0` is average, `+2` is elite, negative is below-average.
This is why the dashboard's y-axis can dip negative: it is a population-relative score, not
a time.

Two properties make this the right abstraction:

1. **Comparability** — every event and sex is mapped onto one scale, so cross-event
   modeling and cross-event peer-finding work.
2. **A single implementation** — the fitted normalizer is published in the artifact bundle
   and re-loaded by the dashboard, so a manually-entered mark is scored *identically* to
   how the model was trained. There is no second copy to drift.

`WorldAthleticsNormalizer` is an interface-compatible slot for the official World Athletics
scoring tables; swapping it in changes no caller.

## 4. From trajectory to a peak label

"Peak performance age" has to become a number the model can be trained against.
`trajectory.py` fits each athlete's career as a quadratic, `score = a·age² + b·age + c`.
Because score is higher-is-better, a career is an inverted-U and the peak is the parabola's
vertex (requires `a < 0`):

```
peak age = −b / 2a
```

`labels.py` extracts that vertex as the **peak-age label** for each (athlete, event, sex)
that has a clear interior maximum — i.e. an athlete we have actually observed rise *and*
decline. The label is **per event**: a 200 m specialist who also runs the 400 m gets a
separate peak for each, because the events peak at different ages.

A "near-peak window" — the age band whose fitted score is within a tolerance of the peak —
is also derived, and clamped to the observed age range (a flat parabola would otherwise
imply an implausibly wide window).

**Result:** 10,191 per-(athlete, event) peak labels.

## 5. Features — leakage-safe by construction

This is the subtlest and most important modeling decision. The system predicts *when a
developing athlete will peak* — so at training time it must only ever see what would be
known **before** the peak. `features.py` computes features from an athlete's **first *k*
observed seasons only** (k ∈ {3, 5, 7}), and never anything at or after the peak:

- trajectory shape so far: debut age & score, current age & best score, observed span;
- **progression rate** (slope of score vs age) and **recent slope** (last ≤3 seasons);
- consistency (mean and std of score);
- static physical attributes (height, weight) where known.

Training rows are `(features at cutoff k) → full-career peak age`. By construction the
features cannot leak the answer, and the model is trained exactly the way it is used in the
dashboard: from a partial, early career.

The feature set is a **versioned schema** (`FEATURE_SCHEMA_VERSION`) published in the
bundle; the dashboard refuses any bundle whose schema it cannot consume.

**Result:** 27,563 leakage-safe feature rows across ~14 k athletes.

## 6. Enrichment — widening the data

Static physical attributes (height, weight) are signal the model can use, but the source
populates them inconsistently. `enrich/wikidata.py` fills gaps from **Wikidata** (CC0, no
credentials) via SPARQL: it pulls height/mass/date-of-birth for athletes, normalizes names
(Unicode-fold, strip accents), and joins on **name + date-of-birth** to avoid mismatches,
writing only where the source left a gap and tagging the provenance.

**Result:** physical attributes for ~32% of athletes — a real, modest signal wired through
the feature pipeline.

## 7. Data quality — a bug found by looking at the output

A data-science lesson worth highlighting. The dashboard chart for some athletes stretched
its age axis from below −50 to above +50. Two root causes surfaced from *inspecting the
predictions*, not the code:

1. **An unbounded near-peak window.** For a nearly-flat trajectory the window half-width
   `√(τ/−a)` explodes as `a → 0`; one athlete had a 114-year window. Fixed by clamping the
   window to the observed age range — we don't claim "near peak" at ages never observed.
2. **Implausible ages from bad dates.** `age = (perf_date − dob)/365.25` had no bound, and
   the source contained stray dates — e.g. a 1956 Olympic champion (born 1935) carried a
   spurious **2022** result, computing to age 86.5. These garbage points break the chart
   *and* flatten the trajectory fits that produce labels. Fixed with the `AGE_BOUNDS`
   plausibility guard at the one place age is computed.

The guard removed ~100 corrupted rows (0.1% of the data) — and **improved every model's
accuracy**, because a single age-86 point with a normal score distorts the quadratic fit it
sits in. Cleaning the data moved the RNN from 1.41 → 1.38 y MAE. The fix shipped with a
regression test (the actual 1935-born athlete case).

## Population aggregates & indicators

Two more pipeline outputs feed the dashboard:

- `aggregates.py` builds **population percentile bands** (10/25/50/75/90th) of score by
  age-bin per (event, sex) — the grey band an athlete is plotted against.
- `indicators.py` reports which early-career features **correlate with peak age** (the
  "what predicts a late peak?" question), shipped in the bundle for the Indicators page.

---

Next: **[ML modeling →](modeling.md)** · **[Results & limitations →](results.md)**
