# Results & limitations

> [← back to README](../README.md) · Predictive-analysis deep dive

The point of this document is **calibrated honesty**: what the model gets right, what it
gets wrong, how that was measured, and what would move the needle. A peak-prediction system
that overstates its confidence is worse than useless to a coach.

## Headline numbers

Held-out, 5-fold, athlete-grouped cross-validation over 6,998 labelled athletes
(27,563 evaluation rows):

| Model | MAE | RMSE | Bias | Skill vs baseline | 80% interval coverage |
|---|---:|---:|---:|---:|---:|
| Group-mean baseline | 2.24 y | 2.84 y | 0.00 | — | 0.81 |
| Pooled Ridge | 1.60 y | 2.08 y | 0.00 | 0.28 | 0.83 |
| **Bidirectional RNN** | **1.38 y** | **1.84 y** | +0.02 | **0.38** | **0.84** |

Reading these:

- **MAE 1.38 years** — on average the predicted peak age is within ~1.4 years of the true
  peak. For context, an elite sprinter's competitive prime spans only a handful of years, so
  this is genuinely useful resolution.
- **Near-zero bias** — the model is not systematically early or late.
- **Coverage 0.84** vs a nominal 80% interval — the prediction intervals are honest, even
  slightly conservative. The number the dashboard shows next to a projection means what it
  says.
- **Skill 0.38** — the production model closes 38% of the distance from "guess the group
  mean" to "perfect".

## Methodology — why these numbers are trustworthy

- **No leakage, by construction.** Features come only from an athlete's first *k* seasons;
  the label is the full-career peak. An athlete-grouped split (with a runtime assertion that
  no athlete crosses the train/test boundary) means the model is always predicting a
  *stranger's* future from partial data — the real task.
- **Calibrated intervals.** Each model's interval is an **out-of-fold residual std**, so the
  coverage reported is the coverage shipped, not an in-sample illusion.
- **Multiple seeds.** Architecture comparisons were run over several seeds and reported with
  their spread, so a 0.003 y "win" is correctly called a tie rather than a result.

## Where the error comes from — synthetic decomposition

To understand *whether the remaining error is the model's fault or the problem's*, a
synthetic study (`analysis/synthetic_validation.py`) generated athletes with known peak
ages and calibrated label noise, then measured how much error is irreducible. The finding:
a large share of the ~1.4 y error is **inherent to the problem** — peak age is only
partially determined by early-career shape, and the trajectory-fit label itself carries
noise. The model is operating close to the floor the data allows, which is why architecture
search couldn't push past it (see [modeling](modeling.md#the-decisive-finding-data-was-the-lever-not-architecture)).

## A concrete forward-prediction demo

Take real athletes, **truncate them to their first five seasons**, and predict — then
compare to their known full-career peak:

| Athlete | Event | Predicted | True peak | Error |
|---|---|---:|---:|---:|
| Agnė Eggerth | 100 m | 23.0 | 22.8 | **0.2** |
| Adrián Fernández | 400 m | 22.7 | 23.0 | **0.3** |
| Agnė Eggerth | 200 m | 22.8 | 22.1 | **0.7** |
| Aaron Armstrong | 200 m | 24.2 | 26.9 | 2.7 |
| Aliann Pompey | 400 m | 25.0 | 29.1 | 4.1 |
| Aham Okeke | 100 m | 22.4 | 26.5 | 4.1 |

The pattern *is* the lesson: the model nails athletes who peak near the norm (~22–25) and
**under-predicts genuine late-bloomers**. Aliann Pompey's best 400 m came at 31; nothing in
her first five seasons signals that, so the model pulls her toward the population mean. This
is textbook regression-to-the-mean on a partially-predictable target, and the interval
correctly widens for these cases but cannot always cover a true outlier.

## Strengths

- **Typical trajectories** — predicted within ~1 year for the majority of athletes who peak
  near the population age.
- **Calibrated uncertainty** — the interval is trustworthy in aggregate (84% coverage).
- **Event-specific** — separate peaks per event (a 200 m and 400 m peak can differ by years),
  which the per-(athlete, event) labeling captures.
- **Honest about what it doesn't know** — the dashboard distinguishes an **observed** peak
  (history) from a **projected** one (a model guess), so a user is never shown a prediction
  dressed up as a fact.

## Limitations

- **Late-bloomers** are systematically under-predicted — the dominant error mode, and an
  inherent one: the signal isn't in the early data.
- **Sprints only (v1)** — 100/200/400 m. The design is event-group-agnostic, but other
  families (jumps, throws, distance) need their own scrape and normalization fitting.
- **Source completeness** — peaks depend on having enough career seasons; athletes with very
  few seasons get neither a label nor a projection (the system shows a blank rather than
  inventing one).
- **Physical attributes are sparse** — known for ~32% of athletes; a useful but partial
  signal.

## What would move the floor

Consistent with the project's central finding — **data, not model complexity** — the highest-
leverage next steps are about data, not architecture:

1. **More labelled careers** — the lever that has worked every time so far.
2. **Per-season context features** — wind, competition tier, finishing place, and percentile-
   vs-population-at-age were prototyped in the enriched RNN experiments; folding the best of
   them into production is the most promising modeling step.
3. **Wider physical coverage** — more height/weight via additional CC0 sources.
4. **A late-bloomer-aware target or loss** — explicitly modeling the heavy tail rather than
   regressing its mean.

---

Next: **[Dashboard →](dashboard.md)** · **[ML modeling →](modeling.md)**
