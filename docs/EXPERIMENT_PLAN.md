# EXPERIMENT PLAN

This is the experimental protocol. **It contains no results.** Every table below that will hold results shows
placeholders (`—`); those values are filled only from saved result files (ER-007).

---

## 1. Research questions

| ID | Question | Evidence |
|---|---|---|
| RQ1 | Does PSO-tuned Random Forest **improve or maintain** held-out test performance relative to the default RF? | paired per-fold test metrics, PSO vs. baseline |
| RQ2 | Does PSO's **feedback** help, compared with the same budget spent without feedback? | anytime curves and test metrics, PSO vs. random search |
| RQ3 | How does the PSO search **converge**? | gbest curves, mean swarm fitness, diversity, convergence iteration |
| RQ4 | How **stable** are the results across seeds and data folds? | spread of test metrics and chosen configurations across the 5 outer folds |
| RQ5 | How do the answers differ **across datasets** (easy/small, larger multi-class, real-world clinical)? | dataset-wise comparison tables and plots |

## 2. Datasets

| Property | Iris | Digits | Heart Disease (Cleveland) |
|---|---|---|---|
| Source | `sklearn.datasets.load_iris` (UCI Iris) | `sklearn.datasets.load_digits` (UCI Optical Recognition of Handwritten Digits, test set) | UCI Heart Disease id 45, file `processed.cleveland.data`, DOI 10.24432/C52P4X, CC BY 4.0 |
| Target | species (setosa, versicolor, virginica) | digit 0–9 | `num` (0–4), binarized: `num > 0` → 1 (disease) |
| Samples | 150 (measured) | 1,797 (measured) | 303 (measured) |
| Features | 4 continuous (cm) | 64 pixel intensities, 0–16 | 13: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal |
| Classes | 3 | 10 | 2 |
| Class counts | 50 / 50 / 50 (measured) | 174–183 per class (measured) | 164 / 139 (measured) |
| Max/min class ratio | 1.00 | 1.05 | 1.18 (measured) |
| Missing values | none | none | `?` in `ca` (4) and `thal` (2) (measured) |
| Exact duplicate (X, y) rows | 1 (measured; kept) | 0 (measured) | 0 (measured) |
| Categorical features | — | — | cp, restecg, slope, thal (integer-coded); sex, fbs, exang (binary) |
| Preprocessing | none | none | NaN → `SimpleImputer(most_frequent)` inside the pipeline; integer codes kept |
| Fitness metric | accuracy | accuracy | accuracy (the gate switches to balanced accuracy only if the ratio exceeds 1.5) |
| Outer test fold size | 30 | ≈359–360 | ≈60–61 |
| Inner validation fold size | 24 | ≈287–288 | ≈48–49 |

The Heart values were measured in Phase 2 by the audit (`data/DATASET_AUDIT.md`), before any experiment runs;
all of them match the values expected in Phase 0.

## 3. Preprocessing

- No scaling or encoding. Random Forest is invariant to monotone feature scaling and splits integer codes directly.
  One-hot encoding for Heart is an optional ablation (§12).
- **Heart only:** `?` → NaN at load, then `SimpleImputer(strategy="most_frequent")` as the first pipeline step. It is
  refit inside every inner-CV training part, in the final refit, and in the baseline (MLR-004).
- Nothing is fitted on the full dataset. The audit computes descriptive counts only and fits no model parameters
  (DR-010).

## 4. Data splitting

| Level | Procedure | Purpose |
|---|---|---|
| Outer | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` | fold *k* = held-out test set of run *k*; every sample tested exactly once |
| Inner | `StratifiedKFold(n_splits=5, shuffle=True, random_state=k)` on the optimization portion of run *k*, **fixed for the whole run** | validation for fitness (the PSO feedback) |
| Deployment | inner procedure on the full dataset, seed 5 | produces the recommended configuration; no test |

## 5. Methods compared

| Method | Configuration choice | Uses feedback? | Budget |
|---|---|---|---|
| `baseline` | `RandomForestClassifier()` defaults: `n_estimators=100`, `max_depth=None`, `min_samples_split=2` | — | 1 CV score (reference) |
| `random_search` | best of 210 i.i.d. uniform configurations from Ω | **no** (open loop) | 210 evaluations |
| `pso` | PSO gbest after 20 iterations | **yes** (closed loop) | 210 evaluations (10 × 21) |

All three use the same pipeline, RF seed, inner folds (for CV scores), refit procedure and test fold.

## 6. PSO configuration (default experiment)

| Parameter | Value | Source |
|---|---|---|
| Particles N | 10 | PDF slide 10 |
| Iterations T | 20 (plus the initial evaluation) | ADR-014 |
| Evaluations per run | 210 | derived |
| Inertia w | 0.7298 | ADR-013 |
| c1, c2 | 1.49618, 1.49618 | ADR-013 |
| r1, r2 | U[0,1), per particle, per dimension, per iteration | PDF slide 9, ADR-013 |
| Initial positions | U[l, u] | ADR-012 |
| Initial velocities | U(±0.1 × range) = ±(15, 1.8, 0.8) | ADR-012 |
| Velocity clamp | ±0.2 × range = ±(30, 3.6, 1.6) | ADR-012 |
| Boundary | absorbing wall | ADR-012 |
| Integer handling | continuous state; decode = clip(rint) | ADR-011 |
| Topology / update | gbest / synchronous | ADR-014 |
| Stopping | max_iter; patience disabled | ADR-015 |
| Cache | per run and method | ADR-016 |

**Search space (hyperparameter ranges):** `n_estimators` ∈ [50, 200], `max_depth` ∈ [2, 20],
`min_samples_split` ∈ [2, 10], all integers. That gives |Ω| = 25,821, and PSO evaluates at most 210 of them (< 1%).

PSO parameters are **shared across datasets** in the default experiment. Per-dataset overrides are supported by the
configuration but not used (ARCHITECTURE §H).

## 7. Protocol (per dataset)

1. Load the dataset and verify its checksum (Heart).
2. Audit it (counts, missing, duplicates, class ratio) and select the fitness metric by the gate.
3. Build the 5 stratified outer folds (seed 42).
4. For each outer fold *k* = 0…4, with seed *s_k* = *k*:
   1. Create the `FitnessEvaluator` on the optimization portion, with inner 5-fold CV (seed *k*, fixed).
   2. **Baseline:** compute its CV score (reference only), refit on the optimization portion, test once, save.
   3. **Random search:** inside `OptimizationPhase`, evaluate 210 uniform configurations and select the best (earliest
      on ties); then refit, test once, save.
   4. **PSO:** inside `OptimizationPhase`:
      1. initialize the swarm (10 particles) and evaluate every particle, recording each evaluation;
      2. set pbest and gbest;
      3. update velocities (with the clamp);
      4. update positions, apply the absorbing bounds, and decode to integers;
      5. re-evaluate the candidate configurations (the closed-loop feedback);
      6. update pbest and gbest, and record the iteration summary;
      7. repeat steps 3–6 until iteration 20;
      8. extract the gbest configuration.

      After the loop: train the final RF on the whole optimization portion with the gbest configuration, evaluate it
      on the held-out test fold **once**, and save the results and convergence history.
5. **Deployment run:** PSO on the full dataset (seed 5) → `deployment.json`.
6. After all datasets: aggregate the summaries and compare the methods (§8).

This covers all 18 steps required by the brief. Steps 1–3 correspond to the brief's steps 1–3, step 4.2 to step 4,
steps 4.4.1–4.4.8 to steps 5–13, the lines after the loop to steps 14–15, and steps 5–6 plus the saved files to
steps 16–18.

## 8. Evaluation and comparison

### 8.1 Metrics

| Stage | Metric | Role |
|---|---|---|
| Optimization (inner CV) | accuracy (mean of 5 folds) | **fitness**: fed back to PSO |
| Optimization (inner CV) | balanced accuracy, macro F1 | diagnostics, logged only; never fed back |
| Test (outer fold) | accuracy | primary performance measure |
| Test | balanced accuracy, macro precision, macro recall, macro F1 | secondary |
| Test (Heart only) | positive-class (disease = 1) precision, recall, F1 | clinical relevance |
| Test | confusion matrix (per fold and pooled over folds) | error structure |
| Cost | evaluations, unique RF fits, wall time | efficiency |

Precision, recall and F1 use `zero_division=0`, stated explicitly.

### 8.2 Comparison procedure

- **Paired by fold.** For each dataset, method and fold, compute $\Delta_k = M_k^{method} - M_k^{baseline}$ for test
  accuracy (and macro F1). Also compute PSO minus random search.
- **Report:** mean ± std of each metric over the 5 folds; mean ± std of $\Delta_k$; win/tie/loss counts (a tie means
  identical test accuracy on that fold); and pooled accuracy from all out-of-fold predictions.
- **No significance tests.** With 5 pairs, the smallest two-sided Wilcoxon p-value is 0.0625, so significance at 5%
  is impossible by construction. CV folds are also not independent. Results are presented descriptively, and
  differences smaller than one or two test samples (Iris: 1 sample = 3.3 points; Heart: 1 sample ≈ 1.6 points) are
  described as ties or noise (ADR-020).
- **Validation vs. test.** Best validation fitness is reported beside the test metrics, labelled "optimistically
  biased: selection over 210 evaluations". It is never presented as expected performance.

### 8.3 Convergence analysis (RQ3)

Per PSO run: the gbest fitness per iteration; mean, std, min and max swarm fitness per iteration; swarm diversity
(mean distance of the range-normalized positions from the centroid); the convergence iteration (the last gbest
improvement); cache-hit rate per iteration (a proxy for the swarm collapsing onto the same configurations); and the
number of configurations tied with the final gbest. Aggregated over folds: the mean gbest curve with its min–max
band.

### 8.4 Repeated-run and stability analysis (RQ4)

- The 5 outer folds are 5 independent runs, with different PSO seeds, RF seeds, inner folds and test folds. This
  meets proposal Objective 08 (multiple random seeds).
- Report the spread of test metrics, the spread of chosen configurations (per hyperparameter: min, max, mode) and
  boundary hits (a chosen value equal to a bound).
- **Optional** (only if time permits and recorded as a separate named experiment): repeat PSO on one fixed fold with
  seeds 10–14, to isolate optimizer-seed variance from data-split variance.

## 9. Plots (generated only from saved result files)

| # | Figure | Source file(s) | Answers |
|---|---|---|---|
| F1 | System architecture diagram | ARCHITECTURE.md §A (static) | — |
| F2 | Closed-loop feedback diagram | ARCHITECTURE.md §C (static) | — |
| F3 | PSO convergence: gbest fitness vs. iteration, one line per fold plus the mean, per dataset | `pso/iterations.csv` | RQ3 |
| F4 | Anytime curve: best-so-far validation fitness vs. evaluation count, PSO vs. random search (mean ± band over folds) | `*/evaluations.csv` | RQ2 |
| F5 | Test accuracy: baseline vs. random search vs. PSO per dataset (bars = mean, dots = folds) | `summary_folds.csv` | RQ1, RQ2 |
| F6 | Hyperparameter search behaviour: each particle's decoded n_estimators, max_depth and min_samples_split vs. iteration (3 panels, one fold) | `pso/evaluations.csv` | RQ3 |
| F7 | Particle fitness progression: every particle's fitness per iteration (scatter), with the mean and gbest lines | `pso/evaluations.csv`, `pso/iterations.csv` | RQ3 |
| F8 | Best-hyperparameter evolution: the gbest configuration vs. iteration | `pso/iterations.csv` | RQ3 |
| F9 | Dataset-wise comparison: Δ test accuracy vs. baseline, per dataset and method (fold dots, mean marker) | `summary_folds.csv` | RQ1, RQ5 |
| F10 | Pooled confusion matrices (PSO and baseline) per dataset | `*/predictions.csv` | RQ1 |
| F11 | Swarm diversity vs. iteration | `pso/iterations.csv` | RQ3 |

## 10. Tables (filled only from result files)

**T1: Dataset summary.** Filled from `data/DATASET_AUDIT.md`.

**T2: Main results (per dataset).** Filled from `summary.csv`:

| Dataset | Method | Test accuracy (mean ± sd) | Test macro F1 (mean ± sd) | Δ acc vs. baseline (mean ± sd) | W/T/L vs. baseline | Validation fitness (biased) | Unique fits |
|---|---|---|---|---|---|---|---|
| Iris | baseline | — | — | 0 (by definition) | — | — | 1 (by definition) |
| Iris | random_search | — | — | — | — | — | — |
| Iris | pso | — | — | — | — | — | — |
| … | … | … | … | … | … | … | … |

**T3: Chosen configurations per fold.** Filled from `*/final.json`, with columns dataset, fold, method,
n_estimators, max_depth, min_samples_split and n_ties_with_best.

**T4: Convergence summary.** Filled from `pso/final.json` and `iterations.csv`, with columns convergence iteration,
final diversity and cache-hit rate.

**T5: Deployment configurations.** Filled from `deployment.json`, one recommended configuration per dataset, listed
next to the outer-CV performance estimate.

**T6: Runtime.** Filled from `final.json` timing fields and the manifest.

## 11. Result storage

All outputs go to `results/<exp_id>/` in the layout and schema of [RESULTS_SCHEMA.md](RESULTS_SCHEMA.md). The
experiment's resolved config and manifest are saved first, and every number in the report cites its file (DOC-005).

## 12. Compute budget

Evaluations per dataset: $5 \times (210 + 210) + 210 = 2{,}310$. Wall time $\lesssim 2{,}310 \times t_{eval}$.

| Dataset | Measured worst-case $t_{eval}$ (5-fold, parallel folds) | Upper bound on wall time |
|---|---|---|
| Iris | 0.24 s | ≈ 9 min |
| Digits | 0.52 s | ≈ 20 min |
| Heart | TO VERIFY (expected ≈ Iris) | ≈ 9–10 min |
| **Total** | | **≈ 40 min** (upper bound; cache hits and cheaper configurations reduce it) |

The **Phase 6 timing gate** re-measures $t_{eval}$ for all three datasets. If the projected total exceeds 2 hours,
the recorded fallback is inner 3-fold CV (the PDF's original choice), and the change is recorded as an ADR update.

## 13. Optional extensions (separate named experiments; not part of the core claim)

- One-hot encoding of Heart categoricals (ablation).
- Patience-based early stopping (effect on budget and outcome).
- `max_features` as a fourth decision variable.
- Linearly decreasing inertia or a ring topology.
- Fixed-fold, multi-seed PSO stability study (§8.4).

## 14. Threats to validity (stated in the report)

- Small test folds (Iris, Heart): differences of one or two samples are noise.
- Flat fitness landscapes: many configurations tie, so the "best" configuration is one of several.
- Selection bias of validation fitness: handled by reporting test metrics only as performance.
- Default RF is strong on these datasets: a null result (tie) for RQ1 is plausible and will be reported as such.
- A 3-variable search space may be easy enough for random search to match PSO: a null result for RQ2 is plausible
  and will be reported as such.
- Results depend on the scikit-learn version (pinned, recorded in the manifest).
