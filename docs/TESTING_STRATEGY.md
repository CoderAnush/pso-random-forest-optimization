# TESTING STRATEGY

The testing strategy covers unit, integration and experiment-level tests. Test IDs are referenced from
[REQUIREMENTS.md](REQUIREMENTS.md) and [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

**Tooling:** `pytest`; shared fixtures live in `tests/conftest.py`.

**Layout:** `tests/unit/`, `tests/integration/`, `tests/experiment/`.

**Fixtures:**
- `tiny_space`, a small search space;
- `stub_quadratic`, an objective with a known interior optimum and no scikit-learn;
- `iris_fold0`, the real Iris fold 0;
- `test_config`, a minimal budget (e.g. N = 4, T = 2, inner folds 3) used only by tests.

**Principles:**
1. The **optimizer is tested without ML** (stub objectives), and the **evaluator is tested without PSO**. This
   mirrors the architecture's separation.
2. The two headline properties each get **several independent proofs**: *test data never enters the PSO fitness
   loop*, and *the loop is genuinely closed*.
3. Tests assert on behaviour and saved artifacts, not on specific accuracy values. **No test hard-codes an expected
   accuracy from a real dataset.**
4. Every test is deterministic (seeded) and the full suite runs in under 5 minutes.

---

## 1. Unit tests (UT)

| ID | Target | What is asserted |
|---|---|---|
| UT-01 | `SearchSpace.decode`, integer conversion | `decode` returns Python `int`s within the bounds for random positions in and beyond the box. Rounding is half-to-even: `decode([124.5, 8.5, 2.5]) == (124, 8, 2)`. Upper-bound clipping: `[200.4, 20.49, 10.2] → (200, 20, 10)`. Lower-bound clipping: `[49.6, 1.2, 1.9] → (50, 2, 2)`. |
| UT-02 | particle initialization | All initial positions lie in [l, u]; all initial velocities lie within ±0.1·range. The same seed gives identical arrays; different seeds give different arrays. The shapes are (N, 3). |
| UT-03 | velocity update | The optimizer's single-step update, with injected r1 and r2, reproduces the MATHEMATICAL_FORMULATION §9 example exactly (raw velocity (37.5824, 6.9306, −2.0290) to 1e-4; clamped (30, 3.6, −1.6); new position (150.4, 11.8, 3.0); decoded (150, 12, 3)). |
| UT-04 | velocity clamp | After any update, \|v_d\| ≤ v_max,d = 0.2·(u_d − l_d); components below the clamp are unchanged. |
| UT-05 | position update and boundary handling | Crossing a bound gives position = bound and that velocity component = 0 (the §9 boundary example: x₃ = 2.5, v₃ = −1.6 → x₃ = 2, v₃ = 0); components that do not cross are untouched; positions always stay in B. |
| UT-06 | pbest update | pbest changes only on strict improvement (`>`); equal fitness keeps the old pbest position; pbest fitness never decreases. |
| UT-07 | gbest update | gbest fitness is non-decreasing over iterations; it equals the max of the pbest fitnesses; same-iteration ties go to the lowest particle index; equality with the old gbest keeps the old gbest. |
| UT-08 | stopping | With patience disabled, there are exactly T iterations and N·(T+1) objective calls (default: 210), and `stop_reason = "max_iter"`. With patience enabled on a constant objective, the run stops at t = P with `stop_reason = "patience"`. The convergence iteration equals the last iteration in which gbest improved. |
| UT-09 | fitness calculation | `FitnessEvaluator(config)` equals `mean(cross_val_score(same pipeline, X_opt, y_opt, cv=same folds))` exactly; the inner folds are identical across calls; switching the metric to `balanced_accuracy` changes the scorer used. |
| UT-10 | fitness cache | A second call with the same configuration performs no fit (a spy counts `fit` calls), returns an identical fitness, and sets `cache_hit = True` and `fit_time_s = 0`. `n_unique_fits` counts cache misses only. |
| UT-11 | RF builder and baseline | The pipeline's RF has exactly the given n_estimators, max_depth and min_samples_split, plus `random_state = seed` and `n_jobs = 1`; every other `get_params()` value equals `RandomForestClassifier()` defaults. The baseline configuration is (100, None, 2). |
| UT-12 | failure handling | An objective whose fit raises returns `fitness = -inf`, `status = "failed"` and the error text, and the optimizer continues; NaN scores are treated as failures; if **all** evaluations fail, the run raises. |
| UT-13 | dataset loaders | Shapes, label ranges and class counts match `data/DATASET_AUDIT.md`. Heart: `?` → NaN, target is binary and `num > 0 → 1`. A tampered copy of the raw file fails the SHA-256 check. |
| UT-14 | outer folds | The 5 test folds are pairwise disjoint and their union is all indices; each fold's class proportions are within one sample of the overall proportions; the same seed gives identical folds. |
| UT-15 | test-set guard | `HeldOutTestSet.reveal()` raises `TestSetAccessError` inside `OptimizationPhase` (including nested and exception-exit cases) and works after the context exits; test arrays have no public attribute other than via `reveal()`. |
| UT-16 | final metrics | On hand-made `y_true` / `y_pred` arrays: accuracy, balanced accuracy, macro P/R/F1 (`zero_division = 0`), positive-class metrics for the binary case, and the confusion matrix with explicit label order. |
| UT-17 | random search | Samples lie in Ω; exactly `budget` objective calls are made; the sample sequence is identical whether the objective returns random values or constants (**independent of feedback**); ties go to the earliest evaluation; the empirical distribution of each dimension is approximately uniform (chi-square with a generous tolerance, seeded). |
| UT-18 | configuration | Layered merge order (default ← file ← dataset ← CLI); validation rejects low ≥ high, non-integer bounds, `len(run_seeds) != outer_folds` and unknown metrics; the resolved config round-trips to JSON; the config hash is stable. |
| UT-19 | recorder | Written CSVs have exactly the RESULTS_SCHEMA columns in order, with the correct dtypes and nullability. |
| UT-20 | preprocessing leakage | The Heart pipeline's imputer, after `fit(X_train)`, has `statistics_` equal to the per-column mode of `X_train` only (checked against a mode computed on the training rows; injecting a distinctive value only in the test rows does not change the statistics). The pipeline is a no-op for datasets without missing values. |
| UT-21 | class-ratio gate | Ratio ≤ 1.5 keeps `accuracy`; ratio > 1.5 selects `balanced_accuracy`; an explicit metric in config overrides the gate only if the gate is disabled. |
| UT-22 | layering rule | Static AST scan: nothing in `pso_rf/optimization/` imports `sklearn`, `pso_rf.evaluation`, `pso_rf.models`, `pso_rf.datasets` or `pandas`; nothing in `pso_rf/evaluation/` imports `pso_rf.optimization`; `PSOOptimizer.__init__` and `RandomSearch.__init__` have no data parameters. |

## 2. Integration tests (IT)

### IT-01: Closed-loop execution (conceptual loop, no ML)
**PSO → hyperparameters → system → fitness → PSO update must run as a closed loop.**
- System: a stub objective $F(\theta) = -\|(\theta - \theta^\dagger)/(u - l)\|^2$ with $\theta^\dagger = (140, 9, 5)$,
  an interior optimum.
- Run PSO (N = 10, T = 20, seed 0) through the **real** `SearchSpace`, `decode` and callbacks.
- Assert:
  - every objective call received a decoded integer configuration produced by the optimizer;
  - gbest fitness is non-decreasing;
  - the final gbest is strictly better than the best initial particle;
  - the final gbest configuration is within a small distance of $\theta^\dagger$ (tolerance set generously; the
    assertion is on direction, not precision);
  - the configurations proposed at iteration t+1 differ from those at t for at least one particle (the system
    parameters actually change).

### IT-02: Feedback ablation (proves the optimizer depends on feedback)
- **Constant objective:** with $F \equiv 0$, pbest and gbest never change after initialization. Run the same seed on
  the quadratic from IT-01 and on the constant: the trajectories must **diverge** by iteration 2.
- **Shuffled feedback:** wrap the IT-01 objective so that it returns the fitness of a *different*, randomly permuted
  configuration. The trajectory differs from the true-feedback run, and the final gbest is (for this seed) worse.
  This shows the swarm is steered by the fitness values rather than moving independently of them.

### IT-03: Closed loop with the real Random Forest (end to end, small)
- Iris, fold 0, `test_config` budget, run through the **runner**.
- Assert:
  - `evaluations.csv` has N·(T+1) rows;
  - every row's hyperparameters are in bounds and are integers;
  - fitness values lie in [0, 1];
  - `iterations.csv` has T+1 rows;
  - `final.json` exists and contains test metrics;
  - the console trace printed one line per iteration.

### Test-data isolation proofs (IT-04 … IT-07): **test data never enters the PSO fitness loop**

| ID | Proof | Method | Assertion |
|---|---|---|---|
| IT-04 | Index disjointness | For every dataset and fold, compare `OptimizationData.indices` with `HeldOutTestSet.indices` | The intersection is empty; the union is all rows; the test indices of different folds are disjoint. |
| IT-05 | Observational spy | During one run, patch `Pipeline.fit`, `Pipeline.predict` and `RandomForestClassifier.fit`/`predict` to hash every row they receive (SHA-256 of the row bytes plus the label where given). Record the hashes seen **inside** `OptimizationPhase` separately from those seen after. | The set of hashes seen inside `OptimizationPhase` is disjoint from the set of test-row hashes (duplicate rows are handled by comparing against the multiset of *test-only* row hashes; the Iris duplicate is checked explicitly). After the phase, `predict` sees exactly the test rows. |
| IT-06 | **Test-label invariance** | Run fold 0 twice, identically except that the second run permutes `y_test` (or sets it to a constant). | `evaluations.csv`, `iterations.csv` and the chosen configuration are **bit-identical** (excluding timing columns); only the test metrics differ. Any leak of test labels into fitness would break this. |
| IT-07 | Test-feature invariance | As IT-06, but replace `X_test` with random noise of the same shape. | The same bit-identical trajectory. Any leak of test features into fitness or preprocessing would break this. |

A related structural check is UT-22: the optimizer cannot even receive data.

### Remaining integration tests

| ID | Target | What is asserted |
|---|---|---|
| IT-08 | Reproducibility | The same config run twice gives identical `evaluations.csv`, `iterations.csv`, `final.json`, `predictions.csv` and `summary*.csv`, after dropping `timestamp`, `fit_time_s`, `elapsed_s`, `total_time_s` and the `*_at` fields. Parallel folds (`n_jobs_folds = 5`) and serial (`= 1`) give identical scores. |
| IT-09 | Seed sensitivity (sanity) | Changing the run seed changes the PSO trajectory. This guards against a seed being silently ignored. |
| IT-10 | Baseline protocol | The baseline's CV score uses the same inner folds as the optimizers in that run; its final model is refit on the full optimization portion with `random_state = seed`; the test metrics are computed once. |
| IT-11 | Preprocessing leakage (Heart) | Overwrite every `ca`/`thal` value in the **test fold only** with a sentinel (e.g. 99) that never occurs in the optimization portion; capture every imputer fitted during optimization (spy on `SimpleImputer.fit`); assert that no captured `statistics_` contains the sentinel and that the fitness trajectory is unchanged. The Heart pipeline runs through CV with NaNs present. |
| IT-12 | Convergence tracking | `iterations.gbest_fitness[t]` equals the running maximum of `evaluations.fitness` up to the end of iteration t; `best_so_far_fitness` is the running max per row; `convergence_iteration` matches the last increase; the cumulative evaluation and unique-fit counts are consistent with the cache flags. |
| IT-13 | Random search pipeline | Random search through the runner produces `evaluations.csv` with 210 rows (default budget) and the same schema, but with empty PSO-only fields (iteration, particle, position, velocity, pbest); `final.json` is valid. |

## 3. Experiment-level tests (ET)

| ID | Target | What is asserted |
|---|---|---|
| ET-01 | Smoke experiment | `python -m pso_rf run` with `configs/demo.yaml` (Iris, fold 0: the live demo) and with `configs/test.yaml` (all 3 datasets, tiny budget) completes and exits 0; CLI subset overrides (`--datasets`, `--folds`, `--methods`) work, and `verify` passes on the output. |
| ET-02 | Completeness audit | For every dataset × fold × method, the expected files exist. PSO and random search have exactly `budget` evaluations each. The deployment files exist. The manifest's dataset checksums match the files on disk. `config.resolved.json` hash equals the manifest's config hash. |
| ET-03 | Summary consistency | `summary_folds.csv` and `summary.csv` are recomputed from the per-fold `final.json` and `predictions.csv` and match exactly; pooled accuracy equals the accuracy over concatenated predictions; win/tie/loss counts are consistent with the deltas. |
| ET-04 | Temporal isolation | For every run, the `test_evaluated_at` timestamp is later than `optimization_finished_at`, and `run.log` shows no test-set reveal while an `OptimizationPhase` is open. |
| ET-05 | Plots from data | Every figure in `plots/<exp_id>/` has a SOURCES.json entry whose files exist and whose config hash matches; the plotting code imports no model or optimization code (AST check). |
| ET-06 | Timing gate / runtime | Measure $t_{eval}$ for each dataset (worst-case and mid-range configurations) and compute the projected total runtime; record it. In Phase 14, record the actual runtime alongside the prediction. It is a measurement plus a recorded decision rather than a pass/fail on speed. |

## 4. Coverage of the brief's required test topics

| Required topic | Tests |
|---|---|
| Particle initialization | UT-02 |
| Boundary handling | UT-05, UT-01 |
| Integer conversion | UT-01, IT-03 |
| Velocity update | UT-03, UT-04 |
| Position update | UT-03, UT-05 |
| pbest update | UT-06 |
| gbest update | UT-07 |
| Fitness calculation | UT-09, UT-21 |
| Random Forest evaluation | UT-11, UT-09, UT-12, IT-03 |
| Data leakage | UT-20, IT-11, IT-04…IT-07 |
| Closed-loop execution | IT-01, IT-02, IT-03 |
| Convergence tracking | IT-12, UT-08 |
| Reproducibility | IT-08, IT-09, UT-02 |
| Final test isolation | UT-15, IT-04…IT-07, ET-04, UT-22 |

## 5. What is deliberately **not** tested

- Specific accuracy values on real datasets. These are experimental outcomes, not correctness properties, and
  hard-coding them would amount to inventing results.
- That PSO beats random search or the baseline. That is a research question (RQ1 and RQ2), not a correctness
  requirement.

## 6. Running the tests

```
pytest -q                       # full suite (< 5 min)
pytest tests/unit -q            # fast unit tests
pytest -m isolation -q          # IT-04…IT-07, UT-15, UT-22 (marker "isolation")
pytest -m closed_loop -q        # IT-01…IT-03 (marker "closed_loop")
```
