# RESULTS SCHEMA

This document defines exactly how experiment results are stored. The primary formats are CSV (tables) and JSON
(records), written UTF-8 with `\n` line endings. Every column has a fixed type so that the files load unchanged into
pandas or Parquet later.

**Type legend:**

| Type | Meaning |
|---|---|
| `str` | string |
| `int` | 64-bit integer |
| `Int?` | nullable integer (empty cell in CSV, `null` in JSON) |
| `float` | 64-bit float |
| `float?` | nullable float |
| `bool` | `true` / `false` in CSV |
| `ts` | ISO-8601 UTC timestamp, e.g. `2026-10-01T14:30:00.123Z` |
| `json` | JSON-encoded value in one CSV cell (Parquet: native list or struct) |

Floats are written with full precision (`repr`), and `-inf` is written as `-inf`.

**Timing fields** are `timestamp`, `fit_time_s`, `elapsed_s`, `total_time_s` and every `*_at` field. They are the
only fields allowed to differ between reproducible re-runs (RR-005).

---

## 1. Directory layout

```
results/<exp_id>/                         exp_id = <YYYYMMDD-HHMMSS>_<experiment.name>
├── config.resolved.json                  §2
├── manifest.json                         §3
├── run.log                               text log (not a schema file)
├── summary_folds.csv                     §9
├── summary.csv                           §10
└── <dataset>/
    ├── audit.json                        §4
    ├── fold_<k>/                         k = 0..4
    │   ├── baseline/       final.json (§7) · predictions.csv (§8)
    │   ├── random_search/  final.json · predictions.csv · evaluations.csv (§5)
    │   └── pso/            final.json · predictions.csv · evaluations.csv · iterations.csv (§6)
    └── deployment/
        └── pso/            deployment.json (§11) · evaluations.csv · iterations.csv
```

Dataset keys are `iris`, `digits` and `heart_cleveland`; method keys are `baseline`, `random_search` and `pso`.

## 2. `config.resolved.json`

This is the fully merged and validated configuration (ARCHITECTURE §H), with every default made explicit, including
`random_search.budget` resolved from `auto` to an integer. Its SHA-256 (computed over canonical JSON: sorted keys, no
whitespace) is the **config hash**.

## 3. `manifest.json`

| Field | Type | Description |
|---|---|---|
| `exp_id` | str | experiment identifier |
| `created_at` | ts | start time |
| `finished_at` | ts? | end time (null if interrupted) |
| `command` | str | full command line |
| `config_hash` | str | SHA-256 of the canonical resolved config |
| `git_commit` | str | `git rev-parse HEAD` |
| `git_dirty` | bool | uncommitted changes present |
| `python_version` | str | e.g. `3.10.11` |
| `platform` | str | `platform.platform()` |
| `cpu_count` | int | `os.cpu_count()` |
| `packages` | object | `{numpy, scikit-learn, pandas, matplotlib, pyyaml, joblib}` → version strings |
| `datasets` | object | per dataset: `{source, n_samples, n_features, n_classes, sha256_raw (Heart), sha256_arrays}` |
| `status` | str | `completed` \| `failed` \| `interrupted` |

`sha256_arrays` is the SHA-256 of `X.tobytes() + y.tobytes()` after loading. It guards against scikit-learn changing
a bundled dataset.

## 4. `<dataset>/audit.json`

| Field | Type | Description |
|---|---|---|
| `dataset` | str | dataset key |
| `n_samples`, `n_features`, `n_classes` | int | shape facts |
| `class_counts` | object | label → count |
| `class_ratio` | float | max count / min count |
| `fitness_metric` | str | metric chosen by the gate |
| `missing_per_feature` | object | feature → count of NaN |
| `n_duplicate_rows` | int | exact duplicate (X, y) rows |
| `duplicates_removed` | bool | true only if duplicates exceeded 1% |
| `feature_names`, `class_names` | json | names |

## 5. `evaluations.csv` (one row per proposed configuration; PSO and random search)

| # | Column | Type | Description |
|---|---|---|---|
| 1 | `exp_id` | str | experiment id |
| 2 | `dataset` | str | dataset key |
| 3 | `method` | str | `pso` \| `random_search` |
| 4 | `run_id` | str | `fold_<k>` or `deployment` |
| 5 | `outer_fold` | Int? | k (null for deployment) |
| 6 | `seed` | int | run seed |
| 7 | `eval_index` | int | 0-based order of evaluation within the run |
| 8 | `iteration` | Int? | PSO iteration, 0..T (null for random search) |
| 9 | `particle_id` | Int? | 0..N−1 (null for random search) |
| 10 | `pos_n_estimators` | float? | continuous position (PSO only) |
| 11 | `pos_max_depth` | float? | " |
| 12 | `pos_min_samples_split` | float? | " |
| 13 | `vel_n_estimators` | float? | velocity with which the particle arrived at this position (PSO only; the initial velocity at t = 0) |
| 14 | `vel_max_depth` | float? | " |
| 15 | `vel_min_samples_split` | float? | " |
| 16 | `n_estimators` | int | decoded hyperparameter that was evaluated |
| 17 | `max_depth` | int | " |
| 18 | `min_samples_split` | int | " |
| 19 | `fitness` | float | mean inner-CV score of `fitness_metric` (the value fed back); `-inf` if failed |
| 20 | `fitness_metric` | str | `accuracy` \| `balanced_accuracy` |
| 21 | `cv_scores` | json | list of 5 per-fold scores of the fitness metric |
| 22 | `cv_std` | float | standard deviation of `cv_scores` (ddof = 0) |
| 23 | `diag_balanced_accuracy` | float? | mean inner-CV balanced accuracy (diagnostic; never fed back) |
| 24 | `diag_f1_macro` | float? | mean inner-CV macro F1 (diagnostic; never fed back) |
| 25 | `cache_hit` | bool | result reused from this run's cache |
| 26 | `status` | str | `ok` \| `failed` |
| 27 | `error` | str? | exception message if failed |
| 28 | `pbest_fitness` | float? | this particle's pbest fitness **after** this iteration's update (PSO only) |
| 29 | `gbest_fitness` | float? | swarm gbest fitness **after** this iteration's update (PSO only) |
| 30 | `best_so_far_fitness` | float | running max of `fitness` over `eval_index` ≤ this row (both methods; used for anytime curves) |
| 31 | `fit_time_s` | float | wall time of this evaluation (0 for cache hits) |
| 32 | `timestamp` | ts | when the evaluation finished |

Row count: `budget` per run (default 210). For PSO, `eval_index = iteration × N + particle_id`.

## 6. `iterations.csv` (PSO only; one row per iteration t = 0..T)

| # | Column | Type | Description |
|---|---|---|---|
| 1–6 | `exp_id`, `dataset`, `method`, `run_id`, `outer_fold`, `seed` | as §5 | identification |
| 7 | `iteration` | int | t |
| 8 | `gbest_fitness` | float | γᵗ (non-decreasing) |
| 9 | `gbest_n_estimators` | int | decoded gbest configuration |
| 10 | `gbest_max_depth` | int | " |
| 11 | `gbest_min_samples_split` | int | " |
| 12 | `gbest_improved` | bool | γᵗ > γᵗ⁻¹ (true at t = 0) |
| 13 | `mean_fitness` | float | mean of the N particle fitnesses this iteration (failed = excluded, count in 21) |
| 14 | `std_fitness` | float | standard deviation of the same (ddof = 0) |
| 15 | `min_fitness` | float | minimum of the same |
| 16 | `max_fitness` | float | maximum of the same |
| 17 | `diversity` | float | mean Euclidean distance of the range-normalized positions from their centroid |
| 18 | `n_unique_configs` | int | distinct decoded configurations among the N particles |
| 19 | `n_cache_hits` | int | cache hits this iteration |
| 20 | `n_ties_with_gbest` | int | distinct configurations evaluated so far whose fitness equals γᵗ |
| 21 | `n_failed` | int | failed evaluations this iteration |
| 22 | `cumulative_evaluations` | int | N·(t+1) |
| 23 | `cumulative_unique_fits` | int | cache misses so far |
| 24 | `no_improve_count` | int | iterations since the last gbest improvement |
| 25 | `elapsed_s` | float | time since the run started |

## 7. `final.json` (per fold per method)

| Field | Type | Description |
|---|---|---|
| `exp_id`, `dataset`, `method`, `run_id`, `outer_fold`, `seed` | — | identification |
| `best_hyperparameters` | object | `{n_estimators: int, max_depth: int\|null, min_samples_split: int}`; `max_depth: null` only for the baseline |
| `best_validation_fitness` | float | fitness of the chosen configuration (baseline: its CV score). **Optimistically biased for the optimizers.** |
| `validation_cv_scores` | list[float] | per-fold scores of the chosen configuration |
| `fitness_metric` | str | metric used |
| `best_found_at_eval` | Int? | `eval_index` where the best was first found (null for the baseline) |
| `best_found_at_iteration` | Int? | PSO only |
| `n_evaluations` | int | proposed configurations (baseline: 1) |
| `n_unique_fits` | int | cache misses (baseline: 1) |
| `n_iterations` | Int? | PSO: iterations completed (T unless patience fired) |
| `stop_reason` | str? | `max_iter` \| `patience` (PSO); `budget` (random search); null (baseline) |
| `convergence_iteration` | Int? | PSO: last iteration with a gbest improvement |
| `convergence_status` | str? | PSO: `improving_at_end` if `convergence_iteration ≥ T − 2`, else `plateaued` |
| `n_ties_with_best` | Int? | distinct configurations with fitness equal to the best |
| `boundary_hits` | list[str] | names of hyperparameters whose chosen value equals a bound |
| `test_metrics` | object | see below |
| `n_opt_samples`, `n_test_samples` | int | sizes |
| `optimization_finished_at` | ts | when the optimization phase closed |
| `test_evaluated_at` | ts | when the test metrics were computed (must be later; ET-04) |
| `total_time_s` | float | wall time for the method on this fold |

`test_metrics` object:

| Field | Type |
|---|---|
| `accuracy`, `balanced_accuracy`, `precision_macro`, `recall_macro`, `f1_macro` | float |
| `positive_class_precision`, `positive_class_recall`, `positive_class_f1` | float? (Heart only; the positive label is 1) |
| `confusion_matrix` | list[list[int]] (rows = true, columns = predicted) |
| `labels` | list[int] (order of the matrix rows and columns) |

## 8. `predictions.csv` (per fold per method)

| Column | Type | Description |
|---|---|---|
| `row_index` | int | original index in the full dataset |
| `y_true` | int | true label |
| `y_pred` | int | predicted label |

Row count equals the test fold size. Concatenating the files over folds gives exactly one prediction per sample per
method, which is the basis for the pooled confusion matrices.

## 9. `summary_folds.csv` (one row per dataset × method × fold)

| Column | Type | Source |
|---|---|---|
| `exp_id`, `dataset`, `method`, `outer_fold`, `seed` | — | identification |
| `n_estimators`, `max_depth` (Int?), `min_samples_split` | int | `final.json` |
| `validation_fitness` | float | `final.json.best_validation_fitness` |
| `test_accuracy`, `test_balanced_accuracy`, `test_precision_macro`, `test_recall_macro`, `test_f1_macro` | float | `final.json.test_metrics` |
| `delta_accuracy_vs_baseline` | float | this method's test accuracy − baseline's, same fold |
| `delta_f1_macro_vs_baseline` | float | same for macro F1 |
| `delta_accuracy_vs_random_search` | float? | PSO rows only |
| `n_evaluations`, `n_unique_fits` | int | `final.json` |
| `convergence_iteration` | Int? | PSO rows |
| `total_time_s` | float | `final.json` |

## 10. `summary.csv` (one row per dataset × method)

| Column | Type | Description |
|---|---|---|
| `exp_id`, `dataset`, `method` | str | identification |
| `n_folds` | int | 5 |
| `test_accuracy_mean`, `test_accuracy_std` | float | over folds (std with ddof = 1) |
| `test_balanced_accuracy_mean`, `test_balanced_accuracy_std` | float | " |
| `test_f1_macro_mean`, `test_f1_macro_std` | float | " |
| `pooled_test_accuracy` | float | accuracy over all concatenated out-of-fold predictions |
| `validation_fitness_mean` | float | biased; labelled as such in reports |
| `delta_accuracy_mean`, `delta_accuracy_std` | float | vs. the baseline, paired by fold |
| `wins`, `ties`, `losses` | int | fold-level test accuracy vs. the baseline |
| `wins_vs_rs`, `ties_vs_rs`, `losses_vs_rs` | Int? | PSO rows only |
| `n_unique_fits_mean` | float | mean over folds |
| `total_time_s_sum` | float | over folds |

## 11. `deployment.json` (per dataset)

| Field | Type | Description |
|---|---|---|
| `exp_id`, `dataset`, `seed` | — | seed = 5 |
| `recommended_hyperparameters` | object | decoded gbest on the full dataset |
| `validation_fitness` | float | inner-CV fitness on the full dataset; **biased; not a performance estimate** |
| `performance_estimate` | object | copied from `summary.csv` PSO row: `{test_accuracy_mean, test_accuracy_std, source: "summary.csv"}` |
| `n_samples` | int | full dataset size |
| `n_evaluations`, `n_unique_fits`, `convergence_iteration` | int / Int? | as in `final.json` |
| `note` | str | fixed text: "No held-out test data exists for the deployment run; performance is estimated by the outer 5-fold CV of the same procedure." |

## 12. Parquet compatibility

- Column names are snake_case ASCII with no spaces.
- Nullable integers map to pandas `Int64`.
- `json` columns (`cv_scores`) decode to `list<double>`.
- Timestamps map to `timestamp[ms, tz=UTC]`.
- Conversion is one line per file (`pd.read_csv(..., dtype=SCHEMA).to_parquet(...)`), with the dtype maps defined
  once in `experiments/recorder.py`.
