# CONTEXT — Technical reference for implementation

This is the main reference for implementers. It separates what is **CONFIRMED** (fixed by the PDF or by the
project owner), what is **PROPOSED** (a design default adopted in Phase 0, changeable only through an ADR), and what
is **TO VERIFY** (depends on facts that will only be known during implementation).

Related documents:
- [MATHEMATICAL_FORMULATION.md](MATHEMATICAL_FORMULATION.md): the notation used here.
- [ARCHITECTURE.md](ARCHITECTURE.md): diagrams.
- [DECISIONS.md](DECISIONS.md): the reasoning behind each decision.

---

## 1. Current project state (2026-09-24)

| Item | State |
|---|---|
| Phase | **Phase 0 complete when these docs are committed.** No implementation code exists. |
| Repository | `C:\Users\anush\Desktop\PSO`, remote `https://github.com/CoderAnush/pso-random-forest-optimization` |
| Source of truth | `ppt/CB.EN.U4ELC23005_ANUSH_RAMESH_PPT.pdf` (13 slides, image-only; slide 12's references exist only in the PDF text layer), plus the decisions in [DECISIONS.md](DECISIONS.md) |
| Environment (measured) | Windows 11, 20 CPU cores, Python 3.10.11, numpy 1.26.4, scikit-learn 1.7.2, pandas 2.3.3, matplotlib 3.10.6, PyYAML 6.0.1, pytest 9.1.1, joblib 1.5.2 |
| Datasets on disk | none yet. Iris and Digits ship with scikit-learn; Cleveland is downloaded in Phase 2 |
| Results | none. **No result values exist anywhere.** |

## 2. Confirmed requirements and decisions

### 2.1 Confirmed by the PDF

| # | Fact | Slide |
|---|---|---|
| C1 | The system being optimized is a Random Forest classifier | 6 |
| C2 | The optimizer is PSO, implemented from scratch (initialization, fitness evaluation, velocity and position update) | 4, 9 |
| C3 | Decision vector $x = [n\_estimators, max\_depth, min\_samples\_split]$, all integers | 6, 7 |
| C4 | Bounds: 50–200, 2–20, 2–10 | 6, 7, 8 |
| C5 | Objective: maximize validation fitness (classification performance) | 6, 7 |
| C6 | Swarm size: 10 particles | 10 |
| C7 | Loop order: initialize → evaluate → update pbest → update gbest → update velocity → update position → apply bounds → repeat | 9, 10 |
| C8 | Stop at convergence or maximum iterations | 6, 9, 10 |
| C9 | Test data is completely isolated, used once after optimization, never as PSO fitness | 6, 7 |
| C10 | The final model is refit on the full training data with the best hyperparameters | 6 |
| C11 | Output metrics: Accuracy, Precision, Recall, F1 | 6, 10 |
| C12 | A reproducible default-RF baseline, with the aim to "improve or maintain" performance | 3, 4, 11 |
| C13 | Stability is studied over multiple random seeds; convergence is analysed | 3, 4, 11 |
| C14 | Datasets: Iris (3-class), Digits (10-class), Heart Disease (binary) | 5 |
| C15 | Fitness uses stratified k-fold CV on training data only, as mean validation accuracy | 4 (Obj 04), 10 |

### 2.2 Confirmed by the project owner (2026-09-24; these override the PDF where they differ)

| # | Decision | PDF difference | ADR |
|---|---|---|---|
| U1 | The inner fitness estimator is **5-fold** stratified CV, with folds fixed per run | PDF slide 10 says 3-fold | ADR-005 |
| U2 | The fitness metric is **accuracy**, configurable to `balanced_accuracy`, with a class-ratio gate of 1.5 | same as PDF, plus a gate | ADR-006 |
| U3 | The test protocol is an **outer stratified 5-fold (nested CV)** plus a deployment run | PDF shows one train/test split | ADR-007 |
| U4 | The comparators are the **default RF and an equal-budget random search** | PDF has the default RF only | ADR-008 |
| U5 | Heart Disease is the **UCI Cleveland** subset (303 rows); the raw file is committed with its SHA-256 | PDF cites UCI without naming a subset | ADR-009 |
| U6 | The project has its own git repository with a GitHub remote | — | — |

## 3. Proposed design defaults (Phase 0; change only via an ADR)

| # | Decision | Value | ADR |
|---|---|---|---|
| P1 | Integer handling | continuous state; `decode(x) = clip(rint(x), lb, ub)` at evaluation only | ADR-011 |
| P2 | Boundary handling | absorbing wall: clip the position and zero that velocity component | ADR-012 |
| P3 | Velocity limits | $v_{max} = 0.2\times$range = (30, 3.6, 1.6); initial velocity $U(\pm 0.1\times$range$)$ = ±(15, 1.8, 0.8) | ADR-012 |
| P4 | Coefficients | $w = 0.7298$, $c_1 = c_2 = 1.49618$ (constriction-equivalent) | ADR-013 |
| P5 | Iterations | `max_iter = 20` → 10 × 21 = 210 evaluations per PSO run | ADR-014 |
| P6 | Topology and update | global best (gbest) topology, synchronous update | ADR-014 |
| P7 | Stopping | at `max_iter`; patience (tol = 1e-4, P = 5) available but **off** | ADR-015 |
| P8 | Cache | per run, keyed by integer configuration; hits logged as evaluations | ADR-016 |
| P9 | Ties | strict `>`; same-iteration gbest ties go to the lowest particle index; random search ties go to the earliest evaluation | ADR-017 |
| P10 | Seeds | outer seed 42; run seed *k* for outer fold *k* (0–4); deployment seed 5 | ADR-018 |
| P11 | Parallelism | RF `n_jobs=1`; the 5 inner CV folds run in parallel (`n_jobs=5`) | ADR-019 |
| P12 | Preprocessing | Heart: `SimpleImputer(strategy="most_frequent")` inside the pipeline; others: none | ADR-010 |
| P13 | Package layout | `src/pso_rf/…`; the optimization layer imports only NumPy and the standard library | ADR-023 |
| P14 | Test-set guard | the `HeldOutTestSet` wrapper plus an `OptimizationPhase` context that forbids access | ADR-024 |
| P15 | Config format | YAML, layered as defaults ← experiment file ← dataset override ← CLI | ADR-025 |
| P16 | Reporting | descriptive statistics only; validation fitness is never presented as performance | ADR-020 |

## 4. To verify during implementation

| # | Item | Expected (from literature or measurement) | Verify in | How |
|---|---|---|---|---|
| V1 | Cleveland row count, missing values, class split | 303 rows; `ca`: 4 missing, `thal`: 2 missing; `num > 0`: about 139, `num = 0`: about 164 | P2 | dataset audit script output, saved to `data/DATASET_AUDIT.md` |
| V2 | Cleveland duplicates | none expected | P2 | audit |
| V3 | Cleveland download URL and SHA-256 | URL `https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data` (the UCI archive may redirect; fallback is the `heart+disease.zip` bundle) | P2 | download script records both |
| V4 | Class-ratio gate outcome | none triggers (Iris 1.00, Digits 1.05, Cleveland ≈ 1.18) | P2 | audit |
| V5 | Evaluation time and total runtime | worst case per evaluation (5-fold, parallel folds): Iris 0.24 s, Digits 0.52 s (measured); Heart similar to Iris; total ≈ 40 min | P6 | timing benchmark → decision gate (ADR-005) |
| V6 | Parallel folds give identical scores to serial | expected identical | P6 | IT-08 |
| V7 | Iris duplicate row | 1 (measured); kept | P2 | audit |

## 5. Architecture summary

```
Dataset ─▶ Outer split (5 folds) ─┬─▶ Optimization portion ─▶ [ CLOSED LOOP: PSO ⇄ RF + inner 5-fold CV ] ─▶ best config
                                  │                                                                        │
                                  │                                                                        ▼
                                  └─▶ Held-out test fold ───────────────────────────────▶ Final RF refit on the optimization portion
                                      (inaccessible during the loop)                        └─▶ test metrics (one-way; never fed back)
```

The full diagrams are in [ARCHITECTURE.md](ARCHITECTURE.md).

## 6. Data flow

1. **Load:** `load_dataset(name)` returns a `DatasetBundle` with X (float array, NaN for missing), y (int labels),
   names and metadata (source, checksum, audit results).
2. **Outer split:** `outer_folds(bundle, n_splits=5, seed=42)` yields 5 `FoldData` objects. Each contains an
   `OptimizationData` (X_opt, y_opt, original row indices) and a `HeldOutTestSet` (X_test, y_test, indices, guarded).
3. **Inner CV:** `FitnessEvaluator(opt_data, cv_folds=5, seed=k, metric, preprocessing)` builds its
   `StratifiedKFold` once. The fold indices are **fixed** for the evaluator's lifetime.
4. **Evaluate:** each configuration → `build_model(config, seed, preprocessing)` → `cross_validate` on the fixed inner
   folds → `FitnessResult` (fitness, per-fold scores, diagnostics).
5. **Refit and test:** after the optimizer returns, `final_evaluate(config, opt_data, test_set, seed, preprocessing)`
   fits on all of `opt_data` and predicts `test_set`. This is the **only** code path that reads test data.
6. **Persist:** the recorder writes the evaluation and iteration rows during the run; final JSON, predictions and the
   summaries are written after it.

## 7. Control flow (one run = one dataset × one outer fold × one method)

```
run_fold(dataset, k, method):
    fold = folds[k]                                   # opt_data, test_set
    evaluator = FitnessEvaluator(fold.opt, cv=5, seed=k, ...)
    with OptimizationPhase():                         # test_set.reveal() raises inside this block
        if method == "pso":            result = PSOOptimizer(space, evaluator.as_objective(), cfg, rng(k)).run(recorder)
        elif method == "random_search": result = RandomSearch(space, evaluator.as_objective(), 210, rng(k)).run(recorder)
        elif method == "baseline":      result = FixedConfig(BASELINE).score(evaluator)   # CV score only, for reference
    # optimization phase closed; best configuration is now frozen
    metrics = final_evaluate(result.best_config, fold.opt, fold.test, seed=k, ...)
    save(result, metrics)
```

The experiment loops over datasets × folds × methods and then runs the deployment run for each dataset. Order is
deterministic: datasets in config order, folds 0–4, and methods in the order baseline, random_search, pso.

## 8. Module responsibilities

| Module (`src/pso_rf/…`) | Responsibility | May import | Must not import |
|---|---|---|---|
| `datasets/` | Load, checksum-verify and audit datasets; return a `DatasetBundle` | numpy, pandas, sklearn.datasets | optimization, experiments |
| `preprocessing/` | Build the preprocessing steps declared in config | sklearn | optimization |
| `models/` | Build RF pipelines from a configuration; define the baseline configuration | sklearn, preprocessing | optimization, datasets |
| `evaluation/` | Outer folds and test guard (`splits.py`), fitness evaluator and cache (`fitness.py`), final test evaluation (`final.py`), metrics (`metrics.py`) | sklearn, numpy, models | **optimization** |
| `optimization/` | `SearchSpace`, `PSOOptimizer`, `RandomSearch`, event types | numpy, stdlib **only** | sklearn, models, datasets, evaluation |
| `experiments/` | Config loading and validation, seeding, the recorder, the runner (the only place the loop is wired), CLI | everything above | — |
| `visualization/` | Plots produced **from saved result files only** | pandas, matplotlib | models, optimization, evaluation |
| `utils/` | Hashing, I/O helpers, manifest, logging setup | stdlib, numpy | — |

## 9. Interfaces between modules

The signatures below are **specifications**; they are implemented in their phase. Types use Python 3.10 syntax.

```python
# datasets
@dataclass(frozen=True)
class DatasetBundle:
    name: str
    X: np.ndarray            # shape (n, d), float64, NaN = missing
    y: np.ndarray            # shape (n,), int64, labels 0..C-1
    feature_names: list[str]
    class_names: list[str]
    meta: dict               # source, citation, sha256, audit {n, d, class_counts, duplicates, missing}
def load_dataset(name: str, data_dir: Path) -> DatasetBundle

# evaluation.splits
@dataclass(frozen=True)
class OptimizationData:  X: np.ndarray; y: np.ndarray; indices: np.ndarray
class HeldOutTestSet:        # arrays are private; access only via reveal()
    indices: np.ndarray
    def reveal(self) -> tuple[np.ndarray, np.ndarray]   # raises TestSetAccessError inside OptimizationPhase
class OptimizationPhase:     # context manager; sets a process-wide flag
@dataclass(frozen=True)
class FoldData: fold_index: int; opt: OptimizationData; test: HeldOutTestSet
def outer_folds(bundle: DatasetBundle, n_splits: int, seed: int) -> list[FoldData]

# optimization (generic — no ML knowledge)
@dataclass(frozen=True)
class IntParam: name: str; low: int; high: int
class SearchSpace:
    def __init__(self, params: Sequence[IntParam])
    lower: np.ndarray; upper: np.ndarray; names: tuple[str, ...]
    def decode(self, position: np.ndarray) -> dict[str, int]      # clip(rint(position))
    def clip(self, position: np.ndarray) -> np.ndarray
    def sample(self, rng: np.random.Generator) -> dict[str, int]  # uniform over integer grid
@dataclass(frozen=True)
class Evaluation:            # what an objective returns
    fitness: float           # the ONLY value the optimizer uses
    info: Mapping[str, Any]  # forwarded untouched to callbacks (fold scores, cache_hit, timing…)
Objective = Callable[[dict[str, int]], Evaluation]
class PSOOptimizer:
    def __init__(self, space: SearchSpace, objective: Objective, config: PSOConfig, rng: np.random.Generator)
    def run(self, callbacks: Sequence[Callback] = ()) -> OptimizationResult
class RandomSearch:
    def __init__(self, space: SearchSpace, objective: Objective, budget: int, rng: np.random.Generator)
    def run(self, callbacks: Sequence[Callback] = ()) -> OptimizationResult
class Callback(Protocol):
    def on_evaluation(self, event: EvaluationEvent) -> None
    def on_iteration_end(self, summary: IterationSummary) -> None   # PSO only
@dataclass(frozen=True)
class OptimizationResult:
    best_config: dict[str, int]; best_fitness: float; best_position: np.ndarray | None
    n_evaluations: int; n_iterations: int | None; stop_reason: str
    convergence_iteration: int | None; history: list[IterationSummary]

# evaluation.fitness
@dataclass(frozen=True)
class FitnessResult: fitness: float; cv_scores: list[float]; diagnostics: dict[str, float]
                     cache_hit: bool; fit_time_s: float; status: str; error: str | None
class FitnessEvaluator:
    def __init__(self, opt: OptimizationData, cv_folds: int, seed: int, metric: str,
                 preprocessing: PreprocessingSpec, n_jobs_folds: int, cache: bool = True)
        # raises TypeError if given a HeldOutTestSet
    def __call__(self, config: dict[str, int | None]) -> FitnessResult
    def as_objective(self) -> Objective       # adapter: FitnessResult -> Evaluation
    n_unique_fits: int

# evaluation.final
def final_evaluate(config: dict[str, int | None], opt: OptimizationData, test: HeldOutTestSet,
                   seed: int, preprocessing: PreprocessingSpec) -> FinalMetrics

# experiments
def run_experiment(config: ExperimentConfig) -> Path    # returns results/<exp_id>/
```

**Coupling rule.** The optimizer sees only `SearchSpace` and `Objective`. The evaluator sees only
`OptimizationData`. The **runner** in `experiments/` is the only component that knows about both, and it is where
the closed loop is assembled.

## 10. Mathematical formulation (summary)

The full version is in [MATHEMATICAL_FORMULATION.md](MATHEMATICAL_FORMULATION.md).

- $\Omega = \{50..200\} \times \{2..20\} \times \{2..10\}$, with $|\Omega| = 25{,}821$.
- $\mathrm{dec}(x) = \mathrm{clip}(\mathrm{rint}(x), l, u)$.
- $F_k(\theta) = \frac{1}{5}\sum_{j=1}^{5} \mathrm{acc}\big(\mathrm{RF}_{\theta,s_k}(\mathcal{D}^{(k)}_{opt}\setminus Q^{(k)}_j);\ Q^{(k)}_j\big)$.
- Goal: $\max_{\theta\in\Omega} F_k(\theta)$.
- PSO update: $v \leftarrow \mathrm{clip}(w v + c_1 r_1 (p-x) + c_2 r_2 (g-x), \pm v_{max})$, then $x \leftarrow x+v$ with
  an absorbing wall.
- The test metric $M_k$ is defined only on $\hat\theta_k$ and $\mathcal{D}^{(k)}_{test}$, and never enters $F_k$.

## 11. PSO state

| State element | Shape | Type | Meaning |
|---|---|---|---|
| `X` | (N, 3) | float64 | current continuous positions |
| `V` | (N, 3) | float64 | current velocities |
| `fitness` | (N,) | float64 | fitness of `dec(X[i])` at the current iteration |
| `P` | (N, 3) | float64 | continuous personal-best positions |
| `P_fit` | (N,) | float64 | personal-best fitness |
| `P_cfg` | N × dict | int | decoded personal-best configurations |
| `g`, `g_fit`, `g_cfg` | (3,), scalar, dict | — | global best position, fitness and configuration |
| `t` | scalar | int | iteration index: 0 = initial evaluation; 1..T = update iterations |
| `rng` | — | `np.random.Generator` | the only source of randomness |
| `no_improve` | scalar | int | iterations since the last gbest improvement (patience) |

All of this state lives in the optimizer. None of it is visible to the evaluator.

## 12. Random Forest evaluation

- **Pipeline:** `[SimpleImputer(most_frequent)]` (Heart only) → `RandomForestClassifier(n_estimators, max_depth,
  min_samples_split, random_state=seed, n_jobs=1)`. All other RF parameters are left at scikit-learn defaults:
  `criterion="gini"`, `max_features="sqrt"`, `bootstrap=True`, `min_samples_leaf=1`.
- **Scoring:** `cross_validate(pipeline, X_opt, y_opt, cv=fixed_folds, scoring={"accuracy", "balanced_accuracy",
  "f1_macro"}, n_jobs=5, error_score="raise")`. Fitness is the mean of the configured metric. The other two scores
  are diagnostics only.
- **Failure handling:** an exception is caught, logged, recorded as `status = "failed"`, and scored
  `fitness = -inf`. A NaN score is treated the same way. The run continues.
- **Cache:** the key is `(n_estimators, max_depth, min_samples_split)`. A hit returns the stored result with
  `cache_hit = True` and `fit_time_s = 0`.

## 13. Experiment lifecycle

1. Load and validate the config; resolve all layers; compute the config hash; create `results/<exp_id>/`.
2. Write `config.resolved.json` and `manifest.json` (environment, git state, dataset checksums).
3. For each dataset:
   1. Load and audit it; apply the class-ratio gate to choose the metric.
   2. Build the 5 outer folds.
   3. For each fold *k*, run `baseline`, then `random_search`, then `pso`. Each run optimizes inside
      `OptimizationPhase`, then evaluates on the test fold once, then saves its outputs.
   4. Run the deployment PSO on the full dataset (seed 5).
4. Aggregate `summary_folds.csv` and `summary.csv` from the saved per-fold files.
5. The plot command (separate from the run) reads the result files and writes figures to `plots/<exp_id>/`.

## 14. Output artifacts

The full field definitions are in [RESULTS_SCHEMA.md](RESULTS_SCHEMA.md).

```
results/<exp_id>/
  config.resolved.json   manifest.json   run.log   summary.csv   summary_folds.csv
  <dataset>/
    audit.json
    fold_<k>/
      baseline/        final.json  predictions.csv
      random_search/   final.json  predictions.csv  evaluations.csv
      pso/             final.json  predictions.csv  evaluations.csv  iterations.csv
    deployment/pso/    deployment.json  evaluations.csv  iterations.csv
plots/<exp_id>/*.png   plots/<exp_id>/SOURCES.json
```

## 15. Logging

- Python `logging`; library code never uses `print`.
- The console shows INFO: one line per PSO iteration, in the form
  `iter t | gbest (n,d,s) = fitness | mean = … | unique fits = …`. This is the live demonstration trace.
- `run.log` holds DEBUG detail, including one line per evaluation.
- Every log line of a run carries its context: dataset, fold, method and seed.

## 16. Configuration

- `configs/default.yaml` holds all defaults. The full template is in [ARCHITECTURE.md](ARCHITECTURE.md) §H.
- `configs/demo.yaml` is a small budget for live demos and smoke tests (e.g. Iris, fold 0, N = 6, T = 5).
- Layering: `default.yaml` ← experiment file ← `datasets.<name>` overrides ← CLI `--set key=value`.
- Validation at load: `low < high`; integer bounds; `n_particles ≥ 2`; `max_iter ≥ 1`; `0 < w`; `c1, c2 ≥ 0`;
  `run_seeds` has exactly `outer_folds` entries; `metric ∈ {accuracy, balanced_accuracy}`.

## 17. Reproducibility

| Seed | Value | Drives |
|---|---|---|
| `outer_seed` | 42 | outer `StratifiedKFold` shuffle |
| `run_seeds[k]` | *k* ∈ {0, 1, 2, 3, 4} | for fold *k*: PSO generator, random search generator, inner CV shuffle, RF `random_state` (baseline, candidates, final refit) |
| `deployment_seed` | 5 | the same roles for the deployment run |

Determinism guarantees:
- one `np.random.Generator` per optimizer;
- no global random state;
- fixed inner folds;
- RF `random_state` set on every fit;
- parallelism only across CV folds, which scikit-learn makes deterministic because each fold clones the same
  estimator with the same `random_state`;
- sorted, deterministic iteration order.

Re-runs must reproduce the result files exactly, excluding timestamps and timings (RR-005).
