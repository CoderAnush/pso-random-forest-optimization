# CLAUDE.md — Master development instructions

These are the rules for any development session (human or AI) on **PSO-Based Random Forest Hyperparameter
Optimization**. This file summarizes and **enforces** the decisions made in the other documents. If this file and
another document disagree, stop and resolve it: fix the stale document through an ADR in DECISIONS.md. Never
silently pick one.

> **Do not break the closed-loop architecture.**
>
> **Never use held-out test data during PSO optimization.**

---

## 1. Project overview

PSO (implemented from scratch) tunes three Random Forest hyperparameters on Iris, Digits and Heart Disease
(UCI Cleveland), as a **closed feedback loop**:

1. PSO proposes candidate hyperparameters.
2. A Random Forest is trained with them.
3. Its inner-CV validation accuracy is returned to PSO as fitness.
4. PSO updates pbest, gbest, velocities and positions, which produces new hyperparameters.
5. This repeats for 20 iterations.

After the loop ends, the best configuration is refit and scored **once** on a held-out test fold. This is an
Evolutionary Optimization mini-project (Anush Ramesh, CB.EN.U4ELC23005, EEE, Amrita Vishwa Vidyapeetham, Coimbatore).

## 2. Objective

Maximize $F_k(\theta)$, the mean stratified 5-fold CV accuracy of RF(θ) on the optimization portion of outer fold *k*.
Then report honest held-out test performance, compared with the default RF and with an equal-budget random search
(open-loop control).

## 3. Academic requirements (faculty)

- One optimization technique, **PSO**, maximizes a defined objective.
- The optimizer's output changes a system parameter (the RF hyperparameters).
- There is **closed-loop feedback**, with the optimizer as its own block.
- Review 1 shows: block diagram, methods, objective, constraints, decision variables, program snippet, technique and
  why it was chosen. The deck is **≤ 10 slides**.
- Review 2 shows the whole system as a closed loop in which the output is fed back.

Mappings: [REVIEW_1_MAPPING.md](docs/REVIEW_1_MAPPING.md), [REVIEW_2_MAPPING.md](docs/REVIEW_2_MAPPING.md).

## 4. Source-of-truth rules

1. `ppt/CB.EN.U4ELC23005_ANUSH_RAMESH_PPT.pdf` is the original specification. Its slides are images; read them
   visually. Slide 12's references exist only in the text layer.
2. [DECISIONS.md](docs/DECISIONS.md) records where the design **intentionally** departs from the PDF (ADR-005, 007, 008).
   Those ADRs override the PDF.
3. Do not replace PSO with another optimizer, Random Forest with another model, or the three decision variables with
   others. Extensions go in separate, named, optional experiments.
4. When something is ambiguous, write the ambiguity and a proposed resolution in CONTEXT.md or DECISIONS.md, and ask
   the owner. Never assume silently.

## 5. How the PDF is interpreted (with the accepted changes)

| PDF says | Implemented as | ADR |
|---|---|---|
| Train/test split (slide 6) | **Outer stratified 5-fold**; fold *k* is run *k*'s held-out test fold; plus a deployment run on all data | 007 |
| 3-fold stratified CV, mean accuracy (slide 10) | **5-fold** stratified CV, mean **accuracy**, folds fixed per run | 005, 006 |
| Compare against the default-RF baseline (slide 4) | Default RF **and** equal-budget random search | 008 |
| Heart Disease, binary (slide 5) | UCI **Cleveland** `processed.cleveland.data`, `num > 0 → 1` | 009 |
| 10 particles (slide 10) | N = 10 | 014 |
| Apply bounds (slides 9, 10) | Absorbing wall, velocity clamp 0.2 × range | 012 |
| Integer-valued parameters (slide 6) | Continuous state; `decode = clip(rint(x))` at evaluation | 011 |
| Until convergence or maximum iterations | T = 20; patience available but off | 015 |

## 6. System boundaries

| Inside the system | Outside the system |
|---|---|
| Loading 3 datasets; outer and inner CV; the RF pipeline; PSO; random search; recording; summaries; plots; the interactive review demo (`src/pso_rf/app/`, ADR-026) | Other models; other optimizers (except as future comparators); GPUs or clusters; a production web service; hyperparameters beyond the 3 in the core experiment |

## 7. Architecture principles (mandatory)

The demo (`src/pso_rf/app/`) follows these rules too: live runs go through `run_fold` (observers only), and the
results and replay pages read saved files. Never give the demo its own copy of the loop (ADR-026).

1. **Closed loop first.** PSO ⇄ evaluator are wired **only** in `pso_rf/experiments/runner.py`. There is no
   disconnected PSO script and no "optimize, then separately evaluate" shortcut.
2. **No test data leakage.** Test arrays live in `HeldOutTestSet` and are reachable only via `reveal()`, which
   raises inside `OptimizationPhase`. Only `final_evaluate` reads them, after the loop.
3. **Reproducible.** Every seed comes from config. There is one `np.random.Generator` per optimizer and no global
   random state.
4. **Modular.** `optimization/` imports **only NumPy and the standard library**. It has no scikit-learn, no data and
   no dataset names. `evaluation/` **never** imports `optimization/`. UT-22 enforces both.
5. **Configurable.** PSO parameters, bounds, metric, folds and seeds all come from YAML config (global, with
   per-dataset override).
6. **Traceable.** Every evaluation is logged with dataset, method, fold, seed, iteration, particle, position,
   velocity, hyperparameters, fitness, pbest and gbest.
7. **Experimental.** Results are saved to `results/<exp_id>/`, not just printed.
8. **No fake results.** Never write an accuracy, fitness, configuration, runtime or convergence claim that was not
   read from a saved result file.
9. **No leakage through preprocessing.** Anything that learns from data (the Heart imputer) is a pipeline step,
   fitted per fit.
10. **Final test only after optimization.** Refit the chosen configuration on the whole optimization portion, then
    test once.

Diagrams: [ARCHITECTURE.md](docs/ARCHITECTURE.md). Interfaces: [CONTEXT.md](docs/CONTEXT.md) §9.

## 8. The closed-loop requirement

```
PSO → candidate hyperparameters → Random Forest → inner-CV validation → fitness ─┐
 ▲                                                                              │
 └──────────── PSO update (pbest, gbest, velocity, position) ◀──────────────────┘
                                   (20 iterations × 10 particles)
loop ends → best configuration → final RF (refit) → HELD-OUT TEST FOLD → metrics   [one-way; never fed back]
```

| Control concept | Project element |
|---|---|
| Plant | RF training plus inner-CV validation |
| Controller | PSO |
| Manipulated parameters | the 3 hyperparameters |
| Measurement / feedback | inner-CV accuracy = fitness |
| Adaptation | pbest/gbest memory, velocity and position update |
| One cycle / one step | one particle evaluation / one swarm iteration |

**Honest framing:** this is an iterative optimization loop with a static plant and no setpoint (extremum-seeking),
not a physical real-time controller. Random search is the open-loop contrast. See ADR-022.

## 9. PSO requirements

- Standard form: $v \leftarrow \mathrm{clip}(w v + c_1 r_1 (p - x) + c_2 r_2 (g - x), \pm v_{max})$; $x \leftarrow x + v$.
- Absorbing wall: clip x to the bounds and set the offending v component to 0.
- $w = 0.7298$, $c_1 = c_2 = 1.49618$; $r_1, r_2 \sim U[0,1)$ per particle, per dimension, per iteration.
- N = 10, T = 20, so 210 evaluations. Initial positions U[l, u]; initial velocities U(±0.1 × range); $v_{max}$ =
  0.2 × range = (30, 3.6, 1.6).
- gbest topology, **synchronous** update, strict `>` for pbest and gbest, lowest particle index on same-iteration
  ties.
- The state stays continuous; decode with `clip(rint(x))` only for evaluation.
- Stopping at `max_iter`; patience (tol 1e-4, P = 5) is **disabled by default**. Always record the convergence
  iteration and `stop_reason`.
- **From scratch.** pyswarms, Optuna, scikit-optimize, Hyperopt and similar are forbidden.

Full formulation: [MATHEMATICAL_FORMULATION.md](docs/MATHEMATICAL_FORMULATION.md).

## 10. Random Forest requirements

- `RandomForestClassifier(n_estimators, max_depth, min_samples_split, random_state=run_seed, n_jobs=1)`. All other
  parameters are left at scikit-learn 1.7.2 defaults.
- Pipeline: `[SimpleImputer(most_frequent)]` for Heart only, then the RF.
- Baseline: `RandomForestClassifier()` defaults (100, None, 2), with the same seed, pipeline and protocol. It lies
  outside the search space by design.

## 11. Datasets

| Key | Source | Target | Notes |
|---|---|---|---|
| `iris` | `sklearn.datasets.load_iris` | species (3) | 150 × 4, 50/50/50, 1 duplicate row (kept) |
| `digits` | `sklearn.datasets.load_digits` | digit (10) | 1797 × 64, classes 174–183 |
| `heart_cleveland` | UCI id 45 `processed.cleveland.data`, committed in `data/raw/` with its SHA-256 | `num > 0 → 1` | 303 × 13, 164 / 139; `?` → NaN in `ca` (4) and `thal` (2); 0 duplicate rows (measured in Phase 2, `data/DATASET_AUDIT.md`) |

## 12. Decision variables, search space, objective, fitness

- $x = [n\_estimators, max\_depth, min\_samples\_split]$.
- $50 \le n\_estimators \le 200$, $2 \le max\_depth \le 20$, $2 \le min\_samples\_split \le 10$, all integers
  (|Ω| = 25,821).
- Objective: **maximize** $F_k(\theta)$.
- Fitness: the mean of the 5 inner stratified CV scores of `fitness.metric` (default `accuracy`). The class-ratio gate
  switches to `balanced_accuracy` above a max/min class ratio of 1.5. Balanced accuracy and macro F1 are logged as
  diagnostics and **never fed back**.

## 13. Data leakage and train/validation/test rules

1. The outer split is `StratifiedKFold(5, shuffle=True, random_state=42)`. Fold *k* is the **held-out test fold** of
   run *k*.
2. The inner split is `StratifiedKFold(5, shuffle=True, random_state=k)` on the optimization portion, **fixed for the
   whole run**. These inner folds are the "validation data".
3. The test fold must not influence particle fitness, pbest, gbest, velocity, position, the stopping condition,
   hyperparameter selection, or preprocessing statistics.
4. Every optimization runs inside `with OptimizationPhase():`. `final_evaluate` runs after it closes, once per method
   per fold.
5. The deployment run (full dataset, seed 5) has **no** test score. Its performance estimate is the outer-CV mean.
6. Nothing is fitted on the full dataset. The audit only counts things.

## 14. Reproducibility requirements

- Seeds: outer 42; run seed *k* (0–4) drives the PSO generator, the random search generator, the inner CV shuffle
  and RF `random_state` for fold *k*; the deployment seed is 5.
- Save `config.resolved.json` (with its hash) and `manifest.json` (git SHA and dirty flag, versions, platform, CPU
  count, dataset checksums).
- Re-running a config must reproduce all result files exactly, excluding timing fields (IT-08).
- Dependencies are pinned in `requirements.txt`.

## 15. Coding conventions

- Python 3.10, `src/pso_rf/` layout, installed with `pip install -e .`.
- Type hints on all public functions. Frozen dataclasses for configs, events and results. Short docstrings.
- NumPy `Generator` only: never `np.random.seed`, `np.random.rand` or the `random` module.
- `pathlib` for paths, the `logging` module (no `print` in library code), and atomic file writes (`utils/io.py`).
- Small functions and no premature abstraction. Match the existing code's naming and comment density.
- Windows: guard the CLI entry points with `if __name__ == "__main__":` (loky process spawning).
- Tooling: `pytest`; `ruff` for lint and format (configured in `pyproject.toml`).

## 16. Project structure

See [PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md). Key paths:
- `CLAUDE.md` and `README.md` at the repo root; the other 13 design documents in `docs/`
- `src/pso_rf/{datasets,preprocessing,models,evaluation,optimization,experiments,visualization,utils,app}/`;
  `app.py` + `.streamlit/config.toml` launch the demo
- `configs/`, `data/raw/`, `results/<exp_id>/`, `plots/<exp_id>/`, `tests/{unit,integration,experiment}/`

## 17. Testing requirements

See [TESTING_STRATEGY.md](docs/TESTING_STRATEGY.md).
- Test the optimizer with stub objectives (no ML) and the evaluator without PSO.
- The **isolation proofs** must pass before any experiment is reported: UT-15, UT-22, IT-04 (indices), IT-05 (spy),
  IT-06 (test-label invariance), IT-07 (test-feature invariance) and ET-04 (timing order).
- The **closed-loop proofs** must pass: IT-01 (stub loop converges), IT-02 (feedback ablation) and IT-03 (real RF
  loop).
- Never assert specific accuracy values from real datasets.
- The full suite runs in under 5 minutes (`configs/test.yaml`).

## 18. Experiment requirements

See [EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md).
- 3 datasets × 5 outer folds × {baseline, random_search, pso}, plus 1 deployment PSO run per dataset.
- Random search budget = PSO budget = 210; each method gets a fresh cache.
- Results are paired by fold and descriptive only: mean ± sd, deltas, W/T/L, pooled metrics. **No p-values.**
- The Phase 6 timing gate falls back to 3-fold inner CV only if the projected total exceeds 2 hours, and the change
  is recorded as an ADR update. Result: a projected upper bound of 37.5 min, so 5-fold is kept (ADR-005).

## 19. Result logging requirements

See [RESULTS_SCHEMA.md](docs/RESULTS_SCHEMA.md).
- Per run: `evaluations.csv` (every evaluation) and, for PSO, `iterations.csv`; also `final.json` and
  `predictions.csv`.
- Per experiment: `summary_folds.csv`, `summary.csv`, `config.resolved.json`, `manifest.json` and `run.log`.
- The console trace prints one line per PSO iteration (the demo evidence).
- Plots are generated only from these files, and each is recorded in `plots/<exp_id>/SOURCES.json`.

## 20. Documentation requirements

- The 14 design documents are CLAUDE, IDEA, CONTEXT, ARCHITECTURE, REQUIREMENTS, MATHEMATICAL_FORMULATION,
  EXPERIMENT_PLAN, IMPLEMENTATION_PLAN, PROJECT_STRUCTURE, TESTING_STRATEGY, RESULTS_SCHEMA, DECISIONS,
  REVIEW_1_MAPPING and REVIEW_2_MAPPING. `CLAUDE.md` stays at the repo root (Claude Code loads it only from there);
  the other 13 live in `docs/`.
- A changed decision gets a superseding ADR, and every affected document is updated **in the same commit**.
- Label claims CONFIRMED, PROPOSED or TO VERIFY. When Phase 2 or Phase 6 resolves a TO VERIFY item, update
  CONTEXT §4.
- Every number in the report or slides cites its result file.

## 21. Forbidden shortcuts

- ❌ Using test data (or any statistic of it) in fitness, selection, preprocessing or stopping.
- ❌ Evaluating on the test fold more than once per method per fold, or choosing anything based on test results.
- ❌ A PSO that imports scikit-learn or touches data, or a disconnected "run PSO, then separately train" script.
- ❌ PSO or HPO libraries (pyswarms, Optuna, scikit-optimize, Hyperopt, GridSearchCV, RandomizedSearchCV).
- ❌ Fitting the imputer (or anything) on the full dataset before splitting.
- ❌ Re-drawing inner CV folds per evaluation, or sharing a cache between methods.
- ❌ Global random state, time-based seeds, or unseeded randomness.
- ❌ Reporting best validation fitness as performance, or claiming significance with 5 folds.
- ❌ Writing any result, plot or claim that is not produced from saved experiment files.
- ❌ Hard-coding dataset names or sizes in `optimization/` or `evaluation/`.
- ❌ Changing the decision variables, the bounds, the optimizer or the model without an ADR and owner approval.

## 22. Implementation phases

See [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md):
P0 docs → P1 scaffold → P2 datasets → P3 preprocessing → P4 splits and test guard → P5 baseline → P6 evaluator and
timing gate → P7 PSO core → P8 closed loop and random search → P9 convergence → P10 multi-dataset → P11 final
evaluation → P12 plots → P13 tests → P14 reproducibility audit → P15 report and PPT.

Commit at the end of each phase. Keep each phase's tests green before starting the next.

## 23. Acceptance criteria (project complete)

- [ ] All UT, IT and ET tests pass; the isolation and closed-loop proofs pass.
- [ ] `python -m pso_rf run --config configs/default.yaml` completes on a clean tree (`git_dirty = false`).
- [ ] Every dataset × fold × method has complete artifacts matching RESULTS_SCHEMA (ET-02); the summaries can be
      recomputed exactly (ET-03).
- [ ] A re-run reproduces the files exactly, excluding timing fields (IT-08).
- [ ] Figures F3–F11 are generated from files, with SOURCES.json.
- [ ] The report and a ≤ 10-slide deck answer RQ1–RQ5 truthfully (including null results) and cite every number.
- [ ] REVIEW_1_MAPPING and REVIEW_2_MAPPING are fully evidenced.
- [ ] The docs reflect the implemented system; every TO VERIFY item is resolved or explicitly deferred.
