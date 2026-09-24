# IMPLEMENTATION PLAN

Sixteen phases, from documentation to final report. Each phase lists its objective, files, dependencies, inputs,
outputs, tests (IDs from [TESTING_STRATEGY.md](TESTING_STRATEGY.md)) and completion criteria. Tests are written
**within** each phase, test-first where practical. Phase 13 completes and audits the whole suite.

**Global rules for every phase:**
- Keep the closed loop intact.
- Never let test data into the optimization phase.
- Invent no numbers.
- Update the docs when a decision changes (DOC-002).
- Commit at the end of each phase with a conventional message.

```
P0 docs ─▶ P1 scaffold ─▶ P2 datasets ─▶ P3 preprocessing ─▶ P4 splits ─▶ P5 baseline ─▶ P6 evaluator ─┐
                                                                                                        │
                                             P7 PSO core (independent of P2–P6, needs only P1) ─────────┤
                                                                                                        ▼
               P15 report ◀─ P14 repro audit ◀─ P13 tests ◀─ P12 plots ◀─ P11 final eval ◀─ P10 multi-dataset ◀─ P9 convergence ◀─ P8 closed loop
```

---

### Phase 0: Documentation and architecture
- **Objective:** a complete, agreed design before any code.
- **Files:** the 14 `*.md` design documents (`CLAUDE.md` at the repo root, the other 13 in `docs/`), `README.md`
  (stub), `.gitignore`.
- **Dependencies:** the PDF and the owner's decisions.
- **Inputs:** the PDF (all 13 slides), the faculty brief, the Phase 0 instructions, measured environment facts.
- **Outputs:** committed documents; the repository initialized and pushed.
- **Tests:** consistency checks (constants identical across documents; every FR/OR/DR has a test and a phase); a grep
  confirms no invented results.
- **Done when:** the owner approves; the READY checklist in the final Phase 0 summary is complete except the
  explicitly deferred TO VERIFY items.

### Phase 1: Project scaffolding
- **Objective:** an installable package skeleton with tooling and configuration loading.
- **Files:** `pyproject.toml`, `requirements.txt` (pinned: numpy 1.26.4, scikit-learn 1.7.2, pandas 2.3.3,
  matplotlib 3.10.6, PyYAML 6.0.1, joblib 1.5.2, pytest 9.1.1), the `src/pso_rf/**/__init__.py` files,
  `utils/{hashing,io,log}.py`, `experiments/config.py`, `configs/default.yaml`, `configs/demo.yaml`,
  `configs/test.yaml`, `tests/conftest.py`.
- **Dependencies:** P0.
- **Inputs:** the config template (ARCHITECTURE §H).
- **Outputs:** `pip install -e .` works; `pytest` runs (empty suite passes); config loads, merges and validates.
- **Tests:** UT-18 (config loading, merging, validation), UT-22 (layering rule, active from the start).
- **Done when:** a fresh clone installs, the tests pass, and the resolved default config round-trips to JSON.

### Phase 2: Dataset abstraction
- **Objective:** load all three datasets through one interface; resolve every dataset TO VERIFY item.
- **Files:** `datasets/{base,iris,digits,heart_cleveland,audit}.py`, the registry,
  `scripts/download_cleveland.py`, `data/raw/processed.cleveland.data`, `data/raw/MANIFEST.json`,
  `data/DATASET_AUDIT.md`.
- **Dependencies:** P1.
- **Inputs:** scikit-learn bundled data; the UCI Cleveland file (downloaded **once**; URL and SHA-256 recorded).
- **Outputs:** a `DatasetBundle` per dataset; audit output (shapes, class counts, ratio, missing, duplicates, gate
  outcome).
- **Tests:** UT-13 (shapes and classes match the audit; Heart target binarization; `?` → NaN; checksum mismatch
  raises), UT-21 (class-ratio gate).
- **Done when:** the audit is written. CONTEXT §4 items V1–V4 and V7 are updated from TO VERIFY to measured, and the
  EXPERIMENT_PLAN §2 table is filled.

### Phase 3: Preprocessing
- **Objective:** leakage-safe preprocessing, declared per dataset.
- **Files:** `preprocessing/pipeline.py`.
- **Dependencies:** P2.
- **Inputs:** `datasets.<name>.preprocessing` config.
- **Outputs:** a `PreprocessingSpec` → list of scikit-learn steps (Heart: most-frequent imputer; others: empty).
- **Tests:** UT-20 (the imputer is fitted only on the rows passed to `fit`; a no-op for complete datasets).
- **Done when:** a pipeline built for Heart fits and predicts with NaNs present, and the imputer statistics provably
  come from training rows only.

### Phase 4: Train/validation/test split
- **Objective:** outer folds and the test-set guard.
- **Files:** `evaluation/splits.py` (`outer_folds`, `OptimizationData`, `HeldOutTestSet`, `OptimizationPhase`,
  `TestSetAccessError`).
- **Dependencies:** P2.
- **Inputs:** `DatasetBundle`, `split.outer_folds`, `split.outer_seed`.
- **Outputs:** 5 `FoldData` per dataset.
- **Tests:** UT-14 (disjoint, complete, stratified, deterministic), UT-15 (`reveal()` raises inside
  `OptimizationPhase` and works outside), IT-04 (no index overlap on the real datasets).
- **Done when:** the tests pass and the test arrays are unreachable except through `reveal()`.

### Phase 5: Baseline Random Forest
- **Objective:** the RF pipeline builder and the reproducible baseline.
- **Files:** `models/random_forest.py`, `models/baseline.py`.
- **Dependencies:** P3.
- **Inputs:** configuration dict, seed, `PreprocessingSpec`.
- **Outputs:** `build_model(config, seed, preprocessing) -> Pipeline`; `BASELINE_CONFIG = {n_estimators: 100,
  max_depth: None, min_samples_split: 2}`.
- **Tests:** UT-11 (exact parameter passing; all other parameters default; `random_state` set; `n_jobs=1`), IT-10
  (baseline scored under the protocol: CV on the optimization portion, refit, then test; the refit and test steps
  depend on P11 and are completed there).
- **Done when:** the baseline builds and scores deterministically.

### Phase 6: Random Forest evaluator (fitness)
- **Objective:** the fitness function, i.e. the measurement block of the loop.
- **Files:** `evaluation/fitness.py` (`FitnessEvaluator`, `FitnessResult`), `scripts/benchmark_eval.py` (timing
  gate). The `FitnessResult` → `Evaluation` adapter is not here: it is `make_objective` in `experiments/runner.py`
  (Phase 8), because `evaluation/` must not import `optimization/` (UT-22).
- **Dependencies:** P4, P5.
- **Inputs:** `OptimizationData`, `cv_folds = 5`, seed, metric, preprocessing, `n_jobs_folds = 5`.
- **Outputs:** fitness with per-fold scores, diagnostics, cache flag, timing and status.
- **Tests:**
  - UT-09 (equals a manual `cross_val_score` on identical folds);
  - UT-10 (cache hit: no refit, identical result, flag set);
  - UT-12 (an exception gives −inf and `status = failed`; NaN is handled);
  - IT-08 partial (parallel and serial scores identical);
  - ET-06 (**timing gate**: measure $t_{eval}$ at the worst-case configuration and a mid-range one for all three
    datasets, and record the projected total runtime).
- **Done when:** the tests pass. The timing gate outcome is recorded in DECISIONS.md ADR-005: keep 5-fold if the
  projected total is ≤ 2 h, otherwise fall back to 3-fold and record why. CONTEXT §4 V5 and V6 are updated.

### Phase 7: PSO core
- **Objective:** a generic, from-scratch PSO with no ML knowledge.
- **Files:** `optimization/{search_space,events,pso}.py`.
- **Dependencies:** P1 only (it can proceed in parallel with P2–P6).
- **Inputs:** `SearchSpace`, an `Objective` callable, `PSOConfig`, `np.random.Generator`.
- **Outputs:** `OptimizationResult`, plus evaluation and iteration events sent to callbacks.
- **Tests:**
  - UT-01 (decode, rounding and clipping);
  - UT-02 (initialization bounds and seed determinism);
  - UT-03 (velocity update equals the hand-computed worked example in MATHEMATICAL_FORMULATION §9);
  - UT-04 (clamp);
  - UT-05 (absorbing wall);
  - UT-06 (pbest strict improvement);
  - UT-07 (gbest is monotone and ties go to the lowest index);
  - UT-08 (exactly N(T+1) evaluations; the patience rule when enabled);
  - IT-01 (converges on a stub objective with a known interior optimum);
  - IT-02 (feedback ablation);
  - UT-22 (no forbidden imports).
- **Done when:** all the tests pass using stub objectives only, with no scikit-learn involved.

### Phase 8: Closed-loop integration (and the open-loop comparator)
- **Objective:** wire PSO ⇄ evaluator into the closed loop; add random search, the recorder and the seeding.
- **Files:** `optimization/random_search.py`, `experiments/{seeding,recorder,runner}.py` (`run_fold`, and
  `make_objective`, the `FitnessResult` → `Evaluation` adapter).
- **Dependencies:** P6, P7.
- **Inputs:** one dataset, one fold, one method.
- **Outputs:** `evaluations.csv` for a real closed-loop run; `OptimizationResult`.
- **Tests:**
  - IT-03 (end-to-end closed loop on Iris with a tiny budget; artifacts conform to the schema);
  - IT-05 (spy: no test row reaches `fit`/`predict` during optimization);
  - IT-06 and IT-07 (test-label and test-feature invariance);
  - IT-13 (random search pipeline);
  - UT-17 (random search is uniform, respects its budget, and is independent of fitness);
  - UT-19 (recorder columns and types).
- **Done when:** one `run_fold("iris", 0, "pso")` produces a complete evaluation log, and all the isolation tests
  pass.

### Phase 9: Convergence tracking
- **Objective:** per-iteration summaries, the live trace, and the optional patience rule.
- **Files:** `optimization/pso.py` (iteration summary content), `experiments/recorder.py` (`iterations.csv`),
  `utils/log.py` (the console trace).
- **Dependencies:** P8.
- **Inputs:** PSO state per iteration.
- **Outputs:** `iterations.csv` (gbest, mean, std, min, max, diversity, ties, cache hits, cumulative counts); the
  console trace line per iteration.
- **Tests:** IT-12 (the gbest column equals the running maximum of evaluation fitness; row counts; the convergence
  iteration is correct), UT-08 (patience variant).
- **Done when:** the convergence data for a run can reproduce the gbest curve exactly.

### Phase 10: Multi-dataset experiment
- **Objective:** full orchestration across datasets × folds × methods, plus the deployment run and the CLI.
- **Files:** `experiments/runner.py` (`run_experiment`), `experiments/cli.py`, `__main__.py`, `utils/manifest.py`,
  `experiments/README.md`.
- **Dependencies:** P9 (plus P11 for the complete per-fold output; the two are developed together).
- **Inputs:** `configs/default.yaml` and `configs/demo.yaml`.
- **Outputs:** the `results/<exp_id>/` tree, `config.resolved.json`, `manifest.json`, `run.log`.
- **Tests:** ET-01 (demo experiment on all 3 datasets with a tiny budget), ET-02 (audit of completeness: every
  dataset × fold × method present, 210 evaluations for PSO and random search, deployment present, manifest checksums
  match).
- **Done when:** `python -m pso_rf run --config configs/demo.yaml` completes and passes ET-01 and ET-02.

### Phase 11: Final test evaluation
- **Objective:** refit and test once per method per fold; compute metrics; build the summaries.
- **Files:** `evaluation/{final,metrics}.py`, `experiments/summary.py`.
- **Dependencies:** P8 (runs in the same loop as P10).
- **Inputs:** the best configuration, `OptimizationData`, `HeldOutTestSet` (outside `OptimizationPhase`).
- **Outputs:** `final.json`, `predictions.csv`, `summary_folds.csv`, `summary.csv`.
- **Tests:** UT-16 (metric computation on toy predictions), IT-10 (baseline protocol complete), ET-03 (the summaries
  are exactly recomputable from the per-fold files), ET-04 (each test evaluation timestamp is later than the
  optimization end).
- **Done when:** the demo experiment produces summaries that pass ET-03 and ET-04.

### Phase 12: Visualization
- **Objective:** figures F3–F11 from the result files.
- **Files:** `visualization/{convergence,comparison,trajectories}.py`; the `plot` CLI command.
- **Dependencies:** P10, P11.
- **Inputs:** `results/<exp_id>/`.
- **Outputs:** `plots/<exp_id>/*.png`, `plots/<exp_id>/SOURCES.json`.
- **Tests:** ET-05 (each figure is produced from files only, and SOURCES.json lists existing files and the config
  hash).
- **Done when:** all planned figures render from the demo experiment's results.

### Phase 13: Testing
- **Objective:** complete, audit and harden the test suite.
- **Files:** `tests/**`, `configs/test.yaml`.
- **Dependencies:** P1–P12.
- **Inputs:** the test catalogue in TESTING_STRATEGY.md.
- **Outputs:** every UT, IT and ET test implemented; a coverage report.
- **Tests:** all of them. The coverage target is at least 90% line coverage for `optimization/` and `evaluation/`.
- **Done when:** the full suite passes locally in under 5 minutes (the experiment tests use `configs/test.yaml`),
  and every requirement in REQUIREMENTS.md maps to a passing test or a documented review step.

### Phase 14: Reproducibility and experiment audit
- **Objective:** run the real experiment and prove it is reproducible.
- **Files:** `experiments/README.md` (run record), `results/<exp_id>/` (final experiment).
- **Dependencies:** P13.
- **Inputs:** `configs/default.yaml` on a clean git tree.
- **Outputs:** the final experiment results; a re-run comparison.
- **Tests:** IT-08 at full scale for at least one dataset (re-run, compare files excluding timing fields), ET-02
  (completeness), ET-06 (actual runtime recorded vs. the prediction).
- **Done when:** the manifest shows `git_dirty = false`, the re-run matches, and NFR-001 is confirmed or explained.

### Phase 15: Final results, report and PPT
- **Objective:** communicate the real results truthfully for Review 2.
- **Files:** `report/`, the updated `README.md`, the slide deck (≤10 slides).
- **Dependencies:** P14.
- **Inputs:** `results/<exp_id>/`, `plots/<exp_id>/`, REVIEW_1_MAPPING.md, REVIEW_2_MAPPING.md.
- **Outputs:** tables T1–T6, figures, the answers to RQ1–RQ5 (including null results if that is what the data
  shows), and the updated slides with the PPT edits from REVIEW_1_MAPPING §3.
- **Tests:** a DOC-005 check (every number in the report and slides cites its file); an ER-010 check (no validation
  fitness is presented as performance).
- **Done when:** the Review 2 checklist in REVIEW_2_MAPPING.md is fully evidenced.

---

## Critical path and effort (rough guide)

| Phase | Relative effort | Notes |
|---|---|---|
| P1–P5 | small | mostly thin wrappers around scikit-learn |
| P6 | medium | cache, parallel folds, failure handling, timing gate |
| P7 | medium | the core of the course project; test-heavy |
| P8 | medium–large | loop wiring plus the isolation proofs |
| P9–P11 | medium | recording and summaries |
| P12 | medium | plots |
| P13–P14 | medium | audit, full run (≈40 min estimated) |
| P15 | medium | writing and slides |
