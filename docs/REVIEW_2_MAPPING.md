# REVIEW 2 MAPPING

Review 2 requires the whole system to be shown as a **closed-loop system in which the output is fed back**. This
document maps each link of that chain to the implementation and to the concrete evidence that will prove it. All
evidence is produced by the real experiment (Phase 14). **Nothing here is a result.**

## 1. The chain, link by link

| # | Link | What happens in the system | Implementation | Evidence shown at Review 2 |
|---|---|---|---|---|
| 1 | **Input** | A dataset is loaded, audited and split into 5 outer folds; fold *k*'s test part is sealed | `datasets/*`, `evaluation/splits.py` | `data/DATASET_AUDIT.md`; `audit.json`; UT-14 (folds disjoint and complete) |
| 2 | **Optimization** | PSO holds 10 particles (candidate configurations) with velocities and memories | `optimization/pso.py` | the live trace; `evaluations.csv` `pos_*`/`vel_*` columns |
| 3 | **Parameter modification** | Each particle position is decoded to integer hyperparameters and passed to the RF constructor | `SearchSpace.decode` → `models/random_forest.py` | `evaluations.csv` columns `n_estimators`, `max_depth`, `min_samples_split`; F6 (hyperparameter trajectories) |
| 4 | **ML system** | A Random Forest is trained with those hyperparameters on 4/5 of the optimization portion, five times | `evaluation/fitness.py` + pipeline | `fit_time_s`, `cache_hit`; UT-11 (parameters passed exactly) |
| 5 | **Performance evaluation** | Each trained forest is scored on its inner validation fold; the mean is the fitness | `cross_validate` on fixed inner folds | `cv_scores`, `fitness` columns; UT-09 |
| 6 | **Feedback** | The fitness value is returned to PSO, the only information PSO receives | `Objective` → `Evaluation.fitness` | UT-22 (the optimizer has no data access); IT-02 (shuffled or constant feedback changes the trajectory) |
| 7 | **Optimization update** | PSO updates pbest and gbest, then velocities and positions (clamp, absorbing wall) | `PSOOptimizer` update step | `pbest_fitness`, `gbest_fitness` columns; `iterations.csv`; F7 (particle fitness progression); F8 (gbest evolution) |
| 8 | **Loop repeats** | The new positions become new hyperparameters, and links 3–7 repeat for 20 iterations | `PSOOptimizer.run` | F3 (convergence curve); `iterations.csv` (21 rows); the live trace |
| 9 | **Final optimized output** | The best configuration is frozen; the RF is refit on the whole optimization portion; the sealed test fold is opened and scored **once** | `evaluation/final.py` | `final.json` (`optimization_finished_at` < `test_evaluated_at`, ET-04); `predictions.csv` |
| 10 | **Comparison** | The same protocol is applied to the default RF and to open-loop random search | runner | T2 main results; F4 (anytime curve: feedback vs. no feedback); F5, F9 |

## 2. What proves the loop is closed (not just claimed)

| Claim | Proof | Where |
|---|---|---|
| The optimizer's output changes the system | Every evaluated configuration in `evaluations.csv` is the decode of that particle's position at that iteration | IT-03, UT-01 |
| The system's response is measured | Every configuration has 5 CV scores and a fitness | IT-03, UT-09 |
| The measurement is fed back | PSO's pbest/gbest columns change exactly when fitness improves | UT-06, UT-07, IT-12 |
| The next output depends on the feedback | Constant or shuffled feedback produces a different trajectory (feedback ablation) | IT-02 |
| Feedback improves the search | PSO vs. open-loop random search at equal budget (anytime curve and test metrics). This is a research question, and the answer may be a tie. | F4, T2 |
| The output leaves the loop only one way | The test fold is sealed during the loop; permuting test labels changes nothing in the loop | UT-15, IT-06, IT-07 |

## 3. Experimental evidence package (what will be shown)

1. **System diagrams:** F1 (architecture) and F2 (closed-loop diagram with the separated test lane), from
   ARCHITECTURE §A and §C.
2. **One loop, visibly:** for Iris fold 0, the console trace (one line per iteration: gbest configuration, gbest
   fitness, mean fitness, unique fits) plus F6, F7 and F8 for that run.
3. **Convergence:** F3 per dataset (gbest vs. iteration, 5 folds plus the mean), with the convergence iterations
   (T4) and F11 (diversity).
4. **Value of feedback:** F4, the anytime curve of PSO vs. random search, with the mean and fold band per dataset.
5. **Final performance:** T2 (test accuracy and macro F1, mean ± sd; paired deltas vs. the baseline and random
   search; W/T/L), plus F5, F9 and F10 (pooled confusion matrices).
6. **Chosen configurations:** T3 (per fold) and T5 (deployment recommendation next to the outer-CV estimate), with
   boundary hits and tie counts noted.
7. **Honesty statements:** validation fitness labelled as biased, no p-values, and small-test-set caveats (Iris: 1
   sample = 3.3 points). Null results are stated if they occur.
8. **Reproducibility:** `manifest.json` (git SHA, versions, checksums), the re-run comparison from Phase 14, and the
   test-suite summary (including the isolation and closed-loop tests).

Every number on a slide cites its source file (DOC-005).

## 4. Live demonstration outline (≈5 minutes, interactive demo)

Launch before the review with `python -m pso_rf demo` (Streamlit, opens in the browser).

1. **How it works** (1 min): the control-system mapping and the PSO equations. Press **Try to peek at the test
   fold**: the request is blocked with `TestSetAccessError`. Press **Run the feedback ablation**: only true
   feedback steers the swarm to the optimum (Proof 1 and Proof 2).
2. **Live closed-loop lab** (2 min): Heart Disease, N = 10, T = 8, race on, manual pick on. Press **Run**.
   - Every evaluation lights the forward path (PSO → hyperparameters → RF ×5 inner folds → fitness), and every
     iteration lights the feedback edge.
   - The test-fold card stays **SEALED**. Convergence and the 3-D swarm update live.
   - The random-search race runs with its feedback edge drawn as **cut**.
   - At the end the card turns **UNSEALED · scored once**, and the table compares the default RF, the manual
     pick, random search and PSO.
3. **Experiment results** (1.5 min): the verified badge and clean-tree provenance, then held-out accuracy over
   5 folds, the paired deltas, the anytime curves (value of feedback), and the chosen configurations and
   deployment recommendation per dataset.
4. **Swarm replay** (0.5 min): press **Play** on a saved Digits fold to watch the swarm contract onto gbest.

The live lab runs the experiment's own `run_fold`, so what is shown is exactly what was measured (ADR-026).

## 5. Review 2 readiness checklist

| Item | Evidence file | Status |
|---|---|---|
| Closed-loop diagram with feedback and separated test lane | ARCHITECTURE §A/§C → F1, F2 | designed (P0) |
| Loop implemented and wired | `experiments/runner.py`; IT-01, IT-03 | pending (P8) |
| Feedback dependence proven | IT-02 | pending (P7) |
| Test isolation proven | UT-15, IT-04…IT-07, ET-04 | pending (P4, P8, P11) |
| Convergence data and plots | `iterations.csv`, F3, F7, F8, F11 | pending (P9, P12) |
| Baseline and random-search comparison | `summary.csv`, T2, F4, F5, F9 | pending (P11, P12) |
| Multi-dataset results | `summary.csv` across 3 datasets | pending (P14) |
| Reproducibility demonstrated | manifest and re-run comparison | pending (P14) |
| Slides ≤ 10, numbers cited | report / deck | pending (P15) |
