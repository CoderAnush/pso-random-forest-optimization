# REQUIREMENTS — PSO-Based Random Forest Hyperparameter Optimization

Formal requirements specification. The word **shall** marks a mandatory requirement; **should** marks a
recommended one.

**Source column key:**
- `PDF-s<n>`: slide *n* of `ppt/CB.EN.U4ELC23005_ANUSH_RAMESH_PPT.pdf`.
- `FAC`: faculty mini-project brief.
- `BRIEF`: the Phase 0 instruction document.
- `D<n>` / `ADR-<n>`: a design decision in [DECISIONS.md](DECISIONS.md).
- `USER`: an explicit decision by the project owner (2026-09-24).

The **Verified by** column references test IDs in [TESTING_STRATEGY.md](TESTING_STRATEGY.md). The **Phase** column
references [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

---

## 1. Functional requirements (FR)

| ID | Requirement | Source | Verified by | Phase |
|---|---|---|---|---|
| FR-001 | The system shall load the supported classification datasets (`iris`, `digits`, `heart_cleveland`) through one common interface that returns features, labels, feature names, class names and metadata. | PDF-s5, BRIEF | UT-13 | P2 |
| FR-002 | The system shall generate candidate Random Forest hyperparameters using PSO. | PDF-s6, s9 | UT-02, IT-01 | P7 |
| FR-003 | The system shall evaluate each candidate using validation data only: stratified 5-fold cross-validation inside the optimization portion. | PDF-s10, USER (D1) | UT-09, IT-03 | P6 |
| FR-004 | The validation fitness shall be returned to PSO as its feedback signal. | PDF-s6, FAC | IT-01, IT-02 | P8 |
| FR-005 | PSO shall update personal bests, the global best, velocities and positions based on the returned fitness. | PDF-s9 | UT-03…UT-07 | P7 |
| FR-006 | The system shall repeat the optimization loop until the stopping condition is met. | PDF-s6, s10 | UT-08 | P7, P9 |
| FR-007 | The system shall evaluate the final optimized model on held-out test data, only after optimization has finished. | PDF-s6, s7 | IT-04…IT-07, ET-04 | P11 |
| FR-008 | The system shall train and evaluate a default Random Forest baseline under the same split and evaluation protocol. | PDF-s4 (Obj 01, 06) | IT-10 | P5 |
| FR-009 | The system shall run an equal-budget random search (open-loop comparator) using the same evaluator, folds and budget as PSO. | USER (D4) | UT-17, IT-13 | P8 |
| FR-010 | The system shall partition each dataset into 5 stratified outer folds. Each fold serves once as the held-out test set. | USER (D3) | UT-14 | P4 |
| FR-011 | The system shall convert each continuous particle position into a valid integer configuration by rounding and clipping to the bounds. | PDF-s7, D6 | UT-01 | P7 |
| FR-012 | The system shall record every evaluation (dataset, method, fold, seed, iteration, particle, position, velocity, hyperparameters, fitness, pbest, gbest). | BRIEF | UT-19, IT-12 | P8 |
| FR-013 | The system shall record a per-iteration convergence summary (gbest, mean and std of swarm fitness, diversity). | PDF-s4 (Obj 07), BRIEF | IT-12 | P9 |
| FR-014 | The system shall save per-fold final results and an experiment-level summary to disk, not only to the console. | BRIEF | ET-02, ET-03 | P10, P11 |
| FR-015 | The system shall generate the plots in [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) §9 from saved result files only. | BRIEF | ET-05 | P12 |
| FR-016 | The system shall run a complete experiment from one command-line call that takes a configuration file. | BRIEF | ET-01 | P10 |
| FR-017 | The system shall perform a deployment run (PSO on the full dataset) that outputs one recommended configuration per dataset, clearly labelled as having no test score. | D3 | ET-02 | P10 |
| FR-018 | The system shall apply dataset-specific preprocessing declared in configuration (e.g. imputation for Heart Disease). | D5 | UT-20, IT-11 | P3 |
| FR-019 | The fitness evaluator shall cache results by integer configuration within a run and mark repeated evaluations as cache hits. | ADR-016 | UT-10 | P6 |
| FR-020 | The system shall allow running a subset of datasets, methods or folds via configuration or CLI override. | BRIEF | UT-18, ET-01 | P10 |
| FR-021 | The system shall print a live, per-iteration trace (iteration, gbest configuration, gbest fitness, mean fitness) for demonstration. | AR-008 | IT-03 | P9 |

## 2. Non-functional requirements (NFR)

| ID | Requirement | Source | Verified by | Phase |
|---|---|---|---|---|
| NFR-001 | The full default experiment (3 datasets × 5 folds × 3 methods, plus deployment runs) should complete within about 1 hour on the development machine (20 cores). Measured worst-case evaluation timings predict about 40 min. **TO VERIFY at P6.** | D12 | ET-06 | P6, P14 |
| NFR-002 | The system shall run on CPU only, with Python 3.10 on Windows 11. It should also run on Linux. | environment | ET-01 | P1 |
| NFR-003 | The optimization package shall not import model, dataset or scikit-learn code. The evaluation package shall not import optimization code. | BRIEF (principle 4) | UT-22 | P1, P7 |
| NFR-004 | All public functions shall have type hints and a docstring. Configuration and result records shall be typed dataclasses. | BRIEF | code review | all |
| NFR-005 | Experiments shall not require network access. All datasets are available locally (bundled with scikit-learn or committed under `data/raw/`). | D5 | UT-13 | P2 |
| NFR-006 | A failure while evaluating one candidate shall not crash the run. It is logged and scored as the worst possible fitness. | BRIEF (error handling) | UT-12 | P6 |
| NFR-007 | Parallelism shall not change results. Parallel CV folds must give results identical to serial execution. | D12 | IT-08 | P6 |

## 3. Academic requirements (AR)

| ID | Requirement | Source | Verified by | Phase |
|---|---|---|---|---|
| AR-001 | The project shall use exactly one optimization technique as its optimizer: **Particle Swarm Optimization**. Random search appears only as a non-feedback comparator. | FAC | review | P7 |
| AR-002 | The optimizer shall maximize a clearly defined objective: validation fitness. | FAC | review | P0 |
| AR-003 | The optimizer's output shall change a system parameter: the Random Forest hyperparameters. | FAC | IT-01 | P8 |
| AR-004 | The system shall contain closed-loop feedback: optimizer output → parameter change → system response → measured feedback → optimizer. | FAC | IT-01, IT-02 | P8 |
| AR-005 | The optimization program shall be a distinct block of the system, in code (`pso_rf.optimization`) and in diagrams. | FAC | UT-22, review | P1, P7 |
| AR-006 | The Review 1 material shall contain: block diagram, methods, objective function, constraints, decision variables, program snippet, optimization technique, and why the technique was selected. | FAC | [REVIEW_1_MAPPING.md](REVIEW_1_MAPPING.md) | P0, P15 |
| AR-007 | The presentation shall not exceed 10 slides. | FAC | review | P15 |
| AR-008 | Review 2 shall demonstrate the full system as a closed loop in which the output is fed back into the system. | FAC | [REVIEW_2_MAPPING.md](REVIEW_2_MAPPING.md) | P15 |
| AR-009 | The documentation shall state what the system is, where optimization is applied, which method is used, why, and how it works for this system. | FAC | [IDEA.md](IDEA.md) | P0 |

## 4. Machine-learning requirements (MLR)

| ID | Requirement | Source | Verified by | Phase |
|---|---|---|---|---|
| MLR-001 | The model shall be scikit-learn's `RandomForestClassifier`. | PDF-s6 | UT-11 | P5 |
| MLR-002 | Only `n_estimators`, `max_depth` and `min_samples_split` shall be tuned. All other RF parameters stay at scikit-learn defaults, except `random_state` (run seed) and `n_jobs` (1). | PDF-s2, s7, D11 | UT-11 | P5 |
| MLR-003 | Every Random Forest shall receive `random_state` equal to the run seed. | ADR-018 | UT-11, IT-08 | P5 |
| MLR-004 | Any preprocessing that learns from data (e.g. imputation) shall be inside the model pipeline, so it is fitted only on the training part of each fit. | BRIEF (principle 9), ADR-010 | UT-20, IT-11 | P3 |
| MLR-005 | The final model of each method shall be refit on the full optimization portion of the fold, with that method's chosen configuration, before test evaluation. | PDF-s6 | IT-10 | P11 |
| MLR-006 | Test evaluation shall report accuracy, balanced accuracy, macro precision, macro recall, macro F1 and the confusion matrix. For Heart Disease it also reports precision, recall and F1 for the positive (disease) class. | PDF-s6, s10, D2 | UT-16 | P11 |
| MLR-007 | The baseline shall be `RandomForestClassifier()` with scikit-learn defaults (`n_estimators=100`, `max_depth=None`, `min_samples_split=2`), `random_state` = run seed, inside the same pipeline. | PDF-s4, D4 | UT-11, IT-10 | P5 |

## 5. Optimization requirements (OR)

| ID | Requirement | Source | Verified by | Phase |
|---|---|---|---|---|
| OR-001 | The decision vector shall be $x = [n\_estimators, max\_depth, min\_samples\_split]$. | PDF-s6, s7 | UT-01 | P7 |
| OR-002 | Bounds: $50 \le n\_estimators \le 200$, $2 \le max\_depth \le 20$, $2 \le min\_samples\_split \le 10$. | PDF-s6, s7, s8 | UT-01, UT-05 | P7 |
| OR-003 | Every evaluated configuration shall consist of integers within the bounds. | PDF-s6, s7 | UT-01 | P7 |
| OR-004 | The optimization shall **maximize** $F(x)$. | PDF-s7 | UT-07 | P7 |
| OR-005 | $F(x)$ shall be the mean stratified 5-fold CV score on the optimization portion. The metric is `accuracy` by default and `balanced_accuracy` when the class-ratio gate triggers or when configured. | PDF-s10, USER (D1, D2) | UT-09, UT-21 | P6 |
| OR-006 | Velocity update: $v \leftarrow w v + c_1 r_1 (p - x) + c_2 r_2 (g - x)$, with $r_1, r_2 \sim U[0,1)$ drawn per particle and per dimension. Position update: $x \leftarrow x + v$. | PDF-s9 | UT-03 | P7 |
| OR-007 | Default coefficients: $w = 0.7298$, $c_1 = c_2 = 1.49618$. | D8 | UT-18 | P7 |
| OR-008 | Default swarm size: 10 particles. | PDF-s10 | UT-02 | P7 |
| OR-009 | Default maximum iterations: 20 update iterations after the initial evaluation, giving 10 × 21 = 210 evaluations per run. | D9 | UT-08 | P7 |
| OR-010 | Initial positions shall be uniform in the bounds. Initial velocities shall be uniform in $\pm 0.1 \times$ range per dimension. | D7 | UT-02 | P7 |
| OR-011 | Velocities shall be clamped to $\pm v_{max,d}$, where $v_{max,d} = 0.2 \times (ub_d - lb_d)$. | D7 | UT-04 | P7 |
| OR-012 | Boundary handling shall be an absorbing wall. A position component that leaves the bounds is clipped to the bound, and that velocity component is set to 0. | D7 | UT-05 | P7 |
| OR-013 | Updates shall be synchronous: all particles are evaluated, then pbest and gbest are updated, then all particles move. pbest and gbest change only on strict improvement (`>`). Same-iteration ties for gbest go to the lowest particle index. | D9, ADR-017 | UT-06, UT-07 | P7 |
| OR-014 | The swarm state (positions, velocities, pbest positions) shall stay continuous. Rounding happens only when decoding for evaluation. | D6 | UT-01, UT-03 | P7 |
| OR-015 | Default stopping: at `max_iter`. An optional patience rule (stop when gbest improves by less than `tol` over `P` iterations) is **disabled by default**. The last iteration with a gbest improvement is always recorded. | PDF-s10, D10 | UT-08 | P9 |
| OR-016 | PSO shall be implemented from scratch with NumPy. No PSO or HPO library (e.g. pyswarms, Optuna, scikit-optimize) shall be used. | PDF-s4 (Obj 02) | review | P7 |
| OR-017 | All randomness in an optimizer shall come from one `numpy.random.Generator` seeded from configuration. | ADR-018 | UT-02, IT-08 | P7 |
| OR-018 | Random search shall sample configurations uniformly and independently from the integer search space, with a budget equal to PSO's evaluation budget (default 210). The samples shall not depend on fitness values. | D4 | UT-17 | P8 |
| OR-019 | The optimizer shall interact with the system only through an objective callable that takes a configuration and returns a fitness. It shall have no access to data. | BRIEF (principle 4), ADR-024 | UT-22, IT-05 | P7 |

## 6. Data requirements (DR)

| ID | Requirement | Source | Verified by | Phase |
|---|---|---|---|---|
| DR-001 | Iris and Digits shall be loaded from scikit-learn's bundled copies. Heart Disease shall be the UCI **Cleveland** subset (`processed.cleveland.data`). | PDF-s5, s12, USER (D5) | UT-13 | P2 |
| DR-002 | The Cleveland raw file shall be committed under `data/raw/` together with its SHA-256 checksum and citation. The loader shall verify the checksum. | D5 | UT-13 | P2 |
| DR-003 | The Heart Disease target shall be binarized: `num > 0 → 1` (disease), `num = 0 → 0`. | D5 | UT-13 | P2 |
| DR-004 | Missing values (`?`) shall be read as NaN and imputed inside the pipeline with the most-frequent strategy. Rows shall not be dropped. | D5, ADR-010 | UT-20, IT-11 | P3 |
| DR-005 | Every dataset shall be audited for exact duplicate (X, y) rows. The audit is reported. Duplicates are removed before splitting only if they exceed 1% of rows. | D5 | UT-13 | P2 |
| DR-006 | Every dataset shall be audited for class balance. If the largest-to-smallest class ratio exceeds 1.5, that dataset's fitness metric becomes `balanced_accuracy`. | D2 | UT-21 | P2, P6 |
| DR-007 | Outer folds shall be produced by `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. | D3 | UT-14 | P4 |
| DR-008 | The held-out test fold shall never be accessible while the optimization phase of its run is active. | PDF-s6, s7, BRIEF | UT-15, IT-04…IT-07 | P4 |
| DR-009 | Inner CV folds shall be produced by `StratifiedKFold(n_splits=5, shuffle=True, random_state=run_seed)` on the optimization portion. They stay fixed for every evaluation within a run. | D1 | UT-09 | P6 |
| DR-010 | No preprocessing or statistic shall be computed on the full dataset before splitting. The only exceptions are the audits in DR-005 and DR-006, which fit no model parameters. | BRIEF (principle 9) | IT-11 | P3 |

## 7. Experiment requirements (ER)

| ID | Requirement | Source | Verified by | Phase |
|---|---|---|---|---|
| ER-001 | For each dataset and each of the 5 outer folds, the system shall run three methods: `baseline`, `random_search` and `pso`. | USER (D3, D4) | ET-02 | P10 |
| ER-002 | For each dataset, the system shall perform one deployment PSO run on the full dataset. | D3 | ET-02 | P10 |
| ER-003 | Comparisons between methods shall be paired by outer fold. | D3 | ET-03 | P11 |
| ER-004 | Results shall be reported as mean ± standard deviation across folds, win/tie/loss counts, and pooled confusion matrices. **No p-values or significance claims** shall be made with 5 pairs. | D3, ADR-020 | review | P11, P15 |
| ER-005 | Convergence shall be analysed from recorded per-iteration data (gbest curve, mean fitness, diversity). | PDF-s4 (Obj 07) | IT-12 | P9, P12 |
| ER-006 | An anytime curve (best-so-far validation fitness vs. evaluation count, PSO vs. random search) shall be produced. | D4 | ET-05 | P12 |
| ER-007 | No result value (accuracy, fitness, configuration, convergence claim) shall be written anywhere unless it was read from a saved result file. | BRIEF (principle 8) | ET-03, ET-05 | all |
| ER-008 | Reports shall state the number of configurations tied with the best validation fitness, and any best configurations lying on a search-space boundary. | D2, D11 | review | P11 |
| ER-009 | Reports shall distinguish evaluations (proposed configurations) from unique RF fits (cache misses). | ADR-016 | IT-12 | P11 |
| ER-010 | The best validation fitness shall never be presented as expected generalization performance. Only test-fold metrics are performance. | ADR-020 | review | P11, P15 |

## 8. Reproducibility requirements (RR)

| ID | Requirement | Source | Verified by | Phase |
|---|---|---|---|---|
| RR-001 | Every seed shall be set in configuration: the outer split seed (42), the run seeds (0–4, one per outer fold) and the deployment seed (5). | PDF-s4 (Obj 08), BRIEF | UT-18 | P1 |
| RR-002 | In run *k*, the run seed shall drive the PSO generator, the random search generator, the inner CV shuffle and RF `random_state`. | ADR-018 | IT-08 | P8 |
| RR-003 | The fully resolved configuration shall be saved with every experiment. | BRIEF | ET-02 | P10 |
| RR-004 | A manifest shall record the git commit, dirty flag, Python and library versions, platform, CPU count, dataset checksums and config hash. | BRIEF | ET-02 | P10 |
| RR-005 | Re-running the same configuration shall give identical `evaluations.csv`, `iterations.csv` and `final.json` contents, excluding timestamps and timing fields. | BRIEF | IT-08 | P14 |
| RR-006 | Dependencies shall be pinned in `requirements.txt`. | BRIEF | review | P1 |
| RR-007 | No code shall use global random state (`numpy.random.seed`, the `random` module) or time-based seeds. | ADR-018 | review, IT-08 | all |

## 9. Documentation requirements (DOC)

| ID | Requirement | Source | Verified by | Phase |
|---|---|---|---|---|
| DOC-001 | The 14 design documents listed in [CLAUDE.md](../CLAUDE.md) shall exist before implementation begins. | BRIEF | review | P0 |
| DOC-002 | Any change to a decision shall be recorded as a new or superseding ADR in [DECISIONS.md](DECISIONS.md), and the affected documents updated in the same commit. | BRIEF | review | all |
| DOC-003 | Documents shall label statements as CONFIRMED, PROPOSED or TO VERIFY where a claim depends on unverified facts. | BRIEF | review | P0 |
| DOC-004 | `README.md` shall explain installation, how to run an experiment, where results are stored, and link the design documents. | BRIEF | review | P1, P15 |
| DOC-005 | Reports and slides shall cite the result file that each number comes from. | ER-007 | ET-05 | P15 |

---

## Traceability summary

Every FR, OR and DR requirement above names at least one verifying test and one implementation phase. When a
requirement changes, update the requirement row, the referenced test in [TESTING_STRATEGY.md](TESTING_STRATEGY.md),
and the phase in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) together.
