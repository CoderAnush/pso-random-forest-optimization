# DECISIONS — Architecture Decision Records

Each record gives the decision, its rationale, the alternatives considered, the consequences, and its impact on the
original PPT.

**Status values:**
- **Accepted (PDF):** fixed by the proposal.
- **Accepted (owner):** chosen by the project owner after the Phase 0 design review on 2026-09-24.
- **Accepted (default):** a Phase 0 design default.
- **Superseded:** replaced by a later ADR.

Changes to any decision are made by adding a superseding ADR and updating the affected documents in the same commit
(DOC-002).

| ADR | Topic | Status | Changes the PPT? |
|---|---|---|---|
| 001 | PSO as the optimizer | Accepted (PDF) | no |
| 002 | Random Forest as the system | Accepted (PDF) | no |
| 003 | The three hyperparameters and their bounds | Accepted (PDF) | no |
| 004 | Validation fitness, never test | Accepted (PDF) | no |
| 005 | Inner 5-fold stratified CV, folds fixed per run | Accepted (owner) | **yes** (slide 10: 3-fold → 5-fold) |
| 006 | Accuracy as fitness (configurable) | Accepted (owner) | no |
| 007 | Outer 5-fold nested CV plus a deployment run | Accepted (owner) | **yes** (slides 6, 10) |
| 008 | Comparators: default RF plus equal-budget random search | Accepted (owner) | **yes** (slides 4, 11) |
| 009 | Heart Disease = UCI Cleveland, vendored | Accepted (owner) | no (clarifies) |
| 010 | Preprocessing inside the pipeline | Accepted (default) | no |
| 011 | Integer handling: continuous state plus decode | Accepted (default) | no (fills a gap) |
| 012 | Boundary handling and velocity limits | Accepted (default) | no (fills a gap) |
| 013 | PSO coefficients | Accepted (default) | no (fills a gap) |
| 014 | Swarm size, iterations, topology, synchronous update | Accepted (PDF + default) | no |
| 015 | Stopping and the definition of convergence | Accepted (default) | no |
| 016 | Fitness caching and duplicate candidates | Accepted (default) | no |
| 017 | Tie-breaking | Accepted (default) | no |
| 018 | Seed policy | Accepted (default) | no |
| 019 | Parallelism | Accepted (default) | no |
| 020 | Reporting rules | Accepted (default) | no |
| 021 | Multi-dataset evaluation | Accepted (PDF) | no |
| 022 | Why the system is closed-loop | Accepted (PDF + analysis) | no |
| 023 | Package name and import layering | Accepted (default) | no |
| 024 | Test-set access guard | Accepted (default) | no |
| 025 | Configuration format and layering | Accepted (default) | no |

---

## ADR-001: PSO as the optimizer
- **Decision:** use Particle Swarm Optimization (global-best, standard velocity form), implemented from scratch in
  NumPy.
- **Rationale:**
  - The fitness has no gradient, since accuracy is piecewise constant in the hyperparameters, and PSO is
    derivative-free.
  - Each evaluation is expensive, and a small swarm shares information between particles.
  - The search space is a low-dimensional bounded box, which PSO handles natively.
  - PSO has an explicit exploration/exploitation balance and is simple to implement and explain.
  - The course requires one optimization technique, and the proposal names PSO (PDF slides 4, 8, 9).
- **Alternatives:**
  - *Genetic Algorithm:* workable, but needs operators for mixed integer ranges and has more parameters.
  - *Ant Colony Optimization:* designed for path and combinatorial problems.
  - *Grid search:* 25,821 configurations, which is impractical.
  - *Random search:* no feedback. It is used here as the open-loop control, not as the optimizer (ADR-008).
  - *Bayesian optimization:* strong, but outside the course's evolutionary scope.
- **Consequences:** a PSO-specific integer mapping (ADR-011) and boundary rule (ADR-012) are needed. Libraries such as
  pyswarms and Optuna are forbidden (OR-016).
- **PPT impact:** none.

## ADR-002: Random Forest as the system being optimized
- **Decision:** scikit-learn `RandomForestClassifier` inside a pipeline.
- **Rationale:**
  - It is the proposal's system (PDF slide 6).
  - Its hyperparameters are few and interpretable.
  - It is fast on these datasets: a worst-case 5-fold evaluation measured 0.24 s on Iris and 0.52 s on Digits.
  - It needs no feature scaling.
- **Alternatives:** gradient boosting or SVM. These would change the project and are not considered.
- **Consequences:** the other RF parameters are fixed at scikit-learn defaults, so results depend on the
  scikit-learn version (pinned to 1.7.2 and recorded in the manifest).
- **PPT impact:** none.

## ADR-003: The three hyperparameters and their bounds
- **Decision:** $x = [n\_estimators \in [50,200],\ max\_depth \in [2,20],\ min\_samples\_split \in [2,10]]$, all
  integers, exactly as in PDF slides 6–8.
- **Rationale:** these are the proposal's decision variables, and changing them would change the core of the
  project. They cover ensemble size, tree complexity and regularization.
- **Critique, recorded honestly:**
  1. Accuracy is roughly non-decreasing in `n_estimators`, so PSO is expected to drift toward 200. The variable acts
     more as a cost knob than a tuning knob.
  2. `max_depth ≤ 20` may be binding on Digits, where the default RF grows unlimited-depth trees.
  3. `max_features`, often RF's most influential hyperparameter, is not tuned.
- **Alternatives:** add `max_features` or `min_samples_leaf`, or widen the depth bound. Kept as optional extensions
  (EXPERIMENT_PLAN §13).
- **Consequences:** boundary hits are recorded and reported (`final.json.boundary_hits`). The baseline's
  `max_depth=None` lies outside the search space (ADR-008).
- **PPT impact:** none.

## ADR-004: Validation fitness, and the test set is never used as fitness
- **Decision:** fitness is computed only from the optimization portion (via inner CV). The held-out test fold is used
  once per method, after optimization ends.
- **Rationale:** selecting hyperparameters by test performance would leak the test set into the model choice and
  inflate the reported performance (PDF slides 6 and 7 state this explicitly).
- **Alternatives:** none acceptable.
- **Consequences:** a test-set guard (ADR-024) and isolation tests (IT-04 to IT-07).
- **PPT impact:** none.

## ADR-005: Inner 5-fold stratified CV, with folds fixed per run
- **Decision:** $F_k(\theta)$ is the mean accuracy over `StratifiedKFold(5, shuffle=True, random_state=k)` on the
  optimization portion. The partition is created once per run and reused for every evaluation.
- **Rationale (design review):**
  - Every training sample is validated exactly once in k-fold CV, so the fitness resolution is $1/n$ for any k. What
    k changes is the per-fold training size: 67% of the data for 3-fold, 80% for 5-fold.
  - `min_samples_split` and `max_depth` act on absolute sample counts, so tuning them at 80% size transfers better to
    the final refit at 100%.
  - 5-fold also has lower estimator variance.
  - With the folds run in parallel, measured worst-case time was 0.20 s vs. 0.24 s on Iris and 0.48 s vs. 0.52 s on
    Digits, so the extra cost is only 4–20%.
  - Fixing the folds within a run (common random numbers) makes $F_k$ deterministic. Particles are compared on
    identical data, the cache is exact, and pbest is not a lucky draw.
- **Alternatives:**

  | Alternative | Why rejected |
  |---|---|
  | 3-fold (the PPT) | smaller training folds |
  | a fixed hold-out validation split | Iris would get 30 validation samples, so fitness moves in 3.3-point steps, and PSO would overfit one split |
  | 10-fold | marginal gain; 12-sample validation folds on Iris |
  | repeated k-fold | its cost is better spent on outer repetitions (ADR-007) |
  | RF out-of-bag score | biased toward more trees, and `n_estimators` is a decision variable; RF-specific; cannot refit the imputer per bootstrap |
  | re-drawing folds for every evaluation | makes the fitness stochastic, breaks caching, and makes pbest favour lucky draws |

- **Consequences:** validation folds are smaller (Iris 24, Heart ≈48). **Timing gate at Phase 6:** if the projected
  total runtime exceeds 2 hours, fall back to 3-fold and record an update here.
- **PPT impact:** slide 10, "3-Fold Stratified Cross-Validation" → "5-Fold", and "Mean 3-Fold Validation Accuracy" →
  "Mean 5-Fold". Slide 4, Objective 04 ("k-fold") needs no change.
- **Timing gate result (Phase 6, 2026-09-24):** measured with `scripts/benchmark_eval.py` on the optimization portion
  of outer fold 0 (5 inner folds run in parallel, RF `n_jobs = 1`, cache off, median of 3 repetitions after one
  warm-up evaluation; Windows 11 build 26200, 20 CPUs, Python 3.10.11, scikit-learn 1.7.2). Evaluations per dataset
  = 5 × (210 + 210) + 210 = 2,310.

  | Dataset | Optimization rows | $t_{eval}$ worst case (200, 20, 2) | $t_{eval}$ mid-range (125, 11, 6) | Upper bound (2,310 × worst) | Typical (2,310 × mid) |
  |---|---|---|---|---|---|
  | Iris | 120 | 0.220 s | 0.128 s | 8.5 min | 4.9 min |
  | Digits | 1,437 | 0.533 s | 0.325 s | 20.5 min | 12.5 min |
  | Heart (Cleveland) | 242 | 0.222 s | 0.143 s | 8.5 min | 5.5 min |
  | **Total** | | | | **37.5 min (0.62 h)** | **22.9 min** |

  **Decision:** the projected upper bound (37.5 min) is below the 2-hour gate, so **5-fold inner CV is kept** and the
  3-fold fallback is not used.

## ADR-006: Accuracy as the fitness metric (configurable)
- **Decision:** fitness = mean inner-CV **accuracy**. The config key `fitness.metric ∈ {accuracy, balanced_accuracy}`
  selects it, with a gate: a dataset whose max/min class ratio exceeds 1.5 uses `balanced_accuracy`. Balanced
  accuracy and macro F1 are logged as diagnostics and are **never fed back**.
- **Rationale (design review):**
  - The PDF defines fitness as mean validation accuracy (slide 10).
  - Class balance, measured: Iris 1.00, Digits 1.05, Cleveland 1.18 (Phase 2 audit). At this level of
    balance, accuracy and balanced accuracy are essentially interchangeable (identical for Iris), so switching buys
    nothing measurable.
  - Accuracy is the easiest metric to explain.
  - The gate keeps the pipeline correct if an imbalanced dataset is added.
- **Alternatives:**
  - *Balanced accuracy:* no gain here, and it deviates from the PPT.
  - *Macro F1:* fold-averaged F1 is biased on small folds and ill-defined when a fold predicts no member of a class.
    That makes it a noisy feedback signal, though it is fine as a test metric.
  - *Log-loss or Brier score:* smooth, and it would break ties, but it optimizes probability calibration rather than
    classification. It penalizes deep trees for overconfidence and changes what "best" means.
- **Consequences:** Iris will show many ties at the top, which is reported as `n_ties_with_best`. A tied gbest is never
  called "the optimum".
- **PPT impact:** none.

## ADR-007: Outer 5-fold nested cross-validation plus a deployment run
- **Decision:**
  - The test protocol is `StratifiedKFold(5, shuffle=True, random_state=42)`. Run *k* holds out fold *k* as its test
    set, runs the complete closed loop on the other 80%, refits, and tests once.
  - Seeds: run *k* uses seed *k*.
  - A deployment PSO run on the full dataset (seed 5) produces the single recommended configuration. Its performance
    estimate is the outer-CV mean.
- **Rationale (design review):**
  - A single 80/20 split (the PPT) gives Iris a 30-sample test set, where one error equals 3.3 points, and Heart about
    61.
  - With outer folds, **every sample is tested exactly once**: 150 test predictions on Iris and 303 on Heart.
  - It also gives 5 paired baseline-vs-optimizer comparisons.
  - The compute equals running 5 PSO seeds on one split, which the proposal already intended (Objective 08).
  - Each run remains a complete, independent closed loop.
- **Alternatives:**
  - *Single split with 5 seeds:* the test-set luck is fixed across all runs.
  - *Repeated random 80/20 splits:* the test sets overlap and coverage is uneven.
  - *Repeated nested CV (2×5):* more stable, but doubles the cost for a mini-project. Available as a config change.
- **Consequences:**
  - There are 5 fold-level best configurations per dataset. Their spread is reported; the deployment run gives the
    single recommendation.
  - "Held-out test set" is explained as "held-out test fold".
  - Statistics are descriptive only (ADR-020).
- **PPT impact:**
  - Slide 6, box 2: "Train/Test Split" → "Outer 5-fold split (train / held-out test fold)"; box 9 → "held-out test
    fold".
  - Slide 10, the input panel: the same change.
  - Slide 11: add "every sample tested once across 5 folds".

## ADR-008: Comparators are the default RF plus an equal-budget random search
- **Decision:**
  - The **baseline** is `RandomForestClassifier()` with defaults (100 trees, `max_depth=None`,
    `min_samples_split=2`) in the same pipeline, with the same seed and protocol.
  - The **random search** evaluates 210 i.i.d. uniform configurations from Ω, using the same evaluator and folds,
    with a fresh cache and **no feedback**.
- **Rationale (design review):**
  - Beating the baseline shows that *tuning* helps. Beating random search shows that *PSO's feedback* helps, which is
    the project's actual claim and the academic requirement (a closed loop).
  - The anytime curve (best-so-far vs. evaluations) is direct visual evidence of the loop's value.
  - It costs about 20 lines and roughly doubles compute, to an estimated 40 minutes worst case in total.
- **Alternatives:**
  - *Baseline only (the PPT):* cannot separate "tuning" from "feedback".
  - *Grid search:* too costly.
  - *A mid-range in-space configuration as the baseline:* less meaningful than "what you get without tuning".
- **Consequences:**
  - PSO may tie random search on a 3-variable space; that is reported honestly as a result.
  - The baseline lies outside Ω (unlimited depth), which is documented.
- **PPT impact:** slide 4, Objective 06 → "Compare the optimized RF against the default RF **and an equal-budget
  random search**". Slide 11: add a "PSO vs open-loop random search" outcome.

## ADR-009: Heart Disease is the UCI Cleveland subset, vendored with a checksum
- **Decision:** use `processed.cleveland.data` from the UCI Heart Disease dataset (id 45, DOI 10.24432/C52P4X,
  CC BY 4.0).
  - It is downloaded once by a script and committed under `data/raw/` with its SHA-256 and citation.
  - The target is `num > 0 → 1`.
  - The 13 standard features are used.
- **Rationale (design review):**
  - It has clear provenance and a stable file, and it is the most-cited version, which makes comparison with the
    literature possible. It matches the PPT's UCI citation.
  - Only a few values are missing, which also exercises the leakage-safe imputation.
  - Committing the file removes any network dependency and version drift.
- **Alternatives:**

  | Version | Why rejected |
  |---|---|
  | Kaggle `heart.csv` (1,025 rows) | only about 302 unique rows; copies fall into both train and test, contaminating the test set and inflating scores; undocumented provenance; target coding reported inconsistent with UCI |
  | 4-site UCI combined (≈920 rows) | site confounding (disease prevalence differs strongly by hospital); heavy missingness that is not at random (`ca`/`thal`/`slope` mostly absent outside Cleveland; `chol = 0` means missing) |
  | Statlog Heart (270 rows) | the credible runner-up: cleanest data, no missing values; but smaller, less documented lineage, and leaves the imputation rule untested |
  | OpenML `heart-c` | the same data with string categoricals and an extra service dependency |

- **Consequences:** the row count, missing counts and class split were measured in Phase 2 and match the
  expectation: 303 rows; 4 missing in `ca` and 2 in `thal`; 164 vs. 139 (`data/DATASET_AUDIT.md`).
- **PPT impact:** none. It clarifies "Heart Disease (binary, real-world)".

## ADR-010: Preprocessing inside the model pipeline
- **Decision:** preprocessing that learns from data is a pipeline step. For Heart that is
  `SimpleImputer(strategy="most_frequent")`; the other datasets need none. Integer-coded categoricals are kept as they
  are. Nothing is fitted on the full dataset before splitting.
- **Rationale:** the pipeline is refit on each inner-CV training part, the final refit and the baseline, so there is
  no leakage (BRIEF principle 9). The most-frequent strategy suits `ca` (count 0–3) and `thal` (categorical). Trees
  split integer codes adequately.
- **Alternatives:**
  - *Drop the ~6 incomplete rows:* also leakage-free and simpler, but loses data and leaves the rule untested.
  - *RF native NaN support (scikit-learn ≥ 1.4):* less transparent.
  - *One-hot encoding:* an optional ablation.
- **Consequences:** a small imputer is fitted per fit, which costs almost nothing. Leakage tests UT-20 and IT-11
  cover it.
- **PPT impact:** none.

## ADR-011: Integer handling uses a continuous state with decoding at evaluation
- **Decision:** positions, velocities and pbest/gbest positions stay continuous. The configuration evaluated is
  `decode(x) = clip(rint(x), l, u)`, with ties rounding to even (NumPy `rint`). pbest and gbest store both the
  continuous position and the decoded configuration.
- **Rationale:** rounding the state itself makes any velocity component below 0.5 round back to the same integer, so
  particles stall. A continuous state lets small moves accumulate. This is the standard, simplest integer adaptation.
- **Alternatives:** round the state each step (stalls); binary or discrete PSO (more complex, and departs from the
  standard equations); probabilistic rounding (adds noise to a deterministic fitness).
- **Consequences:** many positions map to the same configuration, which produces duplicate candidates (ADR-016).
- **PPT impact:** none. It fills the gap left by "integer-valued parameters" on slide 6.

## ADR-012: Boundary handling and velocity limits
- **Decision:**
  - Absorbing wall: if $x + v$ leaves $[l_d, u_d]$, set $x_d$ to the bound and $v_d = 0$.
  - Velocity clamp: $v_{max,d} = 0.2 (u_d - l_d)$, i.e. (30, 3.6, 1.6).
  - Initial positions are uniform in the box. Initial velocities are uniform in $\pm 0.1 (u_d - l_d)$.
- **Rationale:** absorbing is simple and keeps positions valid. It lets particles rest *on* a bound, which matters
  because optima may lie there (see ADR-003's critique). The clamp stops particles from crossing the whole box in one
  step within the short 20-iteration budget. Small initial velocities avoid most particles hitting the walls on the
  first move.
- **Alternatives:**
  - *Reflect:* good for interior optima, but it pushes particles away from boundary optima.
  - *Random re-initialization:* throws away information.
  - *No clamp:* risk of velocity explosion.
  - *Clamp = range:* the Eberhart–Shi setting with constriction, too loose for this budget.
  - *Zero initial velocity:* also common, but the first step is then driven only by pbest and gbest.
- **Consequences:** the clamp can limit how fast particles travel toward a distant gbest (see the worked example in
  MATHEMATICAL_FORMULATION §9). Both fractions are configurable.
- **PPT impact:** none. It fills the "Apply Bounds" step on slides 9 and 10.

## ADR-013: PSO coefficients
- **Decision:** constant $w = 0.7298$, $c_1 = c_2 = 1.49618$. $r_1, r_2 \sim U[0,1)$ are drawn per particle, per
  dimension, per iteration from one seeded `numpy.random.Generator`.
- **Rationale:** this is the constriction-equivalent parameter set (Clerc & Kennedy 2002, as popularized by Eberhart
  & Shi 2000), with a published convergence analysis. It is widely used and has no schedule to justify. Equal
  coefficients balance the cognitive and social pulls.
- **Alternatives:** linearly decreasing inertia from 0.9 to 0.4 (Shi & Eberhart 1998), which is fine but ties behaviour
  to T; $c_1 = c_2 = 2$ with $w = 1$ (the original 1995 form), which is unstable without a clamp.
- **Consequences:** fixed across datasets by default; the values are overridable per dataset.
- **PPT impact:** none. The PDF shows $w$, $c_1$ and $c_2$ symbolically.

## ADR-014: Swarm size, iterations, topology and update order
- **Decision:** N = 10 particles (PDF slide 10); T = 20 update iterations after the initial evaluation, giving 210
  evaluations per run. Global-best topology. **Synchronous** update: evaluate all particles, then update pbest and
  gbest, then move all particles.
- **Rationale:**
  - 210 evaluations is less than 1% of Ω and affordable (an estimated ≤ 20 minutes for the slowest dataset).
  - Ten particles are ample for three dimensions.
  - The gbest topology matches the PDF.
  - A synchronous update matches the PDF's step order (slide 9) and is order-independent, so the result does not
    depend on the order in which particles are evaluated. That keeps it deterministic and parallelizable.
- **Alternatives:** larger swarms or more iterations (a config change); ring or lbest topology (more robust to
  premature convergence, but unnecessary at this scale); asynchronous updates (they react faster, but depend on
  particle order).
- **Consequences:** random search receives the same 210-evaluation budget (ADR-008).
- **PPT impact:** none. It fills in the unspecified iteration count.

## ADR-015: Stopping and the definition of convergence
- **Decision:** stop at `max_iter = 20` by default. An optional patience rule (stop when gbest has improved by less
  than `tol = 1e-4` over `P = 5` iterations) is **disabled by default**.
  - **Convergence** is reported, not enforced. Each run records the convergence iteration (the last gbest
    improvement), `stop_reason`, and a status: `plateaued` if gbest did not improve in the final 3 iterations,
    otherwise `improving_at_end`.
- **Rationale:** a fixed budget keeps the PSO-vs-random-search comparison fair and makes run costs identical and
  predictable. On a piecewise-constant fitness, "no improvement for P iterations" is common early on, so early
  stopping could end runs prematurely.
- **Alternatives:** patience on by default (unequal budgets); stopping on swarm diversity (extra parameter).
- **Consequences:** some iterations may be spent after convergence. They are cheap thanks to caching (ADR-016).
- **PPT impact:** none. It makes "until convergence or maximum iterations" concrete.

## ADR-016: Fitness caching and duplicate candidates
- **Decision:** the evaluator caches `FitnessResult` by the decoded integer configuration within one run, with a fresh
  cache per method and fold. A cache hit is still a logged evaluation that counts toward the budget
  (`cache_hit = true`, `fit_time_s = 0`). Reports show both evaluations and unique fits.
- **Rationale:** the fitness is deterministic within a run (ADR-005), so caching is exact. Converging swarms produce
  many duplicate configurations, and caching makes late iterations nearly free. Counting cache hits as evaluations
  keeps the budget definition identical for PSO and random search.
- **Alternatives:** no cache (wasteful); de-duplicating particles by perturbing them (changes PSO's dynamics); a
  shared cache across methods (one method would benefit from the other's work).
- **Consequences:** `n_unique_fits` measures true computational cost. The cache-hit rate is a convergence indicator.
- **PPT impact:** none.

## ADR-017: Tie-breaking
- **Decision:**
  - pbest and gbest are replaced only on **strict** improvement (`>`).
  - Among particles tying for a new gbest in the same iteration, the lowest particle index wins.
  - For random search, the earliest evaluation wins.
- **Rationale:** deterministic and simple; it follows the standard PSO formulation.
- **Alternatives:** prefer the cheaper configuration on ties (fewer trees), which is attractive but changes the
  objective into a lexicographic one; random tie-breaking, which adds noise.
- **Consequences:** on flat landscapes (Iris), the reported best is "first found among equals". This is disclosed via
  `n_ties_with_best`.
- **PPT impact:** none.

## ADR-018: Seed policy
- **Decision:**
  - `outer_seed = 42` drives the outer split.
  - For outer fold *k*, `run_seeds[k] = k` (0–4) drives the PSO generator, the random search generator, the inner CV
    shuffle, and RF `random_state` for every fit in that run (baseline, candidates, final refit).
  - `deployment_seed = 5` plays the same roles for the deployment run.
  - No global random state, no Python `random` module, no time-based seeds.
- **Rationale:** each run is fully specified by one integer. Using the same seed for all methods in a fold makes the
  comparisons paired. The seeds differ across folds, so the 5 runs cover the stability study in Objective 08.
- **Alternatives:** separate seeds per component (more knobs, no benefit); a `SeedSequence` spawn tree (more robust
  stream independence, but harder to explain; not needed at this scale).
- **Consequences:** the whole experiment is reproducible from the config file alone (RR-005).
- **PPT impact:** none.

## ADR-019: Parallelism
- **Decision:** RF uses `n_jobs=1`. The 5 inner CV folds run in parallel (`cross_validate(..., n_jobs=5)`, loky
  backend). Particles are evaluated sequentially.
- **Rationale:** measured on the development machine (20 cores), tree-level parallelism is **slower** on small data
  (Iris: 0.97 s vs. 0.20 s with fold-level parallelism) because of thread overhead. Fold-level parallelism is
  deterministic, since each fold clones the same estimator with the same `random_state`. Sequential particles keep the
  code simple and the order fixed.
- **Alternatives:** parallel particles (listed as an extension point; would need a batch objective); tree-level
  `n_jobs=-1` (slower here).
- **Consequences:** on Windows, the CLI must guard its entry point with `if __name__ == "__main__":`. IT-08 verifies
  that parallel and serial runs give identical results.
- **PPT impact:** none.

## ADR-020: Reporting rules
- **Decision:**
  1. Only held-out test-fold metrics are called "performance".
  2. Best validation fitness is always labelled "optimistically biased (selection over N evaluations)".
  3. Statistics are descriptive only (mean ± sd, paired deltas, win/tie/loss, pooled metrics), with **no p-values**.
  4. Differences within one or two test samples are described as ties or noise.
  5. Null results are reported as null results.
  6. Every number cites its result file.
- **Rationale:**
  - The winner's curse: the maximum of 210 noisy estimates is biased upward.
  - With 5 pairs, the minimum two-sided Wilcoxon p-value is 0.0625, so significance is impossible, and CV folds are
    not independent anyway.
  - The brief forbids fabricated or overstated claims.
- **Alternatives:** corrected resampled t-tests (Nadeau & Bengio), which are unnecessary for a descriptive
  mini-project study.
- **Consequences:** conclusions are cautious and honest; the reader sees the variability.
- **PPT impact:** the results slide will follow these rules.

## ADR-021: Multi-dataset evaluation
- **Decision:** evaluate on Iris, Digits and Heart Disease (Cleveland) through one pipeline, with PSO parameters shared
  across datasets.
- **Rationale:** these are the proposal's datasets (slides 4–5, Objective 09). They cover a small, easy set; a larger
  10-class image set; and a small, noisy clinical set with missing values. That tests whether conclusions generalize.
- **Alternatives:** a single dataset (weaker evidence); more datasets (possible later through the registry).
- **Consequences:** dataset-specific behaviour (e.g. flat fitness on Iris) is expected and reported per dataset.
- **PPT impact:** none.

## ADR-022: Why the system is closed-loop
- **Decision:** the system is presented as an iterative **closed optimization loop**:
  - **Controller:** PSO.
  - **Control action:** the decoded hyperparameters.
  - **Plant:** training plus validation of the RF.
  - **Measurement and feedback:** the inner-CV accuracy.
  - **Controller adaptation:** the pbest/gbest memory plus the velocity and position updates.
  - The final test evaluation is a separate, open, one-way path.
- **Rationale:** it meets the faculty definition exactly. The optimizer's output changes a system parameter, the
  system responds, the response is measured and fed back, and the next output depends on it. This is shown formally
  in MATHEMATICAL_FORMULATION §8 and evidenced by IT-01, IT-02 and the random-search comparison.
- **Honest limits:** the plant is static (no dynamics), there is no setpoint (the loop is extremum-seeking), and the
  controller acts through memory. It is not a physical real-time control loop.
- **Alternatives:** present it as "just an optimizer" (fails the academic requirement); overclaim control-theoretic
  properties (inaccurate).
- **Consequences:** the diagrams always show the feedback arrow and the separated test lane. Review 2 demonstrates the
  loop live, through the per-iteration trace.
- **PPT impact:** none. It strengthens the slide 6 narrative.

## ADR-023: Package name and import layering
- **Decision:** the code lives in `src/pso_rf/`. The `optimization` package imports only NumPy and the standard
  library; `evaluation` never imports `optimization`; only `experiments` wires them together. UT-22 enforces this.
- **Rationale:** the brief requires PSO without dataset logic and an evaluator without PSO logic, and making the rule
  testable enforces it. The namespace avoids clashes with the HuggingFace `datasets` package and generic `models` or
  `utils` modules.
- **Alternatives:** a flat `src/datasets/`, `src/models/` layout (import collisions); a single module (untestable
  separation).
- **Consequences:** slightly more files, and a clear architecture.
- **PPT impact:** none.

## ADR-024: Test-set access guard
- **Decision:**
  - Test arrays live inside `HeldOutTestSet` and are reachable only through `reveal()`.
  - The runner wraps every optimization in `with OptimizationPhase():`, and `reveal()` raises `TestSetAccessError`
    inside it.
  - The evaluator's constructor rejects a `HeldOutTestSet`.
  - The optimizers take no data at all.
- **Rationale:** test isolation becomes a runtime-enforced, testable property rather than a convention.
- **Alternatives:** keeping test data in a separate process (overkill); relying on code review alone (weak).
- **Consequences:** an accidental leak crashes loudly. Tests UT-15 and IT-05 to IT-07 prove isolation.
- **PPT impact:** none.

## ADR-025: Configuration format and layering
- **Decision:** YAML configs are layered as `default.yaml` ← experiment file ← `datasets.<name>` block ← CLI `--set`.
  The result is validated into frozen dataclasses and saved as `config.resolved.json` with a SHA-256 hash.
- **Rationale:** it is human-readable, supports both global and per-dataset PSO parameters (the brief's options A and
  B), and gives one file that fully describes an experiment. PyYAML 6.0.1 is already installed.
- **Alternatives:** JSON (less readable, no comments); Hydra (heavy for a mini-project); CLI arguments only (not
  reproducible).
- **Consequences:** every run is described by a hashable config.
- **PPT impact:** none.
