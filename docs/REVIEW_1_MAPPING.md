# REVIEW 1 MAPPING

**Status: Review 1 completed (2026-09). The slide-edit list below is carried forward as input for the Review 2 deck.**

This document maps the faculty's Review 1 requirements to the project, the documents that define each item, the
evidence (or code) that will back it, and the slide that presents it. The current deck is
`ppt/CB.EN.U4ELC23005_ANUSH_RAMESH_PPT.pdf`, with 13 pages: 12 content slides plus a thank-you slide.

## 1. Requirement → implementation → evidence → slide

| Review 1 requirement | Project implementation | Defining document | Evidence / code | Slide (current → proposed) |
|---|---|---|---|---|
| **Block diagram** | Dataset → outer 5-fold split → [closed loop: PSO → candidate hyperparameters → RF → inner 5-fold validation → fitness → PSO update] → best configuration → final RF → held-out test fold → metrics | ARCHITECTURE §A, §C | Mermaid and ASCII diagrams; later F1 and F2 | current 6 → **proposed 5** |
| **Methods** | PSO (from scratch) for hyperparameter optimization; RF as the system; stratified inner 5-fold CV as the validation measurement; outer 5-fold CV for unbiased testing; default RF and equal-budget random search as comparators | IDEA §3, §12; EXPERIMENT_PLAN §4–§7 | `pso_rf.optimization`, `pso_rf.evaluation`, `pso_rf.experiments.runner` (phases P6–P11) | current 10 → **proposed 8** |
| **Objective function** | Maximize $F_k(\theta)$ = mean 5-fold stratified CV accuracy of RF(θ) on the optimization portion of fold *k* | MATHEMATICAL_FORMULATION §6 | `FitnessEvaluator` (P6); UT-09 | current 7 → **proposed 6** |
| **Constraints** | 50 ≤ n_estimators ≤ 200; 2 ≤ max_depth ≤ 20; 2 ≤ min_samples_split ≤ 10; all integers; test data excluded from the objective | MATHEMATICAL_FORMULATION §3, §4, §10 | `SearchSpace`, absorbing wall, `decode`; UT-01, UT-05; isolation tests IT-04 to IT-07 | current 6, 7 → **proposed 6** |
| **Decision variables** | $x = [n\_estimators, max\_depth, min\_samples\_split]$; one particle = one candidate configuration, e.g. [120, 8, 4] | MATHEMATICAL_FORMULATION §2 | `IntParam` × 3 in `configs/default.yaml` | current 7 → **proposed 6** |
| **Program snippet** | The PSO core loop: velocity update, clamp, position update, absorbing wall, decode, evaluate, pbest/gbest. **Until Phase 7 this is the documented pseudocode in MATHEMATICAL_FORMULATION §7.7.** After Phase 7 it is replaced by the real excerpt from `src/pso_rf/optimization/pso.py`, which must match it. | MATHEMATICAL_FORMULATION §7.7 | `pso.py` (P7); UT-03 proves the implemented update equals the worked example | **new → proposed 8** |
| **Optimization technique** | Particle Swarm Optimization: position, velocity, pbest, gbest, $v \leftarrow wv + c_1r_1(p-x) + c_2r_2(g-x)$, $x \leftarrow x + v$; $w = 0.7298$, $c_1 = c_2 = 1.49618$, N = 10, T = 20 | MATHEMATICAL_FORMULATION §7; DECISIONS ADR-013, ADR-014 | `PSOOptimizer` | current 9 → **proposed 7** |
| **Why PSO** | Derivative-free; population-based; native to bounded boxes; balances exploration and exploitation; simple and explainable; suited to expensive evaluations in a low-dimensional space; alternatives compared | IDEA §6; DECISIONS ADR-001 | — | current 8 → **proposed 7** |
| **Closed-loop feedback** | Fitness is fed back to PSO, which updates pbest, gbest, velocity and position; the new positions become new RF hyperparameters; the loop repeats for 20 iterations. The test path is separate and one-way. Control-concept mapping table. | IDEA §9; ARCHITECTURE §C; DECISIONS ADR-022 | IT-01, IT-02; later the per-iteration trace and the anytime curve (PSO vs. open-loop random search) | current 6, 7, 8, 10 → **proposed 5** |

## 2. Proposed ≤ 10-slide deck (current deck has 12 content slides)

| # | Proposed slide | Built from current slides | Content changes |
|---|---|---|---|
| 1 | Title | 1 | none |
| 2 | Problem statement and motivation | 2 + 3 | Merge: the hyperparameter problem, why manual tuning fails, and the motivation list. Keep the three hyperparameter cards. |
| 3 | Objectives | 4 | Objective 06 → "compare with the default RF **and an equal-budget random search**". Optionally group 9 objectives into 6 to save space. |
| 4 | Datasets and evaluation protocol | 5 (+ new) | Add a row of facts (samples, classes; Heart = UCI Cleveland) and a small outer/inner CV diagram. |
| 5 | System architecture: closed loop | 6 | Box 2 → "Outer 5-fold split"; box 9 → "Held-out test fold"; keep the feedback arrow; add the control-concept legend (controller = PSO, plant = RF plus validation, feedback = fitness). |
| 6 | Optimization formulation | 7 | Objective text "mean 5-fold CV accuracy"; constraints and decision variables unchanged. |
| 7 | PSO: why and how | 8 + 9 | Merge: the why-PSO list (4 points), the update equations, the particle and swarm picture, and the parameter values (w, c1, c2, N, T). |
| 8 | Implementation methodology and program snippet | 10 (+ snippet) | "3-Fold" → "5-Fold" (twice); add the core-loop snippet (pseudocode now, real code after Phase 7). |
| 9 | Expected results and evaluation plan | 11 | Add the PSO vs. random search (feedback vs. no feedback) outcome; keep "improve or maintain"; **no numbers**. |
| 10 | References | 12 | **Fix:** the reference text is currently invisible on the rendered slide (it exists only in the PDF text layer). Re-type it visibly and add Clerc & Kennedy (2002) and Bergstra & Bengio (2012). |
| — | Thank you | 13 | Drop it, or fold it into slide 10's footer, to stay within 10 slides. |

## 3. Required PPT edits (from the accepted design decisions)

| Current slide | Edit | Reason |
|---|---|---|
| 4 (Objectives) | Obj 06: "…against the baseline" → "…against the default RF **and an equal-budget random search**" | ADR-008 |
| 6 (Architecture) | Box 2 "TRAIN / TEST SPLIT" → "OUTER 5-FOLD SPLIT"; box 9 "HELD-OUT TEST DATA" → "HELD-OUT TEST FOLD (one per run)" | ADR-007 |
| 10 (Methodology) | Step 4 "3-Fold Stratified Cross-Validation" → "5-Fold…"; step 5 "Mean 3-Fold Validation Accuracy" → "Mean 5-Fold…"; the input panel's split → outer folds | ADR-005, ADR-007 |
| 11 (Expected results) | Add "PSO reaches good configurations with fewer evaluations than open-loop random search (to be tested)" | ADR-008 |
| 12 (References) | Make the text visible; add references 4 and 5 from IDEA.md | presentation defect |

## 4. Review 1 talking points (for the viva)

1. **What is the system?** A Random Forest classifier whose three hyperparameters are the adjustable parameters.
2. **Where is optimization applied?** On those hyperparameters. PSO is a separate block that proposes them.
3. **Why PSO?** No gradients, a bounded low-dimensional space, expensive evaluations, and it is population-based and
   simple.
4. **How does it work here?** Each particle is a configuration; fitness is the 5-fold CV accuracy; pbest and gbest
   pull particles toward good regions; positions are rounded to integers only for evaluation; the walls keep them in
   bounds.
5. **Where is the feedback?** Fitness goes back to PSO and changes the next configurations. Random search, which has
   no feedback, is the control.
6. **How is cheating prevented?** The test fold is sealed during optimization, enforced in code and proven by tests
   (for example, shuffling the test labels changes nothing in the optimization).
