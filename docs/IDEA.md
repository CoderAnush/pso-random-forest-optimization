# IDEA — PSO-Based Random Forest Hyperparameter Optimization

*A multi-dataset experimental study using Iris, Digits and Heart Disease classification.*

Anush Ramesh (CB.EN.U4ELC23005), EEE Department, Amrita Vishwa Vidyapeetham, Coimbatore.
Evolutionary Optimization mini-project.

> **Status of this document:** Phase 0 design. It contains no results. Every result mentioned is an
> *expected outcome* stated qualitatively, and the experiments that will produce real numbers have not been run.

---

## 1. Problem statement

Random Forest is a strong, widely used classifier, but how well it performs depends on its **hyperparameters**:
settings chosen before training rather than learned from data. Three of them matter directly for this project:

| Hyperparameter | Meaning | Effect |
|---|---|---|
| `n_estimators` | number of trees in the forest | more trees give more stable votes but cost more compute |
| `max_depth` | maximum depth of each tree | deeper trees fit more detail but risk overfitting |
| `min_samples_split` | minimum samples needed to split a node | larger values give simpler, more regularized trees |

Choosing these by hand is trial-and-error. It is slow, depends on the experimenter's experience, and can easily miss
better settings. The three variables together allow 151 × 19 × 9 = **25,821** integer combinations, and each one needs
a full train-and-validate cycle to score, so exhaustive search is impractical.

**Need:** an automated, efficient and reproducible way to find good Random Forest hyperparameters, with an honest
assessment of whether the result is actually better than the default configuration.

## 2. Motivation

The original proposal lists five motivations:
1. Automate hyperparameter selection.
2. Reduce manual experimentation.
3. Explore a discrete search space efficiently.
4. Compare the optimized forest against a baseline.
5. Study convergence and stability across random seeds.

This design adds one more:

6. **Show that the optimizer's feedback loop is what produces the improvement**, rather than simply trying many
   configurations. This is done by comparing PSO with a random search that tries the same number of configurations
   *without* feedback.

## 3. Proposed solution

Treat the Random Forest as a **system whose parameters are adjusted by an optimizer in a closed feedback loop**:

```
        ┌────────────────────────────────────────────────────────────────┐
        │                  CLOSED OPTIMIZATION LOOP                      │
        │                                                                │
        │   ┌──────────────┐  candidate          ┌───────────────────┐   │
        │   │ PSO          │  hyperparameters    │ Random Forest     │   │
        │   │ optimizer    │ ──────────────────▶ │ trained on the    │   │
        │   │ (controller) │                     │ training portion  │   │
        │   └──────▲───────┘                     └─────────┬─────────┘   │
        │          │                                       │             │
        │          │ fitness = validation accuracy         │ predictions │
        │          │ (feedback signal)                     ▼             │
        │   ┌──────┴───────┐                     ┌───────────────────┐   │
        │   │ PSO update   │ ◀────────────────── │ Validation        │   │
        │   │ pbest, gbest │     fitness         │ (5-fold CV inside │   │
        │   │ velocity,    │                     │ training data)    │   │
        │   │ position     │                     └───────────────────┘   │
        │   └──────────────┘                                             │
        │        repeat for 20 iterations × 10 particles                 │
        └────────────────────────────────┬───────────────────────────────┘
                                         │ best hyperparameters (after the loop ends)
                                         ▼
               Final Random Forest ──▶ HELD-OUT TEST DATA ──▶ final metrics
               (open path: nothing here ever flows back into the loop)
```

Each **particle** in the swarm is one candidate configuration, for example `[120, 8, 4]`, meaning 120 trees,
depth 8, and a minimum of 4 samples to split. The candidate is sent to the Random Forest, which is trained and
validated. The validation score, or **fitness**, is fed back to PSO. PSO uses it to move every particle toward
better regions, and the new positions become new candidate configurations. After a fixed number of iterations, the
best configuration found is used to train a final model, and that model is evaluated **once** on data the loop never
saw.

## 4. Why Random Forest

- It is a strong, general-purpose classifier for tabular and small image data (Breiman, 2001).
- Its behaviour is controlled by a few interpretable hyperparameters, which gives a clear, low-dimensional search
  space.
- It trains in well under a second on these datasets, so hundreds of evaluations are affordable.
- It is robust to feature scaling and handles integer-coded categorical features, so preprocessing stays minimal.
  That keeps the focus on optimization.

## 5. Why hyperparameter optimization

The same algorithm can perform noticeably better or worse depending on its settings. Hyperparameter optimization
turns "guess and check" into a defined optimization problem, with decision variables, constraints and an objective,
which is exactly the structure this course studies.

## 6. Why Particle Swarm Optimization

| Property of the problem | Why PSO fits |
|---|---|
| No gradient exists: accuracy is a step function of the hyperparameters | PSO is **derivative-free** and only needs fitness values |
| Each evaluation is expensive (a full train-and-validate cycle) | PSO's population shares information, so each evaluation informs all particles |
| The variables have natural bounds | PSO works directly in a **bounded box** |
| Search needs both breadth and focus | PSO balances **exploration** (inertia, personal memory) and **exploitation** (attraction to the swarm's best) |
| Low dimension (3 variables) | A small swarm (10 particles) is enough, keeping cost low |
| Simple to implement and explain | Two update equations, which the course asks us to implement **from scratch** |

Compared with the alternatives in the course: a Genetic Algorithm needs crossover and mutation operators designed for
mixed integer ranges, and Ant Colony Optimization suits path and combinatorial problems rather than numeric ranges.
PSO is the most direct fit for a small, bounded, numeric search space (see [DECISIONS.md](DECISIONS.md), ADR-001).

## 7. How PSO works in this project

Each particle *i* has a **position** $x_i$ (a candidate configuration) and a **velocity** $v_i$ (its direction and
step size). It also remembers its **personal best** $p_i$, the best configuration it has personally found. The swarm
shares a **global best** $g$, the best configuration any particle has found.

In each iteration, every particle updates:

$$v_i \leftarrow w\,v_i + c_1 r_1 (p_i - x_i) + c_2 r_2 (g - x_i), \qquad x_i \leftarrow x_i + v_i$$

The three terms of the velocity update:
- **Inertia** $w\,v_i$ keeps some of the previous movement.
- **Cognitive** $c_1 r_1 (p_i - x_i)$ pulls toward the particle's own best.
- **Social** $c_2 r_2 (g - x_i)$ pulls toward the swarm's best.
- $r_1, r_2$ are fresh random numbers in [0, 1) that keep the search stochastic.

Because the hyperparameters are integers, particles move in continuous space and are **rounded and clipped to the
allowed ranges** only when a configuration is evaluated. The full formulation is in
[MATHEMATICAL_FORMULATION.md](MATHEMATICAL_FORMULATION.md).

## 8. The system as blocks

| Block | Role | Input | Output |
|---|---|---|---|
| **Dataset** | Iris, Digits or Heart Disease | — | features X, labels y |
| **Outer split** | Separates optimization data from **held-out test data** | X, y | optimization portion, test portion |
| **PSO optimizer** (the optimization block) | Proposes candidate hyperparameters | fitness feedback | candidate configuration |
| **Random Forest** (the system being optimized) | Trains with the candidate configuration | configuration and training data | trained model |
| **Validation** | Measures how well the model generalizes, using 5-fold cross-validation **inside** the optimization portion | trained models | validation accuracy |
| **Fitness function** | Turns the measurement into the feedback signal | validation accuracy | fitness value, sent back to PSO |
| **PSO update** | Updates memory (pbest, gbest) and moves particles | fitness | new candidate configurations |
| **Final model and test** | After optimization ends, refits on the whole optimization portion and scores it on the test portion | best configuration | final metrics |

## 9. The feedback mechanism, and why this is a closed loop

| Control-system idea | In this project |
|---|---|
| Plant (system being controlled) | "Train a Random Forest with these settings and measure its validation accuracy" |
| Controller | PSO |
| Manipulated variables / control action | the three hyperparameters of each particle |
| Measurement | 5-fold cross-validated accuracy on the optimization data |
| Feedback signal | the fitness value returned to PSO |
| Controller adaptation | updates to personal best, global best, velocity and position |
| One feedback cycle | one particle evaluated |
| One control step | one swarm iteration (all 10 particles evaluated, then all moved) |

The loop is **closed** because the next configurations PSO proposes depend on the fitness measured for the previous
ones. Remove the feedback and PSO cannot steer.

**Honest distinction.** This is an *iterative optimization loop*, not a physical real-time control loop. The "plant"
has no dynamics or time constant; the same configuration always gives the same score within a run. There is also no
setpoint to track: the loop seeks a maximum. It still meets the course definition: the optimizer's output changes a
system parameter, the system's response is measured, and that measurement is fed back to decide the next output.

**Evidence that the loop is real:**
1. Every evaluation is logged, showing how configurations change in response to fitness.
2. A convergence curve shows the best fitness rising over iterations.
3. PSO is compared with a random search of equal budget that has **no feedback** (an open-loop control). If PSO finds
   better configurations faster, the feedback is doing work.

## 10. Datasets

| Dataset | Task | Characteristics | Source |
|---|---|---|---|
| Iris | 3-class classification | small, 150 samples, perfectly balanced | scikit-learn bundled copy of the UCI dataset |
| Digits | 10-class handwritten digit recognition | 1,797 images of 8×8 pixels (64 features) | scikit-learn bundled copy of the UCI Optical Digits test set |
| Heart Disease | Binary classification (disease / no disease) | real-world clinical data, 303 patients, 6 missing values (measured in Phase 2) | UCI Heart Disease, **Cleveland** subset |

The Cleveland subset was chosen over larger versions found online. The widely circulated 1,025-row version is mostly
duplicated rows, which would leak test data into training. See [DECISIONS.md](DECISIONS.md), ADR-009.

## 11. Objective, decision variables and constraints

**Decision variables:** $x = [n\_estimators,\ max\_depth,\ min\_samples\_split]$

**Constraints:**
- $50 \le n\_estimators \le 200$
- $2 \le max\_depth \le 20$
- $2 \le min\_samples\_split \le 10$
- all three are integers

**Objective:** maximize $F(x)$, the mean 5-fold cross-validated **accuracy** of a Random Forest built with $x$,
measured only on the optimization portion of the data.

The held-out test data is **never** part of $F(x)$.

## 12. How the experiment is run

For each dataset:
1. The data is split into **5 outer folds**. Each fold takes one turn as the held-out test set while the other four
   are used for optimization.
2. In each turn, three methods are run on the optimization data:
   - the **default Random Forest** (no tuning);
   - **random search** (210 random configurations, no feedback);
   - **PSO** (10 particles × 21 evaluation rounds = 210 evaluations, with feedback).
3. Each method's chosen configuration is refit on the optimization data and tested **once** on the held-out fold.
4. Results are compared fold by fold, which gives 5 paired comparisons per dataset. Every sample in the dataset ends
   up in a test set exactly once.
5. A final PSO run on the full dataset produces the single recommended configuration. Its expected performance is the
   average over the 5 outer folds.

This is a stronger protocol than the single train/test split in the original proposal. The reasons are recorded in
[DECISIONS.md](DECISIONS.md) (ADR-005, ADR-007, ADR-008).

## 13. Expected outcomes (qualitative, not results)

- PSO should find configurations that **improve or maintain** classification performance relative to the default
  Random Forest. On easy datasets such as Iris, the default is already strong, so a tie is a realistic and acceptable
  outcome.
- The best validation fitness should rise over iterations and then plateau (**convergence**).
- PSO should reach good configurations in fewer evaluations than random search. This is a hypothesis to be tested,
  not a promise: on a small 3-variable space, random search may be competitive.
- Results should be **reproducible**: the same configuration file and seeds give the same results.
- Best configurations may differ between folds. Such variation would indicate a flat performance landscape, where
  many settings perform about equally well.

## 14. Limitations

- Test sets are small for Iris (30 samples per fold) and Heart Disease (about 61 per fold), so differences of one
  or two test samples are within noise.
- With 5 paired folds, no statistical test can show significance at the 5% level, so results are reported
  descriptively.
- Validation accuracy has coarse steps on small datasets, so many configurations tie. The "best" configuration is
  then one of several equally good ones.
- Only three hyperparameters are tuned. Others, such as `max_features`, can matter as much and are left at their
  defaults.
- Adding more trees rarely hurts accuracy, so PSO is expected to push `n_estimators` toward its upper bound. The number
  of trees acts more as a cost knob than a tuning knob.
- The best validation score found is optimistically biased, because it is the maximum of many noisy measurements.
  Only the held-out test results are reported as performance.

## 15. Future extensions

- Add `max_features` or `min_samples_leaf` as further decision variables.
- Multi-objective PSO, trading accuracy against model size or training time.
- Other PSO variants: linearly decreasing inertia, ring topology, or asynchronous updates.
- Parallel evaluation of particles.
- Comparison with other optimizers (Genetic Algorithm, Bayesian optimization) at equal budget.
- Additional or imbalanced datasets, which would activate the balanced-accuracy fitness option.

## References

1. J. Kennedy and R. Eberhart, "Particle swarm optimization," *Proc. IEEE Int. Conf. Neural Networks (ICNN)*,
   Perth, Australia, 1995, pp. 1942–1948.
2. L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.
3. D. Dua and C. Graff, *UCI Machine Learning Repository*, University of California, Irvine, 2019.
4. M. Clerc and J. Kennedy, "The particle swarm — explosion, stability, and convergence in a multidimensional complex
   space," *IEEE Trans. Evolutionary Computation*, vol. 6, no. 1, pp. 58–73, 2002.
5. J. Bergstra and Y. Bengio, "Random search for hyper-parameter optimization," *JMLR*, vol. 13, pp. 281–305, 2012.
