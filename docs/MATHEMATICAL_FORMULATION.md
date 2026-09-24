# MATHEMATICAL FORMULATION

This document defines the optimization problem, the PSO algorithm, the integer mapping, and the separate final
evaluation. Notation is used consistently across all project documents.

---

## 1. Data and partitions

- Dataset: $\mathcal{D} = \{(\mathbf{a}_i, y_i)\}_{i=1}^{n}$, with feature vectors $\mathbf{a}_i \in \mathbb{R}^{d}$
  and labels $y_i \in \{0, \dots, C-1\}$.
- **Outer partition.** $\mathcal{D}$ is split into $K_{out} = 5$ disjoint, stratified folds
  $P_1, \dots, P_5$, using `StratifiedKFold(5, shuffle=True, random_state=42)`. For outer fold $k$:

$$\mathcal{D}^{(k)}_{test} = P_k, \qquad \mathcal{D}^{(k)}_{opt} = \mathcal{D} \setminus P_k .$$

- **Inner partition.** $\mathcal{D}^{(k)}_{opt}$ is split into $K_{in} = 5$ disjoint, stratified folds
  $Q^{(k)}_1, \dots, Q^{(k)}_5$, using `StratifiedKFold(5, shuffle=True, random_state=s_k)` with run seed $s_k = k$.
  **This inner partition is fixed for the entire optimization run of fold $k$.**

The **validation data** of this project is the collection of inner folds $Q^{(k)}_j$. The **held-out test data** is
$P_k$. By construction, $P_k \cap \mathcal{D}^{(k)}_{opt} = \varnothing$.

## 2. Decision variables

$$\theta = (\theta_1, \theta_2, \theta_3) = (n\_estimators,\ max\_depth,\ min\_samples\_split)$$

## 3. Search space and constraints

Lower and upper bounds:

$$l = (50,\ 2,\ 2), \qquad u = (200,\ 20,\ 10)$$

| Constraint | Meaning |
|---|---|
| $50 \le \theta_1 \le 200$ | number of trees |
| $2 \le \theta_2 \le 20$ | maximum tree depth |
| $2 \le \theta_3 \le 10$ | minimum samples to split a node; scikit-learn requires ≥ 2, so the bound is also a validity constraint |
| $\theta \in \mathbb{Z}^3$ | integer-valued |

Feasible set:

$$\Omega = \mathbb{Z}^3 \cap [l, u] = \{50,\dots,200\} \times \{2,\dots,20\} \times \{2,\dots,10\}, \qquad |\Omega| = 151 \cdot 19 \cdot 9 = 25{,}821 .$$

PSO operates in the **continuous relaxation** $B = [l, u] \subset \mathbb{R}^3$ (the search box).

## 4. Integer mapping (decode operator)

$$\mathrm{dec}: B \to \Omega, \qquad \mathrm{dec}(x) = \mathrm{clip}\big(\mathrm{rint}(x),\ l,\ u\big)$$

- $\mathrm{rint}$ rounds to the nearest integer, with ties going to the even integer (NumPy `np.rint`).
- $\mathrm{clip}$ applies component-wise. It only matters if a caller passes a point outside $B$, since PSO keeps
  $x \in B$.
- The swarm state ($x$, $v$, $p$, $g$) stays **continuous**. $\mathrm{dec}$ is applied only to evaluate.

*Why not round the state itself?* A particle whose velocity component is below 0.5 would be rounded back to the same
integer every iteration, so it would never move in that dimension. Keeping the state continuous lets small, repeated
velocity contributions accumulate.

## 5. The system being optimized

For $\theta \in \Omega$, seed $s$ and a training set $S$, let $h = \mathrm{RF}_{\theta, s}(S)$ be the Random Forest
pipeline trained on $S$:
- the imputer step, used only for Heart Disease, is fitted on $S$ only;
- `RandomForestClassifier(n_estimators=θ₁, max_depth=θ₂, min_samples_split=θ₃, random_state=s)`;
- all other parameters are scikit-learn defaults.

Accuracy of a classifier $h$ on a labelled set $S'$:

$$\mathrm{acc}(h; S') = \frac{1}{|S'|} \sum_{(\mathbf{a}, y) \in S'} \mathbb{1}[h(\mathbf{a}) = y]$$

## 6. Objective (fitness) function

For outer fold $k$, the fitness of a configuration is the mean inner-CV accuracy:

$$\boxed{\ F_k(\theta) = \frac{1}{5} \sum_{j=1}^{5} \mathrm{acc}\Big(\mathrm{RF}_{\theta, s_k}\big(\mathcal{D}^{(k)}_{opt} \setminus Q^{(k)}_j\big);\ Q^{(k)}_j\Big)\ }$$

The fitness of a continuous particle position is $f_k(x) = F_k(\mathrm{dec}(x))$.

**Optimization problem (for each outer fold $k$):**

$$\theta^{*}_k \in \arg\max_{\theta \in \Omega} F_k(\theta)$$

PSO returns an approximation $\hat\theta_k = \mathrm{dec}(g_k^{final})$.

**Alternative metric (configuration option).** With `fitness.metric = balanced_accuracy`, $\mathrm{acc}$ is replaced
by the balanced accuracy $\frac{1}{C}\sum_c \mathrm{recall}_c$. This is selected automatically for a dataset whose
largest-to-smallest class ratio exceeds 1.5. None of the three datasets is expected to trigger it (to verify in
Phase 2).

**Properties that matter for the design:**
1. **Test independence.** $F_k$ is a function of $\mathcal{D}^{(k)}_{opt}$ only. Changing any value in
   $\mathcal{D}^{(k)}_{test}$ leaves $F_k$, and therefore the whole PSO trajectory, unchanged. This is tested
   directly (TESTING_STRATEGY IT-06 and IT-07).
2. **Deterministic within a run.** The inner folds and seed $s_k$ are fixed, so $F_k(\theta)$ always returns the same
   value for the same $\theta$. That makes caching exact and pbest comparisons fair.
3. **Piecewise constant, with resolution $1/|\mathcal{D}^{(k)}_{opt}|$.** Each optimization sample is validated
   exactly once, so $F_k$ is a multiple of $1/|\mathcal{D}^{(k)}_{opt}|$ (strictly, a mean of per-fold fractions). On
   Iris, $|\mathcal{D}^{(k)}_{opt}| = 120$, so many configurations tie. $F_k$ has no gradient, which is why a
   derivative-free method is needed.
4. **Optimistic bias of the maximum.** $\max_\theta F_k(\theta)$ over many evaluations over-estimates the true
   generalization accuracy of $\hat\theta_k$ (the winner's curse). For that reason it is **never** reported as
   performance (see §10).

## 7. Particle Swarm Optimization

### 7.1 Definitions

| Symbol | Name | Definition |
|---|---|---|
| $N = 10$ | swarm size | number of particles |
| $x_i^t \in B$ | position | particle $i$'s continuous position at iteration $t$; its candidate configuration is $\mathrm{dec}(x_i^t)$ |
| $v_i^t \in \mathbb{R}^3$ | velocity | step applied to reach the next position |
| $f_i^t = f_k(x_i^t)$ | fitness | measured feedback for particle $i$ at iteration $t$ |
| $p_i^t$, $\phi_i^t$ | personal best | best position visited by particle $i$ up to $t$, and its fitness |
| $g^t$, $\gamma^t$ | global best | best personal best of the swarm up to $t$, and its fitness |
| $w = 0.7298$ | inertia weight | fraction of the previous velocity retained |
| $c_1 = 1.49618$ | cognitive coefficient | pull toward the particle's own best |
| $c_2 = 1.49618$ | social coefficient | pull toward the swarm's best |
| $r_1, r_2 \sim U[0,1)^3$ | random factors | drawn independently per particle, per dimension, per iteration |
| $v_{max} = 0.2(u - l) = (30,\ 3.6,\ 1.6)$ | velocity clamp | per-dimension maximum speed |
| $T = 20$ | maximum iterations | update iterations after the initial evaluation |

Iteration $t = 0$ is the evaluation of the initial swarm. Iterations $t = 1, \dots, T$ each evaluate the moved swarm.
Total evaluations per run: $N(T+1) = 210$.

### 7.2 Initialization ($t = 0$)

$$x_{i,d}^0 \sim U[l_d, u_d], \qquad v_{i,d}^0 \sim U\big[-0.1(u_d - l_d),\ 0.1(u_d - l_d)\big]$$

Evaluate $f_i^0$ for all $i$. Then set $p_i^0 = x_i^0$, $\phi_i^0 = f_i^0$, and $g^0 = p_{i^*}^0$ with
$i^* = \min \arg\max_i \phi_i^0$ (lowest index on ties).

### 7.3 Velocity update

For each particle $i$ and dimension $d$:

$$\tilde v_{i,d}^{t+1} = \underbrace{w\, v_{i,d}^{t}}_{\text{inertia}} + \underbrace{c_1 r_{1,i,d} \big(p_{i,d}^{t} - x_{i,d}^{t}\big)}_{\text{cognitive}} + \underbrace{c_2 r_{2,i,d} \big(g_d^{t} - x_{i,d}^{t}\big)}_{\text{social}}$$

$$v_{i,d}^{t+1} = \mathrm{clip}\big(\tilde v_{i,d}^{t+1},\ -v_{max,d},\ v_{max,d}\big)$$

### 7.4 Position update with absorbing boundary

$$\tilde x_{i,d}^{t+1} = x_{i,d}^{t} + v_{i,d}^{t+1}$$

$$
\big(x_{i,d}^{t+1},\ v_{i,d}^{t+1}\big) =
\begin{cases}
(l_d,\ 0) & \text{if } \tilde x_{i,d}^{t+1} < l_d \\
(u_d,\ 0) & \text{if } \tilde x_{i,d}^{t+1} > u_d \\
(\tilde x_{i,d}^{t+1},\ v_{i,d}^{t+1}) & \text{otherwise}
\end{cases}
$$

Zeroing the velocity keeps a particle from repeatedly pushing against a wall. The particle may still sit **at** the
bound, which matters if the optimum is there (e.g. $n\_estimators = 200$).

### 7.5 Evaluation and memory update (synchronous)

After **all** particles have moved, evaluate $f_i^{t+1} = F_k(\mathrm{dec}(x_i^{t+1}))$ for every $i$. Then:

$$
(p_i^{t+1}, \phi_i^{t+1}) =
\begin{cases}
(x_i^{t+1},\ f_i^{t+1}) & \text{if } f_i^{t+1} > \phi_i^{t} \quad \text{(strict improvement)} \\
(p_i^{t},\ \phi_i^{t}) & \text{otherwise}
\end{cases}
$$

$$
i^* = \min \arg\max_i \phi_i^{t+1}, \qquad
(g^{t+1}, \gamma^{t+1}) =
\begin{cases}
(p_{i^*}^{t+1},\ \phi_{i^*}^{t+1}) & \text{if } \phi_{i^*}^{t+1} > \gamma^{t} \\
(g^{t},\ \gamma^{t}) & \text{otherwise}
\end{cases}
$$

Consequence: $\gamma^t$ is **non-decreasing** in $t$. This is the convergence curve.

### 7.6 Stopping predicate

- **Default:** stop after iteration $T = 20$.
- **Optional patience rule** (disabled by default): stop at the first $t \ge P$ with
  $\gamma^{t} - \gamma^{t-P} < \mathrm{tol}$, where $P = 5$ and $\mathrm{tol} = 10^{-4}$.

Whichever applies, the run records `stop_reason` and the **convergence iteration**
$t_c = \max\{t : \gamma^t > \gamma^{t-1}\}$, the last iteration in which gbest improved (0 if it never did).

### 7.7 Algorithm (pseudocode)

```
input: space (l, u), objective f, N, T, w, c1, c2, v_max, rng
X ← U(l, u)                         (N×3)
V ← U(−0.1(u−l), 0.1(u−l))          (N×3)
F ← [f(dec(X_i)) for i]             ← feedback from the system
P ← X;  Pf ← F;  g ← P[argmax Pf];  gf ← max Pf
for t = 1 … T:
    R1, R2 ← rng.random((N,3)), rng.random((N,3))
    V ← clip(w·V + c1·R1·(P − X) + c2·R2·(g − X), −v_max, v_max)
    X̃ ← X + V
    out ← (X̃ < l) | (X̃ > u);   X ← clip(X̃, l, u);   V[out] ← 0
    F ← [f(dec(X_i)) for i]         ← new parameters sent to the system, new feedback returned
    improved ← F > Pf;   P[improved] ← X[improved];   Pf[improved] ← F[improved]
    i* ← argmax Pf;  if Pf[i*] > gf: g ← P[i*]; gf ← Pf[i*]
    record iteration summary; check optional patience
return dec(g), gf
```

## 8. The closed loop, stated formally

Let $\mathcal{H}^t = \{(x_i^s, f_i^s) : s \le t,\ i = 1..N\}$ be the history of proposals and feedback. PSO's next
proposal is

$$x^{t+1} = \Phi\big(x^t, v^t, p^t, g^t, r^t\big), \qquad \text{where } p^t, g^t \text{ are functions of } \mathcal{H}^t .$$

The next control action therefore **depends on the measured feedback**. This is what makes the loop closed.

**Contrast: random search (the open-loop comparator).** Configurations $\theta^{(1)}, \dots, \theta^{(M)}$ are drawn
i.i.d. from $\mathrm{Uniform}(\Omega)$, with $M = N(T+1) = 210$. $\theta^{(j+1)}$ is independent of
$F_k(\theta^{(1)}), \dots, F_k(\theta^{(j)})$; the feedback is used only to pick the winner at the end:

$$\hat\theta^{RS}_k = \theta^{(j^*)}, \qquad j^* = \min \arg\max_j F_k(\theta^{(j)}) .$$

Both methods get the same budget and the same $F_k$. Any systematic difference between them is attributable to the
feedback.

## 9. Worked example: one particle update (illustration only)

> The numbers below are **invented for illustration**. They are not experimental data. Arithmetic was checked
> numerically.

Assume particle $i$ at iteration $t$ has:

| | $n\_estimators$ | $max\_depth$ | $min\_samples\_split$ |
|---|---|---|---|
| position $x$ | 120.4 | 8.2 | 4.6 |
| decoded $\mathrm{dec}(x)$ | 120 | 8 | 5 |
| velocity $v$ | 5.0 | −1.0 | 0.5 |
| personal best $p$ | 130 | 10 | 3 |
| global best $g$ | 180 | 15 | 3 |
| $r_1$ | 0.5 | 0.2 | 0.9 |
| $r_2$ | 0.3 | 0.7 | 0.1 |

Velocity terms, with $w = 0.7298$ and $c_1 = c_2 = 1.49618$:

| Term | $d=1$ | $d=2$ | $d=3$ |
|---|---|---|---|
| inertia $w v$ | 3.6490 | −0.7298 | 0.3649 |
| cognitive $c_1 r_1 (p - x)$ | 7.1817 | 0.5386 | −2.1545 |
| social $c_2 r_2 (g - x)$ | 26.7517 | 7.1218 | −0.2394 |
| raw $\tilde v$ | 37.5824 | 6.9306 | −2.0290 |
| clamp $\pm v_{max}$ = ±(30, 3.6, 1.6) | **30.0** | **3.6** | **−1.6** |
| new position $x + v$ | 150.4 | 11.8 | 3.0 |
| inside bounds? | yes | yes | yes |
| **decoded next configuration** | **150** | **12** | **3** |

The particle moves from configuration (120, 8, 5) to (150, 12, 3), pulled mainly by the global best. The clamp limits
each step, so it cannot jump straight to the global best.

**Boundary illustration.** If a particle had $x_3 = 2.5$ and $v_3 = -1.6$, then $\tilde x_3 = 0.9 < l_3 = 2$. It is
set to $x_3 = 2$, $v_3 = 0$, and decodes to $min\_samples\_split = 2$.

## 10. Final evaluation (separate from the optimization objective)

After the optimization of fold $k$ has **terminated** and $\hat\theta_k$ is fixed:

$$\hat h_k = \mathrm{RF}_{\hat\theta_k,\ s_k}\big(\mathcal{D}^{(k)}_{opt}\big) \qquad \text{(refit on the whole optimization portion)}$$

$$M_k = m\big(\hat h_k;\ \mathcal{D}^{(k)}_{test}\big)$$

Here $m$ is each test metric: accuracy, balanced accuracy, macro precision, macro recall, macro F1 and the confusion
matrix; for Heart Disease, also the positive-class precision, recall and F1.

- $\mathcal{D}^{(k)}_{test}$ appears **only** here. It appears nowhere in $F_k$, $\arg\max$, $p$, $g$, $v$, $x$ or the
  stopping predicate.
- $M_k$ is computed **once** per method per fold, and nothing computed from it flows back to any optimizer.
- The same formula is applied to the baseline, where $\theta_0 = (100, \text{None}, 2)$. Here $\text{None}$ means
  unlimited depth, so $\theta_0 \notin \Omega$. The baseline is scikit-learn's default, not a point in the search
  space. The same formula also applies to random search, with $\hat\theta^{RS}_k$.

**Aggregation over outer folds (descriptive):**
- **Mean and standard deviation:** $\bar M = \frac{1}{5}\sum_k M_k$ and $\mathrm{sd}(M_k)$.
- **Paired difference against the baseline:** $\Delta_k = M_k^{PSO} - M_k^{base}$, plus win/tie/loss counts.
- **Pooled predictions:** since the $P_k$ partition $\mathcal{D}$, every sample receives exactly one test prediction
  per method. These give a single pooled confusion matrix.

**Deployment configuration.** A final PSO run on all of $\mathcal{D}$ (seed 5, same inner CV procedure) gives
$\hat\theta_{deploy}$. Its expected performance is **estimated by $\bar M^{PSO}$**. The deployment run has no test
data, so it produces no test score of its own.

## 11. Computational budget

Let $t_{eval}$ be the wall time of one fitness evaluation (5 inner fits run in parallel). Per dataset:

$$\text{evaluations} = \underbrace{5 \times (210_{PSO} + 210_{RS})}_{\text{outer folds}} + \underbrace{210}_{\text{deployment}} = 2{,}310, \qquad \text{wall time} \lesssim 2{,}310 \cdot t_{eval}$$

Measured worst-case $t_{eval}$ (200 trees, depth 20) is 0.24 s for Iris and 0.52 s for Digits. That bounds the
runtime at about 9 min for Iris and 20 min for Digits. Heart is expected to be similar to Iris (to verify). The
actual time will be lower, because of cache hits and cheaper configurations. The baseline and final refits add a
negligible 5 + 1 fits per fold and method.
