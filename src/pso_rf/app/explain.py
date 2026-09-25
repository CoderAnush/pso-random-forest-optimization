"""How it works: the closed-loop mapping, the PSO equations, and two live proofs (test isolation, feedback)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from pso_rf.app import charts, components
from pso_rf.evaluation import HeldOutTestSet, OptimizationPhase, TestSetAccessError
from pso_rf.optimization import Callback, IntParam, PSOConfig, PSOOptimizer, SearchSpace

SPACE = SearchSpace(
    [IntParam("n_estimators", 50, 200), IntParam("max_depth", 2, 20), IntParam("min_samples_split", 2, 10)]
)
OPTIMUM = np.array([140.0, 9.0, 5.0])


def render() -> None:
    st.markdown(
        components.hero(
            "How the closed loop works",
            "Where optimization is applied, what is fed back, and why the test data can never leak into the search.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(components.loop_diagram("idle"), unsafe_allow_html=True)

    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown("#### Control-system view")
        st.dataframe(
            pd.DataFrame(
                [
                    ("Plant (system)", "Random Forest trained + validated on the optimization data"),
                    ("Controller", "Particle Swarm Optimization"),
                    (
                        "Control action",
                        "a particle's decoded hyperparameters (n_estimators, max_depth, min_samples_split)",
                    ),
                    ("Measurement / feedback", "mean 5-fold CV accuracy = fitness, returned to PSO"),
                    ("Adaptation", "pbest / gbest memory, then velocity and position update"),
                    ("One cycle / one step", "one particle evaluation / one swarm iteration"),
                    ("Open-loop contrast", "random search: same plant and budget, but no feedback"),
                ],
                columns=["concept", "in this project"],
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Honest framing: a static plant with no setpoint (extremum seeking), not a real-time controller."
        )
    with c2:
        st.markdown("#### PSO update (from scratch)")
        st.latex(r"v \leftarrow \mathrm{clip}\big(w v + c_1 r_1 (p - x) + c_2 r_2 (g - x),\ \pm v_{max}\big)")
        st.latex(r"x \leftarrow \mathrm{clip}(x + v,\ l,\ u),\quad v_d \leftarrow 0 \text{ where clipped}")
        st.latex(
            r"\theta = \mathrm{clip}(\mathrm{rint}(x)) \in \Omega,\quad |\Omega| = 151 \cdot 19 \cdot 9 = 25{,}821"
        )
        st.markdown(
            "$w = 0.7298$, $c_1 = c_2 = 1.49618$, $v_{max} = 0.2 \\times$ range, "
            "N = 10 particles, T = 20 iterations → 210 evaluations."
        )

    st.divider()
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("#### Proof 1 · the test fold is sealed during optimization")
        st.caption(
            "Try to read held-out data while an optimization phase is open, exactly as a buggy search would."
        )
        if st.button("Try to peek at the test fold", use_container_width=True):
            test = HeldOutTestSet(np.zeros((3, 2)), np.array([0, 1, 0]), np.array([0, 1, 2]))
            try:
                with OptimizationPhase():
                    test.reveal()
                st.error("The test fold was readable — this must never happen.")
            except TestSetAccessError as exc:
                st.success(f"Blocked: `TestSetAccessError` — {exc}")
            X, _ = test.reveal()
            st.info(f"After the phase closes the final evaluation may read it once ({len(X)} rows here).")
        st.caption(
            "In the test suite this is backed by a row-hash spy on every fit/predict and by label/feature "
            "invariance (IT-05…IT-07)."
        )
    with p2:
        st.markdown("#### Proof 2 · the search depends on the feedback")
        st.caption(
            "A stub system with a known optimum, so this runs instantly: same seed, three kinds of feedback."
        )
        if st.button("Run the feedback ablation", use_container_width=True):
            st.plotly_chart(
                charts.ablation(_ablation(), "gbest distance to the optimum"), use_container_width=True
            )
            st.caption(
                "Only true feedback steers the swarm to the optimum; constant feedback never moves gbest, "
                "and shuffled feedback steers it somewhere wrong."
            )


class _Distance(Callback):
    def __init__(self) -> None:
        self.values: list[float] = []

    def on_iteration_end(self, summary) -> None:
        theta = np.array([summary.gbest_config[n] for n in SPACE.names], dtype=float)
        self.values.append(float(np.linalg.norm((theta - OPTIMUM) / SPACE.range_)))


def _true(config: dict[str, int]) -> float:
    theta = np.array([config[n] for n in SPACE.names], dtype=float)
    return -float(np.sum(((theta - OPTIMUM) / SPACE.range_) ** 2))


def _ablation() -> dict[str, list[float]]:
    def reflected(config: dict[str, int]) -> float:
        return _true({p.name: p.low + p.high - config[p.name] for p in SPACE.params})

    curves = {}
    for name, objective in (
        ("true feedback", _true),
        ("constant feedback", lambda c: 0.0),
        ("shuffled feedback", reflected),
    ):
        tracker = _Distance()
        PSOOptimizer(SPACE, objective, PSOConfig(), np.random.default_rng(0)).run([tracker])
        curves[name] = tracker.values
    return curves
