"""Swarm replay: step through a saved PSO run of the real experiment, iteration by iteration (files only)."""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from pso_rf.app import charts, components
from pso_rf.app.data import DATASET_LABEL, list_experiments

REPO = Path(__file__).resolve().parents[3]


def render() -> None:
    st.markdown(
        components.hero(
            "Swarm replay",
            "Replay a saved PSO run: where every particle was, how it moved (velocity), and how gbest evolved.",
        ),
        unsafe_allow_html=True,
    )
    experiments = [e for e in list_experiments(REPO / "results") if e.datasets]
    if not experiments:
        st.warning("No saved experiment to replay yet.")
        return
    with st.sidebar:
        st.subheader("Replay")
        exp_id = st.selectbox("Results directory", [e.exp_id for e in experiments], key="replay_exp")
        exp = next(e for e in experiments if e.exp_id == exp_id)
        labels = {DATASET_LABEL.get(d, d): d for d in exp.datasets}
        dataset = labels[st.selectbox("Dataset", list(labels), key="replay_ds")]
        runs = {f"outer fold {k}": k for k in exp.folds_of(dataset)}
        runs["deployment (all data)"] = None
        fold = runs[st.selectbox("Run", list(runs), key="replay_fold")]
    events, its = exp.evaluations(dataset, fold, "pso"), exp.iterations(dataset, fold, "pso")
    if events.empty or its.empty:
        st.info("This run has no PSO trace (yet).")
        return
    last = int(its.iteration.max())
    if "replay_t" not in st.session_state or st.session_state.get("replay_key") != (exp_id, dataset, fold):
        st.session_state.replay_t, st.session_state.replay_key = 0, (exp_id, dataset, fold)

    c1, c2, c3 = st.columns([1, 1, 4])
    play = c1.button("▶ Play", use_container_width=True)
    if c2.button("⟲ Reset", use_container_width=True):
        st.session_state.replay_t = 0
    t = c3.slider("Iteration", 0, last, key="replay_t")

    loop_slot, kpi_slot = st.empty(), st.empty()
    left, right = st.columns([1.3, 1])
    swarm_slot, paths_slot = left.empty(), right.empty()
    table_slot = st.empty()

    def draw(step: int) -> None:
        row = its[its.iteration == step].iloc[0]
        gbest = {k: int(row[f"gbest_{k}"]) for k in charts.HYPERPARAMETERS}
        loop_slot.markdown(
            components.loop_diagram(
                "update",
                iteration=step,
                total_iterations=last,
                gbest=gbest,
                gbest_fitness=float(row.gbest_fitness),
            ),
            unsafe_allow_html=True,
        )
        kpi_slot.markdown(
            components.kpis(
                [
                    (
                        "iteration",
                        f"{step} / {last}",
                        "improved" if row.gbest_improved else "no gbest change",
                    ),
                    ("gbest fitness", f"{row.gbest_fitness:.4f}", "validation accuracy"),
                    ("gbest config", components.config_text(gbest), "(n, depth, split)"),
                    ("swarm mean", f"{row.mean_fitness:.4f}", f"diversity {row.diversity:.2f}"),
                    (
                        "cache hits",
                        f"{int(row.n_cache_hits)}",
                        f"{int(row.cumulative_unique_fits)} unique fits so far",
                    ),
                ]
            ),
            unsafe_allow_html=True,
        )
        current = events[events.iteration == step].to_dict("records")
        history = events[events.iteration < step].to_dict("records")
        swarm_slot.plotly_chart(
            charts.swarm_3d(current, history, gbest, title=f"Swarm at iteration {step}"),
            use_container_width=True,
        )
        fig = charts.hyperparameter_paths(its, "gbest hyperparameters over the run")
        fig.add_vline(x=step, line={"color": charts.MUTED, "dash": "dot"})
        paths_slot.plotly_chart(fig, use_container_width=True)
        cols = ["particle_id", *charts.HYPERPARAMETERS, "fitness", "pbest_fitness", "cache_hit"]
        table_slot.dataframe(
            events[events.iteration == step][cols], hide_index=True, use_container_width=True
        )

    if play:
        for step in range(t, last + 1):
            draw(step)
            time.sleep(0.7)
    else:
        draw(t)
    st.caption(
        f"Source: {exp.run_dir(dataset, fold, 'pso').relative_to(REPO).as_posix()}/"
        "{evaluations,iterations}.csv"
    )
