"""Live closed-loop lab: run the real PSO ⇄ Random Forest loop on one outer fold and watch it close.

Every run goes through ``pso_rf.experiments.runner.run_fold``, the same code path as the experiment, with an
observer callback that redraws the page after each evaluation. The held-out test fold stays sealed during each
search; test metrics appear only after the loop has ended. Each live run is saved under ``results/live/``.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from pso_rf.app import charts, components
from pso_rf.app.data import DATASET_LABEL, DATASET_TASK
from pso_rf.datasets import DatasetBundle, load_dataset
from pso_rf.datasets.audit import audit
from pso_rf.evaluation import FitnessEvaluator, OptimizationPhase, outer_folds
from pso_rf.evaluation.final import final_evaluate
from pso_rf.experiments.config import ExperimentConfig, load_config
from pso_rf.experiments.runner import positive_label_for, run_fold
from pso_rf.optimization import Callback, EvaluationEvent, IterationSummary

REPO = Path(__file__).resolve().parents[3]
T_EVAL_MID = {"iris": 0.13, "digits": 0.33, "heart_cleveland": 0.14}  # Phase 6 benchmark (s per evaluation)


@st.cache_resource(show_spinner=False)
def _bundle(name: str) -> DatasetBundle:
    return load_dataset(name, REPO / "data")


def _config(n_particles: int, max_iter: int, fold: int, seed: int) -> ExperimentConfig:
    seeds = [0, 1, 2, 3, 4]
    seeds[fold] = seed
    overrides = [
        f"pso.n_particles={n_particles}",
        f"pso.max_iter={max_iter}",
        f"split.run_seeds=[{', '.join(map(str, seeds))}]",
    ]
    return load_config([REPO / "configs" / "default.yaml"], overrides)


class LiveView(Callback):
    """Observer of optimizer events: stores them and redraws the loop diagram, KPIs and charts."""

    def __init__(
        self,
        slots: dict[str, Any],
        method: str,
        total_iterations: int,
        budget: int,
        inner_folds: int,
        manual: dict[str, int] | None,
        rs_reference: list[float] | None = None,
    ) -> None:
        self.slots, self.method = slots, method
        self.total_iterations, self.budget, self.inner_folds = total_iterations, budget, inner_folds
        self.manual = manual
        self.events: list[dict[str, Any]] = []
        self.iterations: list[dict[str, Any]] = []
        self.best_so_far: list[float] = []
        self.best_config: dict[str, int] | None = None
        self.rs_reference = rs_reference

    def on_evaluation(self, event: EvaluationEvent) -> None:
        row = {
            "eval_index": event.eval_index,
            "iteration": event.iteration,
            "particle_id": event.particle_id,
            **event.config,
            "fitness": event.fitness,
            "cache_hit": bool(event.info.get("cache_hit", False)) if event.info else False,
        }
        if event.position is not None:
            row.update({f"pos_{k}": v for k, v in zip(charts.HYPERPARAMETERS, event.position, strict=True)})
            row.update({f"vel_{k}": v for k, v in zip(charts.HYPERPARAMETERS, event.velocity, strict=True)})
        self.events.append(row)
        if not self.best_so_far or event.fitness > self.best_so_far[-1]:
            self.best_config = dict(event.config)
        self.best_so_far.append(event.best_so_far_fitness)
        open_loop = self.method == "random_search"
        self.slots["loop"].markdown(
            components.loop_diagram(
                "evaluate",
                iteration=event.iteration,
                total_iterations=self.total_iterations,
                particle=event.particle_id if not open_loop else event.eval_index + 1,
                config=event.config,
                fitness=event.fitness,
                gbest=self.best_config,
                gbest_fitness=event.best_so_far_fitness,
                inner_folds=self.inner_folds,
                open_loop=open_loop,
            ),
            unsafe_allow_html=True,
        )
        done = len(self.events)
        self.slots["progress"].progress(
            min(done / self.budget, 1.0),
            text=f"{charts.METHOD_LABEL[self.method]}: {done} / {self.budget} configurations evaluated",
        )
        if open_loop and (done % 5 == 0 or done == self.budget):
            self._draw_race()

    def on_iteration_end(self, summary: IterationSummary) -> None:
        self.iterations.append(
            {**asdict(summary), **{f"gbest_{k}": summary.gbest_config[k] for k in charts.HYPERPARAMETERS}}
        )
        this_iteration = [e for e in self.events if e["iteration"] == summary.iteration]
        best = max(this_iteration, key=lambda e: e["fitness"]) if this_iteration else None
        self.slots["loop"].markdown(
            components.loop_diagram(
                "update",
                iteration=summary.iteration,
                total_iterations=self.total_iterations,
                particle=None if best is None else best["particle_id"],
                config=None if best is None else {k: best[k] for k in charts.HYPERPARAMETERS},
                fitness=None if best is None else best["fitness"],
                gbest=summary.gbest_config,
                gbest_fitness=summary.gbest_fitness,
                inner_folds=self.inner_folds,
            ),
            unsafe_allow_html=True,
        )
        unique = sum(not e["cache_hit"] for e in self.events)
        self.slots["kpis"].markdown(
            components.kpis(
                [
                    ("iteration", f"{summary.iteration} / {self.total_iterations}", "0 = initial swarm"),
                    ("gbest fitness", f"{summary.gbest_fitness:.4f}", "validation accuracy, inner CV"),
                    ("gbest config", components.config_text(summary.gbest_config), "(n, depth, split)"),
                    ("swarm mean", f"{summary.mean_fitness:.4f}", f"diversity {summary.diversity:.2f}"),
                    ("evaluations", f"{len(self.events)}", f"{unique} unique RF fits (rest cached)"),
                ]
            ),
            unsafe_allow_html=True,
        )
        self.slots["convergence"].plotly_chart(
            charts.convergence(self.iterations, "Convergence (live)"), use_container_width=True
        )
        current = [e for e in self.events if e["iteration"] == summary.iteration]
        history = [e for e in self.events if e["iteration"] < summary.iteration]
        self.slots["swarm"].plotly_chart(
            charts.swarm_3d(
                current, history, summary.gbest_config, self.manual, title="Swarm in the search space (live)"
            ),
            use_container_width=True,
        )

    def _draw_race(self) -> None:
        curves = (
            {"pso": self.rs_reference, "random_search": self.best_so_far}
            if self.rs_reference
            else {"random_search": self.best_so_far}
        )
        self.slots["race"].plotly_chart(
            charts.anytime(curves, "Race at equal budget: closed loop vs open loop"), use_container_width=True
        )


def _score_manual(
    cfg: ExperimentConfig, bundle: DatasetBundle, fold: Any, config: dict[str, int], metric: str
) -> dict[str, Any]:
    """Your hand-picked configuration, scored exactly like a candidate (inner CV) and then on the test fold."""
    settings = cfg.for_dataset(bundle.name)
    seed = cfg.split.run_seeds[fold.fold_index]
    evaluator = FitnessEvaluator(
        fold.opt,
        cfg.split.inner_folds,
        seed,
        metric,
        settings.preprocessing,
        n_jobs_folds=settings.fitness.n_jobs_folds,
    )
    with OptimizationPhase():
        scored = evaluator(config)
    final = final_evaluate(
        config,
        fold.opt,
        fold.test,
        seed,
        settings.preprocessing,
        labels=list(range(len(bundle.class_names))),
        positive_label=positive_label_for(bundle),
    )
    return {
        "method": "manual",
        "best_hyperparameters": config,
        "best_validation_fitness": scored.fitness,
        "test_metrics": final.metrics,
        "n_evaluations": 1,
        "n_unique_fits": 1,
    }


def render() -> None:
    st.markdown(
        components.hero(
            "Live closed-loop lab",
            "PSO proposes Random Forest hyperparameters → the forest is trained → validation accuracy is fed back → "
            "PSO moves the swarm. Watch the loop close, then open the sealed test fold once.",
        ),
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Live run settings")
        label = st.selectbox("Dataset", list(DATASET_LABEL.values()), index=2, key="live_ds")
        dataset = next(k for k, v in DATASET_LABEL.items() if v == label)
        fold = st.select_slider("Outer fold (its test part stays sealed)", options=[0, 1, 2, 3, 4], value=0)
        n_particles = st.slider("Particles (N)", 4, 15, 10)
        max_iter = st.slider("Iterations (T)", 2, 20, 8)
        seed = st.number_input(
            "Run seed", 0, 10_000, value=fold, help="Drives PSO, inner folds and RF seeds."
        )
        race = st.toggle("Race against random search (same budget)", value=True)
        with st.expander("Manual tuning challenge"):
            st.caption("Pick a configuration by hand; it is scored exactly like a PSO candidate.")
            manual_on = st.checkbox("Include my manual pick", value=True)
            manual = {
                "n_estimators": st.slider("n_estimators", 50, 200, 60),
                "max_depth": st.slider("max_depth", 2, 20, 3),
                "min_samples_split": st.slider("min_samples_split", 2, 10, 10),
            }
        budget = n_particles * (max_iter + 1)
        estimate = budget * T_EVAL_MID[dataset] * (2 if race else 1) + 2
        st.caption(f"Budget {budget} evaluations per method · estimated ≈ {estimate:.0f} s")
        start = st.button("▶ Run the closed loop", type="primary", use_container_width=True)

    bundle = _bundle(dataset)
    facts = bundle.meta.get("audit", {})
    folds = outer_folds(bundle, 5, 42)
    st.markdown(
        components.kpis(
            [
                ("dataset", DATASET_LABEL[dataset], DATASET_TASK[dataset]),
                ("samples", f"{facts.get('n_samples')}", f"{facts.get('n_features')} features"),
                ("optimization part", f"{len(folds[fold].opt.y)}", "inner 5-fold CV → fitness"),
                ("held-out test fold", f"{len(folds[fold].test)}", "sealed until the end"),
                ("search space", "25,821", "151 × 19 × 9 integer configs"),
            ]
        ),
        unsafe_allow_html=True,
    )

    left, right = st.columns([3.2, 1])
    slots = {"loop": left.empty(), "seal": right.empty()}
    slots["progress"] = st.empty()
    slots["kpis"] = st.empty()
    c1, c2 = st.columns(2)
    slots["convergence"], slots["swarm"] = c1.empty(), c2.empty()
    slots["race"] = st.empty()
    slots["final"] = st.container()

    snapshot = st.session_state.get("live_snapshot")
    if not start:
        if snapshot and snapshot["dataset"] == dataset:
            _draw_snapshot(snapshot, slots)
        else:
            slots["loop"].markdown(components.loop_diagram("idle", inner_folds=5), unsafe_allow_html=True)
            slots["seal"].markdown(components.seal_card("locked"), unsafe_allow_html=True)
            st.info("Choose settings in the sidebar and press **Run the closed loop**.")
        return

    cfg = _config(n_particles, max_iter, fold, int(seed))
    settings = cfg.for_dataset(dataset)
    metric = audit(bundle, settings.fitness.class_ratio_gate, settings.fitness.metric)["fitness_metric"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = REPO / "results" / "live" / f"{stamp}_{dataset}_fold{fold}"
    exp_id = f"live-{stamp}"
    slots["seal"].markdown(components.seal_card("locked"), unsafe_allow_html=True)
    started = time.perf_counter()

    pso_view = LiveView(slots, "pso", max_iter, budget, cfg.split.inner_folds, manual if manual_on else None)
    records = {
        "pso": run_fold(cfg, bundle, folds[fold], "pso", metric, out / "pso", exp_id, callbacks=[pso_view])
    }
    rs_curve: list[float] = []
    if race:
        rs_view = LiveView(
            slots,
            "random_search",
            max_iter,
            budget,
            cfg.split.inner_folds,
            None,
            rs_reference=pso_view.best_so_far,
        )
        records["random_search"] = run_fold(
            cfg,
            bundle,
            folds[fold],
            "random_search",
            metric,
            out / "random_search",
            exp_id,
            callbacks=[rs_view],
        )
        rs_curve = rs_view.best_so_far
    slots["progress"].progress(1.0, text="Scoring the default RF and opening the test fold…")
    records["baseline"] = run_fold(cfg, bundle, folds[fold], "baseline", metric, out / "baseline", exp_id)
    if manual_on:
        records["manual"] = _score_manual(cfg, bundle, folds[fold], manual, metric)
    snapshot = {
        "dataset": dataset,
        "fold": fold,
        "out": str(out),
        "records": records,
        "iterations": pso_view.iterations,
        "events": pso_view.events,
        "pso_curve": pso_view.best_so_far,
        "rs_curve": rs_curve,
        "manual": manual if manual_on else None,
        "n_test": len(folds[fold].test),
        "elapsed": time.perf_counter() - started,
        "max_iter": max_iter,
        "budget": budget,
    }
    st.session_state["live_snapshot"] = snapshot
    slots["progress"].empty()
    _draw_snapshot(snapshot, slots)


def _draw_snapshot(snap: dict[str, Any], slots: dict[str, Any]) -> None:
    """The finished run: final loop state, unsealed test fold and the comparison of all methods."""
    records = snap["records"]
    pso = records["pso"]
    last = snap["iterations"][-1] if snap["iterations"] else None
    slots["loop"].markdown(
        components.loop_diagram(
            "update",
            iteration=last["iteration"] if last else None,
            total_iterations=snap["max_iter"],
            gbest=pso["best_hyperparameters"],
            gbest_fitness=pso["best_validation_fitness"],
        ),
        unsafe_allow_html=True,
    )
    slots["seal"].markdown(
        components.seal_card(
            "open", f"{snap['n_test']} samples, each method scored once after its own search."
        ),
        unsafe_allow_html=True,
    )
    if snap["iterations"]:
        slots["convergence"].plotly_chart(
            charts.convergence(snap["iterations"], "Convergence"), use_container_width=True
        )
        current = [e for e in snap["events"] if e["iteration"] == last["iteration"]]
        history = [e for e in snap["events"] if e["iteration"] < last["iteration"]]
        slots["swarm"].plotly_chart(
            charts.swarm_3d(
                current, history, pso["best_hyperparameters"], snap["manual"], title="Final swarm"
            ),
            use_container_width=True,
        )
    if snap["rs_curve"]:
        slots["race"].plotly_chart(
            charts.anytime(
                {"pso": snap["pso_curve"], "random_search": snap["rs_curve"]},
                "Race at equal budget: closed loop vs open loop",
            ),
            use_container_width=True,
        )

    with slots["final"]:
        st.subheader("After the loop: the sealed test fold is opened")
        order = [m for m in ("baseline", "manual", "random_search", "pso") if m in records]
        rows = []
        for method in order:
            record = records[method]
            test = record["test_metrics"]
            rows.append(
                {
                    "method": charts.METHOD_LABEL[method],
                    "configuration (n, depth, split)": components.config_text(record["best_hyperparameters"]),
                    "validation accuracy (inner CV)": record["best_validation_fitness"],
                    "test accuracy": test["accuracy"],
                    "test macro F1": test["f1_macro"],
                    "configs evaluated": record["n_evaluations"],
                    "unique RF fits": record["n_unique_fits"],
                }
            )
        table = pd.DataFrame(rows)
        st.dataframe(
            table.style.format(
                {
                    "validation accuracy (inner CV)": "{:.4f}",
                    "test accuracy": "{:.4f}",
                    "test macro F1": "{:.4f}",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
        c1, c2 = st.columns([1.2, 1])
        c1.plotly_chart(
            charts.method_bars(
                [{"method": m, "value": records[m]["test_metrics"]["accuracy"]} for m in order],
                "value",
                "Test accuracy on this fold's held-out data",
            ),
            use_container_width=True,
        )
        one = 1 / snap["n_test"]
        c2.markdown(
            f"**Reading this honestly**\n\n"
            f"- One test sample on this fold is worth **{one:.3f}** accuracy; smaller gaps are noise.\n"
            f"- Validation accuracy is the maximum over many noisy estimates, so it is *optimistically biased*; "
            f"only the test column is performance.\n"
            f"- A single fold is a demonstration. The experiment dashboard averages all 5 outer folds.\n"
            f"- Run saved to `{Path(snap['out']).relative_to(REPO).as_posix()}` "
            f"({snap['elapsed']:.0f} s)."
        )
        with st.expander("Every evaluation of the PSO run (the trace of the closed loop)"):
            events = pd.DataFrame(snap["events"])
            if not events.empty:
                cols = ["iteration", "particle_id", *charts.HYPERPARAMETERS, "fitness", "cache_hit"]
                st.dataframe(events[cols], hide_index=True, use_container_width=True, height=320)
        if not math.isfinite(pso["best_validation_fitness"]):
            st.error("Every PSO evaluation failed.")
