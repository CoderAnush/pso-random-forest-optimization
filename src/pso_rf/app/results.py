"""Experiment dashboard: the full 3 datasets × 5 outer folds experiment, read from ``results/<exp_id>/``.

Every number on this page comes from a saved file. The verification badge runs the same audit as
``python -m pso_rf verify`` (completeness, test isolation in the run log, summary recomputation).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from pso_rf.app import charts, components
from pso_rf.app.data import DATASET_LABEL, Experiment, list_experiments
from pso_rf.experiments.verify import verify_results

REPO = Path(__file__).resolve().parents[3]


@st.cache_data(show_spinner=False)
def _verify(root: str, finished_at: str | None) -> list[str]:
    return verify_results(Path(root))


def _pick_experiment() -> Experiment | None:
    experiments = list_experiments(REPO / "results")
    if not experiments:
        st.warning("No saved experiment yet. Run `python -m pso_rf run --config configs/default.yaml`.")
        return None
    complete = [e for e in experiments if e.complete] or experiments
    with st.sidebar:
        st.subheader("Experiment")
        choice = st.selectbox(
            "Results directory",
            [e.exp_id for e in experiments],
            index=[e.exp_id for e in experiments].index(complete[0].exp_id),
        )
    return next(e for e in experiments if e.exp_id == choice)


def render() -> None:
    st.markdown(
        components.hero(
            "Experiment results",
            "3 datasets × 5 outer folds × {default RF, random search, PSO}: every sample tested exactly once, "
            "each method scored once per fold after its own search.",
        ),
        unsafe_allow_html=True,
    )
    exp = _pick_experiment()
    if exp is None:
        return
    summary, folds = exp.summary(), exp.summary_folds()
    if not exp.complete or summary.empty:
        st.info(
            f"`{exp.exp_id}` is not complete yet (status: {exp.manifest.get('status')}). "
            "Results appear here when the run finishes."
        )
        return

    _provenance(exp)
    _headline(exp, summary, folds)

    st.divider()
    tabs = st.tabs([DATASET_LABEL.get(d, d) for d in exp.datasets])
    for tab, dataset in zip(tabs, exp.datasets, strict=True):
        with tab:
            _dataset_view(exp, dataset, summary, folds)


def _provenance(exp: Experiment) -> None:
    m = exp.manifest
    problems = _verify(str(exp.root), m.get("finished_at"))
    badge = (
        components.pill("verified · all checks pass", "ok")
        if not problems
        else components.pill(f"{len(problems)} verification problem(s)", "bad")
    )
    dirty = m.get("git_dirty")
    tree = (
        components.pill("clean git tree", "ok")
        if dirty is False
        else components.pill("uncommitted changes", "bad")
    )
    st.markdown(
        f"{badge} {tree} {components.pill('config ' + m.get('config_hash', '')[:10], 'info')} "
        f"{components.pill('commit ' + (m.get('git_commit') or '?')[:8], 'info')}",
        unsafe_allow_html=True,
    )
    if problems:
        with st.expander("Verification problems"):
            for problem in problems:
                st.write("•", problem)
    st.caption(
        f"`{exp.exp_id}` · started {m.get('created_at')} · finished {m.get('finished_at')} · "
        f"Python {m.get('python_version')}, scikit-learn {m.get('packages', {}).get('scikit-learn')} · "
        f"{m.get('cpu_count')} CPUs"
    )


def _headline(exp: Experiment, summary: pd.DataFrame, folds: pd.DataFrame) -> None:
    table = summary.assign(dataset_label=summary.dataset.map(DATASET_LABEL), mean=summary.test_accuracy_mean)
    st.plotly_chart(
        charts.grouped_bars(table, "Held-out test accuracy (mean over 5 outer folds)"),
        use_container_width=True,
    )
    c1, c2 = st.columns([1.15, 1])
    deltas = folds[folds.method != "baseline"].assign(
        dataset_label=lambda f: f.dataset.map(DATASET_LABEL), delta=lambda f: f.delta_accuracy_vs_baseline
    )
    c1.plotly_chart(
        charts.delta_dots(deltas, "Paired per-fold difference to the default RF"), use_container_width=True
    )
    with c2:
        st.markdown("**Summary table** (test metrics are performance; validation is optimistically biased)")
        view = pd.DataFrame(
            {
                "dataset": summary.dataset.map(DATASET_LABEL),
                "method": summary.method.map(charts.METHOD_LABEL),
                "test acc (mean ± sd)": [
                    f"{m:.4f} ± {s:.4f}"
                    for m, s in zip(summary.test_accuracy_mean, summary.test_accuracy_std, strict=True)
                ],
                "Δ vs default": [
                    ("–" if pd.isna(d) or meth == "baseline" else f"{d:+.4f}")
                    for d, meth in zip(summary.delta_accuracy_mean, summary.method, strict=True)
                ],
                "W/T/L": [
                    ("–" if pd.isna(w) else f"{int(w)}/{int(t)}/{int(lo)}")
                    for w, t, lo in zip(summary.wins, summary.ties, summary.losses, strict=True)
                ],
                "unique fits": summary.n_unique_fits_mean.round(0).astype(int),
            }
        )
        st.dataframe(view, hide_index=True, use_container_width=True, height=360)
    st.caption(
        "No significance tests: with 5 paired folds the smallest possible two-sided Wilcoxon p-value is "
        "0.0625, so results are reported descriptively (ADR-020). Source: summary.csv, summary_folds.csv."
    )


def _dataset_view(exp: Experiment, dataset: str, summary: pd.DataFrame, folds: pd.DataFrame) -> None:
    audit = exp.audit(dataset) or {}
    rows = summary[summary.dataset == dataset].set_index("method")
    items = [
        (
            "samples",
            f"{audit.get('n_samples')}",
            f"{audit.get('n_features')} features, {audit.get('n_classes')} classes",
        )
    ]
    for method in ("baseline", "random_search", "pso"):
        if method in rows.index:
            r = rows.loc[method]
            items.append(
                (
                    charts.METHOD_LABEL[method],
                    f"{r.test_accuracy_mean:.4f}",
                    f"± {r.test_accuracy_std:.4f} test accuracy",
                )
            )
    st.markdown(components.kpis(items), unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    pso_ev, rs_ev = exp.all_evaluations(dataset, "pso"), exp.all_evaluations(dataset, "random_search")
    curves, bands = {}, {}
    for method, ev in (("random_search", rs_ev), ("pso", pso_ev)):
        if ev.empty:
            continue
        grid = ev.pivot_table(index="eval_index", columns="outer_fold", values="best_so_far_fitness")
        curves[method] = grid.mean(axis=1).to_numpy()
        bands[method] = (grid.min(axis=1).to_numpy(), grid.max(axis=1).to_numpy())
    if curves:
        c1.plotly_chart(
            charts.anytime(curves, "Value of feedback: best-so-far, mean of folds (band = min–max)", bands),
            use_container_width=True,
        )
    its = exp.all_iterations(dataset)
    if not its.empty:
        mean = (
            its.groupby("iteration")
            .agg(
                gbest_fitness=("gbest_fitness", "mean"),
                mean_fitness=("mean_fitness", "mean"),
                min_fitness=("gbest_fitness", "min"),
                max_fitness=("gbest_fitness", "max"),
                gbest_n_estimators=("gbest_n_estimators", "median"),
                gbest_max_depth=("gbest_max_depth", "median"),
                gbest_min_samples_split=("gbest_min_samples_split", "median"),
            )
            .reset_index()
        )
        c2.plotly_chart(
            charts.convergence(
                mean.to_dict("records"), "PSO convergence, mean of folds (band = fold range of gbest)"
            ),
            use_container_width=True,
        )

    st.markdown("**Chosen configurations per outer fold** (ties with the best are common on flat landscapes)")
    sub = folds[folds.dataset == dataset]
    chosen = []
    for row in sub.itertuples():
        final = exp.final(dataset, int(row.outer_fold), row.method) or {}
        chosen.append(
            {
                "fold": row.outer_fold,
                "method": charts.METHOD_LABEL[row.method],
                "config (n, depth, split)": components.config_text(final.get("best_hyperparameters")),
                "validation (biased)": row.validation_fitness,
                "test accuracy": row.test_accuracy,
                "Δ vs default": row.delta_accuracy_vs_baseline,
                "ties with best": final.get("n_ties_with_best"),
                "on a bound": ", ".join(final.get("boundary_hits") or []) or "–",
            }
        )
    st.dataframe(
        pd.DataFrame(chosen).style.format(
            {"validation (biased)": "{:.4f}", "test accuracy": "{:.4f}", "Δ vs default": "{:+.4f}"},
            na_rep="–",
        ),
        hide_index=True,
        use_container_width=True,
    )

    c3, c4 = st.columns(2)
    labels = audit.get("class_names") or []
    for column, method in ((c3, "baseline"), (c4, "pso")):
        pred = exp.predictions(dataset, method)
        if pred.empty or not labels:
            continue
        n = len(labels)
        matrix = np.zeros((n, n), dtype=int)
        np.add.at(matrix, (pred.y_true.to_numpy(), pred.y_pred.to_numpy()), 1)
        names = labels if n <= 3 else [str(i) for i in range(n)]
        column.plotly_chart(
            charts.confusion(matrix, names, f"Pooled confusion · {charts.METHOD_LABEL[method]}"),
            use_container_width=True,
        )

    dep = exp.deployment(dataset)
    if dep:
        est = dep.get("performance_estimate") or {}
        st.success(
            f"**Recommended configuration (deployment run on all data):** "
            f"{components.config_text(dep['recommended_hyperparameters'])} · expected test accuracy "
            f"≈ {est.get('test_accuracy_mean', float('nan')):.4f} ± {est.get('test_accuracy_std', float('nan')):.4f} "
            f"(outer-CV estimate of the same procedure; the deployment run itself has no test data)."
        )
