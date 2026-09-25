"""The figures. Colors follow the validated reference palette: one fixed color per method (never by rank),
datasets are small multiples (never colors), recessive grid and axes, and a legend on every chart."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pso_rf.utils.io import write_json_atomic  # noqa: E402

# reference palette (light): categorical slots 1–3 validate all-pairs; chrome and ink tokens
METHOD_COLOR = {"pso": "#2a78d6", "random_search": "#eb6834", "baseline": "#1baf7a"}
METHOD_MARKER = {"pso": "o", "random_search": "s", "baseline": "D"}
METHOD_LABEL = {
    "pso": "PSO (closed loop)",
    "random_search": "Random search (open loop)",
    "baseline": "Default RF",
}
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQUENTIAL = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DATASET_LABEL = {"iris": "Iris", "digits": "Digits", "heart_cleveland": "Heart Disease (Cleveland)"}
HYPERPARAMETERS = ("n_estimators", "max_depth", "min_samples_split")


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "font.family": "DejaVu Sans",
            "lines.linewidth": 2,
        }
    )


def _title(fig: plt.Figure, text: str) -> None:
    """Figure title, left-aligned above the panel titles (never overlapping them)."""
    fig.suptitle(text, color=INK, fontsize=11, x=0.01, y=1.04, ha="left")


class _Figures:
    """Collects figures and the files each one was drawn from."""

    def __init__(self, root: Path, out: Path) -> None:
        self.root, self.out = root, out
        self.sources: dict[str, list[str]] = {}
        self.config = json.loads((root / "config.resolved.json").read_text(encoding="utf-8"))
        self.datasets = [d for d in self.config["experiment"]["datasets"] if (root / d).is_dir()]

    def rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def csvs(self, pattern: str) -> tuple[pd.DataFrame, list[str]]:
        paths = sorted(self.root.glob(pattern))
        if not paths:
            return pd.DataFrame(), []
        return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True), [self.rel(p) for p in paths]

    def save(self, fig: plt.Figure, name: str, sources: list[str]) -> None:
        fig.savefig(self.out / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.sources[name] = sorted(set(sources))

    def panels(self, n: int, width: float = 4.2, height: float = 3.2, **kw: Any):
        fig, axes = plt.subplots(1, n, figsize=(width * n, height), squeeze=False, **kw)
        return fig, list(axes[0])


def _fold_band(ax, frame: pd.DataFrame, x: str, y: str, color: str, label: str) -> None:
    """Thin line per fold plus the mean over folds."""
    for _, run in frame.groupby("outer_fold"):
        ax.plot(run[x], run[y], color=color, alpha=0.3, linewidth=1)
    mean = frame.groupby(x)[y].mean()
    ax.plot(mean.index, mean.values, color=color, linewidth=2, label=label)


# ------------------------------------------------------------------------------------------------ figures


def fig_convergence(f: _Figures) -> None:
    """F3: gbest validation accuracy vs iteration, one thin line per fold plus the mean."""
    it, sources = f.csvs("*/fold_*/pso/iterations.csv")
    if it.empty:
        return
    fig, axes = f.panels(len(f.datasets), sharey=False)
    for ax, dataset in zip(axes, f.datasets, strict=True):
        _fold_band(
            ax, it[it.dataset == dataset], "iteration", "gbest_fitness", METHOD_COLOR["pso"], "mean of folds"
        )
        ax.plot([], [], color=METHOD_COLOR["pso"], alpha=0.3, linewidth=1, label="one outer fold")
        ax.set(title=DATASET_LABEL.get(dataset, dataset), xlabel="PSO iteration")
    axes[0].set_ylabel("gbest validation accuracy (inner CV)")
    axes[0].legend(loc="lower right")
    _title(fig, "F3 · PSO convergence (validation fitness; optimistically biased, not test performance)")
    f.save(fig, "F3_convergence.png", sources)


def fig_anytime(f: _Figures) -> None:
    """F4: best-so-far validation accuracy vs evaluation count, PSO vs random search (mean + min–max band)."""
    ev, sources = f.csvs("*/fold_*/*/evaluations.csv")
    if ev.empty:
        return
    ev["n_eval"] = ev["eval_index"] + 1
    fig, axes = f.panels(len(f.datasets))
    for ax, dataset in zip(axes, f.datasets, strict=True):
        for method in ("random_search", "pso"):
            sub = ev[(ev.dataset == dataset) & (ev.method == method)]
            if sub.empty:
                continue
            grid = sub.pivot_table(index="n_eval", columns="outer_fold", values="best_so_far_fitness")
            color = METHOD_COLOR[method]
            ax.fill_between(
                grid.index, grid.min(axis=1), grid.max(axis=1), color=color, alpha=0.15, linewidth=0
            )
            ax.plot(grid.index, grid.mean(axis=1), color=color, label=METHOD_LABEL[method])
        ax.set(title=DATASET_LABEL.get(dataset, dataset), xlabel="evaluations (configurations tried)")
    axes[0].set_ylabel("best validation accuracy so far")
    axes[0].legend(loc="lower right")
    _title(fig, "F4 · Value of feedback: equal budgets, mean over outer folds (band = min–max)")
    f.save(fig, "F4_anytime.png", sources)


def _methods(frame: pd.DataFrame) -> list[str]:
    return [m for m in ("baseline", "random_search", "pso") if m in set(frame.method)]


def fig_test_accuracy(f: _Figures) -> None:
    """F5: held-out test accuracy per method: bar = mean over folds, dots = individual folds."""
    path = f.root / "summary_folds.csv"
    folds = pd.read_csv(path)
    fig, axes = f.panels(len(f.datasets), width=3.6)
    for ax, dataset in zip(axes, f.datasets, strict=True):
        sub = folds[folds.dataset == dataset]
        for x, method in enumerate(_methods(sub)):
            values = sub[sub.method == method]["test_accuracy"].to_numpy()
            ax.bar(x, values.mean(), width=0.55, color=METHOD_COLOR[method], label=METHOD_LABEL[method])
            jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else [0.0]
            ax.scatter(
                x + np.asarray(jitter), values, s=18, color=INK_2, zorder=3, edgecolor=SURFACE, linewidth=1
            )
            ax.annotate(
                f"{values.mean():.3f}",
                (x, max(values.max(), values.mean())),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=INK,
            )
        low = max(0.0, sub["test_accuracy"].min() - 0.08)
        ax.set(title=DATASET_LABEL.get(dataset, dataset), xticks=[], ylim=(low, 1.02))
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("test accuracy on held-out folds")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    _title(
        fig,
        f"F5 · Held-out test accuracy (bar = mean of {folds.outer_fold.nunique()} outer folds, dots = folds)",
    )
    f.save(fig, "F5_test_accuracy.png", [f.rel(path)])


def fig_trajectories(f: _Figures) -> None:
    """F6: decoded hyperparameters of every particle vs iteration (outer fold 0), with the gbest path."""
    ev, sources = f.csvs("*/fold_0/pso/evaluations.csv")
    it, more = f.csvs("*/fold_0/pso/iterations.csv")
    if ev.empty:
        return
    fig, axes = plt.subplots(len(f.datasets), 3, figsize=(12, 2.6 * len(f.datasets)), squeeze=False)
    for row, dataset in enumerate(f.datasets):
        sub, gb = ev[ev.dataset == dataset], it[it.dataset == dataset]
        for col, name in enumerate(HYPERPARAMETERS):
            ax = axes[row][col]
            for _, particle in sub.groupby("particle_id"):
                ax.plot(particle["iteration"], particle[name], color=MUTED, alpha=0.45, linewidth=1)
            ax.step(
                gb["iteration"], gb[f"gbest_{name}"], where="post", color=METHOD_COLOR["pso"], label="gbest"
            )
            ax.set_title(f"{DATASET_LABEL.get(dataset, dataset)} · {name}", fontsize=9)
            if row == len(f.datasets) - 1:
                ax.set_xlabel("PSO iteration")
    axes[0][0].plot([], [], color=MUTED, linewidth=1, label="one particle")
    axes[0][0].legend(loc="upper right")
    _title(fig, "F6 · Search behaviour: every particle's configuration per iteration (outer fold 0)")
    fig.tight_layout()
    f.save(fig, "F6_trajectories.png", sources + more)


def fig_particle_fitness(f: _Figures) -> None:
    """F7: every particle's validation accuracy per iteration, with the swarm mean and gbest (fold 0)."""
    ev, sources = f.csvs("*/fold_0/pso/evaluations.csv")
    it, more = f.csvs("*/fold_0/pso/iterations.csv")
    if ev.empty:
        return
    rng = np.random.default_rng(0)  # jitter only; affects drawing, not data
    fig, axes = f.panels(len(f.datasets))
    for ax, dataset in zip(axes, f.datasets, strict=True):
        sub = ev[(ev.dataset == dataset) & np.isfinite(ev.fitness)]
        gb = it[it.dataset == dataset]
        x = sub["iteration"] + rng.uniform(-0.18, 0.18, len(sub))
        ax.scatter(
            x, sub["fitness"], s=10, color=MUTED, alpha=0.6, linewidth=0, label="one particle evaluation"
        )
        ax.plot(
            gb["iteration"],
            gb["mean_fitness"],
            color=INK_2,
            linestyle="--",
            linewidth=1.5,
            label="swarm mean",
        )
        ax.plot(gb["iteration"], gb["gbest_fitness"], color=METHOD_COLOR["pso"], label="gbest")
        ax.set(title=DATASET_LABEL.get(dataset, dataset), xlabel="PSO iteration")
    axes[0].set_ylabel("validation accuracy (inner CV)")
    axes[0].legend(loc="lower right")
    _title(fig, "F7 · Particle fitness progression (outer fold 0)")
    f.save(fig, "F7_particle_fitness.png", sources + more)


def fig_gbest_evolution(f: _Figures) -> None:
    """F8: the gbest configuration vs iteration, one line per outer fold."""
    it, sources = f.csvs("*/fold_*/pso/iterations.csv")
    if it.empty:
        return
    fig, axes = plt.subplots(len(f.datasets), 3, figsize=(12, 2.6 * len(f.datasets)), squeeze=False)
    for row, dataset in enumerate(f.datasets):
        sub = it[it.dataset == dataset]
        for col, name in enumerate(HYPERPARAMETERS):
            ax = axes[row][col]
            for _, run in sub.groupby("outer_fold"):
                ax.step(
                    run["iteration"],
                    run[f"gbest_{name}"],
                    where="post",
                    color=METHOD_COLOR["pso"],
                    alpha=0.55,
                    linewidth=1.3,
                )
            ax.set_title(f"{DATASET_LABEL.get(dataset, dataset)} · gbest {name}", fontsize=9)
            if row == len(f.datasets) - 1:
                ax.set_xlabel("PSO iteration")
    axes[0][0].plot([], [], color=METHOD_COLOR["pso"], alpha=0.55, linewidth=1.3, label="one outer fold")
    axes[0][0].legend(loc="upper right")
    _title(fig, "F8 · Best-hyperparameter evolution")
    fig.tight_layout()
    f.save(fig, "F8_gbest_evolution.png", sources)


def fig_deltas(f: _Figures) -> None:
    """F9: test-accuracy difference to the default RF per fold (dots) and mean (bar marker)."""
    path = f.root / "summary_folds.csv"
    folds = pd.read_csv(path)
    methods = [m for m in ("random_search", "pso") if m in set(folds.method)]
    fig, axes = f.panels(len(f.datasets), width=3.4)
    for ax, dataset in zip(axes, f.datasets, strict=True):
        sub = folds[folds.dataset == dataset]
        ax.axhline(0, color=AXIS, linewidth=1)
        for x, method in enumerate(methods):
            deltas = sub[sub.method == method]["delta_accuracy_vs_baseline"].dropna().to_numpy()
            jitter = np.linspace(-0.1, 0.1, len(deltas)) if len(deltas) > 1 else [0.0]
            ax.scatter(
                x + np.asarray(jitter),
                deltas,
                s=26,
                color=METHOD_COLOR[method],
                marker=METHOD_MARKER[method],
                label=METHOD_LABEL[method],
                edgecolor=SURFACE,
                linewidth=1,
            )
            ax.hlines(deltas.mean(), x - 0.22, x + 0.22, color=INK, linewidth=2)
        extent = max(
            0.02, float(np.nanmax(np.abs(sub["delta_accuracy_vs_baseline"].to_numpy(dtype=float)))) * 1.3
        )
        ax.set_ylim(-extent, extent)  # symmetric about 0: gains and losses read on the same scale
        ax.set(title=DATASET_LABEL.get(dataset, dataset), xticks=[], xlim=(-0.6, len(methods) - 0.4))
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Δ test accuracy vs default RF")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    _title(fig, "F9 · Paired per-fold difference to the default RF (black bar = mean)")
    f.save(fig, "F9_deltas.png", [f.rel(path)])


def fig_confusion(f: _Figures) -> None:
    """F10: pooled out-of-fold confusion matrices, default RF vs PSO, per dataset."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL)
    methods = ("baseline", "pso")
    fig, axes = plt.subplots(len(f.datasets), 2, figsize=(8, 3.8 * len(f.datasets)), squeeze=False)
    sources: list[str] = []
    for row, dataset in enumerate(f.datasets):
        audit = json.loads((f.root / dataset / "audit.json").read_text(encoding="utf-8"))
        labels = list(range(audit["n_classes"]))
        for col, method in enumerate(methods):
            ax = axes[row][col]
            paths = sorted(f.root.glob(f"{dataset}/fold_*/{method}/predictions.csv"))
            sources += [f.rel(p) for p in paths]
            if not paths:
                ax.axis("off")
                continue
            pred = pd.concat([pd.read_csv(p) for p in paths])
            matrix = np.zeros((len(labels), len(labels)), dtype=int)
            np.add.at(matrix, (pred.y_true.to_numpy(), pred.y_pred.to_numpy()), 1)
            ax.imshow(matrix, cmap=cmap, vmin=0)
            ax.grid(False)
            for (i, j), value in np.ndenumerate(matrix):
                if value:
                    dark = value > matrix.max() * 0.55
                    ax.text(
                        j,
                        i,
                        str(value),
                        ha="center",
                        va="center",
                        fontsize=7 if len(labels) > 3 else 9,
                        color="#ffffff" if dark else INK,
                    )
            names = audit["class_names"]
            ax.set_xticks(labels, names if len(labels) <= 3 else labels, fontsize=7)
            ax.set_yticks(labels, names if len(labels) <= 3 else labels, fontsize=7)
            ax.set_title(f"{DATASET_LABEL.get(dataset, dataset)} · {METHOD_LABEL[method]}", fontsize=9)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
    _title(fig, "F10 · Pooled out-of-fold confusion matrices (every sample tested once)")
    fig.tight_layout()
    f.save(fig, "F10_confusion.png", sources)


def fig_diversity(f: _Figures) -> None:
    """F11: swarm diversity (mean normalized distance to the centroid) vs iteration."""
    it, sources = f.csvs("*/fold_*/pso/iterations.csv")
    if it.empty:
        return
    fig, axes = f.panels(len(f.datasets))
    for ax, dataset in zip(axes, f.datasets, strict=True):
        _fold_band(
            ax, it[it.dataset == dataset], "iteration", "diversity", METHOD_COLOR["pso"], "mean of folds"
        )
        ax.plot([], [], color=METHOD_COLOR["pso"], alpha=0.3, linewidth=1, label="one outer fold")
        ax.set(title=DATASET_LABEL.get(dataset, dataset), xlabel="PSO iteration", ylim=(0, None))
    axes[0].set_ylabel("swarm diversity (normalized)")
    axes[0].legend(loc="upper right")
    _title(fig, "F11 · Swarm diversity: exploration → exploitation")
    f.save(fig, "F11_diversity.png", sources)


FIGURES = (
    fig_convergence,
    fig_anytime,
    fig_test_accuracy,
    fig_trajectories,
    fig_particle_fitness,
    fig_gbest_evolution,
    fig_deltas,
    fig_confusion,
    fig_diversity,
)


def make_all_plots(results: Path | str, out: Path | str | None = None) -> Path:
    """Draw F3–F11 from ``results`` into ``out`` (default ``<plots_dir>/<exp_id>``) and write SOURCES.json."""
    root = Path(results)
    config = json.loads((root / "config.resolved.json").read_text(encoding="utf-8"))
    out = Path(out) if out is not None else Path(config["experiment"]["plots_dir"]) / root.name
    out.mkdir(parents=True, exist_ok=True)
    _style()
    figures = _Figures(root, out)
    for draw in FIGURES:
        draw(figures)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    write_json_atomic(
        out / "SOURCES.json",
        {
            "exp_id": root.name,
            "results_dir": root.as_posix(),
            "config_hash": manifest["config_hash"],
            "figures": figures.sources,
        },
    )
    return out
