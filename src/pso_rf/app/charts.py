"""Plotly figure builders for the demo. Pure functions of plain data (no Streamlit), so they are unit-tested.

Colors follow the validated reference palette used by the static figures: one fixed color per method,
a single-hue blue ramp for fitness, recessive grid and axes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly import basedatatypes as _plotly_base

# Compatibility shim: Plotly >= 6 sends numpy arrays to the browser as base64 typed arrays ("bdata"), which the
# plotly.js bundled with Streamlit 1.28 cannot decode (charts render garbled). Switching off that single step
# makes arrays serialize as plain JSON lists; the figures themselves are unchanged.
_plotly_base.convert_to_base64 = lambda obj: None

METHOD_COLOR = {"pso": "#2a78d6", "random_search": "#eb6834", "baseline": "#1baf7a", "manual": "#4a3aa7"}
METHOD_LABEL = {
    "pso": "PSO (closed loop)",
    "random_search": "Random search (open loop)",
    "baseline": "Default RF",
    "manual": "Your manual pick",
}
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
FITNESS_SCALE = [[0.0, "#cde2fb"], [0.35, "#6da7ec"], [0.7, "#256abf"], [1.0, "#0d366b"]]
HYPERPARAMETERS = ("n_estimators", "max_depth", "min_samples_split")
BOUNDS = {"n_estimators": (50, 200), "max_depth": (2, 20), "min_samples_split": (2, 10)}


def _layout(fig: go.Figure, title: str | None = None, height: int = 360, **kw: Any) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 15, "color": INK}, "x": 0.0, "xanchor": "left"}
        if title
        else None,
        height=height,
        margin={"l": 55, "r": 20, "t": 46 if title else 16, "b": 80},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"family": "Inter, Segoe UI, sans-serif", "size": 12, "color": INK_2},
        legend={"orientation": "h", "yanchor": "top", "y": -0.2, "xanchor": "left", "x": 0.0},
        hoverlabel={"bgcolor": "#ffffff", "font_color": INK},
        **kw,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    return fig


def convergence(iterations: Sequence[Mapping[str, Any]], title: str = "Convergence") -> go.Figure:
    """gbest (the fitness fed back and remembered), swarm mean and the min–max band per iteration."""
    frame = pd.DataFrame(iterations)
    fig = go.Figure()
    if not frame.empty:
        fig.add_trace(
            go.Scatter(
                x=frame.iteration,
                y=frame.max_fitness,
                mode="lines",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=frame.iteration,
                y=frame.min_fitness,
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(42,120,214,0.12)",
                name="swarm min–max",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=frame.iteration,
                y=frame.mean_fitness,
                mode="lines",
                line={"color": INK_2, "dash": "dash", "width": 1.5},
                name="swarm mean",
                hovertemplate="iter %{x}<br>mean %{y:.4f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=frame.iteration,
                y=frame.gbest_fitness,
                mode="lines+markers",
                line={"color": METHOD_COLOR["pso"], "width": 2.5, "shape": "hv"},
                marker={"size": 7},
                name="gbest",
                customdata=frame[
                    ["gbest_n_estimators", "gbest_max_depth", "gbest_min_samples_split"]
                ].to_numpy(),
                hovertemplate="iter %{x}<br>gbest %{y:.4f}<br>"
                "(%{customdata[0]}, %{customdata[1]}, %{customdata[2]})<extra></extra>",
            )
        )
    fig.update_xaxes(title="PSO iteration", dtick=1 if len(frame) <= 21 else None)
    fig.update_yaxes(title="validation accuracy (inner 5-fold CV)", tickformat=".3f")
    return _layout(fig, title)


def anytime(
    curves: Mapping[str, Sequence[float]],
    title: str = "PSO vs random search (equal budget)",
    bands: Mapping[str, tuple[Sequence[float], Sequence[float]]] | None = None,
) -> go.Figure:
    """Best-so-far validation accuracy vs number of configurations tried, one line per method."""
    fig = go.Figure()
    for method, values in curves.items():
        x = np.arange(1, len(values) + 1)
        if bands and method in bands:
            low, high = bands[method]
            fig.add_trace(
                go.Scatter(x=x, y=high, mode="lines", line={"width": 0}, hoverinfo="skip", showlegend=False)
            )
            rgba = _rgba(METHOD_COLOR[method], 0.15)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=low,
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor=rgba,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=values,
                mode="lines",
                name=METHOD_LABEL[method],
                line={"color": METHOD_COLOR[method], "width": 2.5, "shape": "hv"},
                hovertemplate="%{x} evaluations<br>best %{y:.4f}<extra>" + METHOD_LABEL[method] + "</extra>",
            )
        )
    fig.update_xaxes(title="configurations evaluated")
    fig.update_yaxes(title="best validation accuracy so far", tickformat=".3f")
    return _layout(fig, title)


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def swarm_3d(
    current: Sequence[Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]] = (),
    gbest: Mapping[str, Any] | None = None,
    manual: Mapping[str, Any] | None = None,
    title: str | None = None,
    height: int = 460,
) -> go.Figure:
    """The swarm in the 3-D search space: current particles colored by fitness, faint past positions,
    velocity arrows, and the global best."""
    fig = go.Figure()
    if history:
        past = pd.DataFrame(history)
        fig.add_trace(
            go.Scatter3d(
                x=past.n_estimators,
                y=past.max_depth,
                z=past.min_samples_split,
                mode="markers",
                marker={"size": 3, "color": AXIS, "opacity": 0.5},
                name="earlier positions",
                hoverinfo="skip",
            )
        )
    if current:
        now = pd.DataFrame(current)
        finite = now[np.isfinite(now.fitness)]
        if {"pos_n_estimators", "vel_n_estimators"} <= set(now.columns):
            xs, ys, zs = [], [], []
            for row in now.itertuples():
                xs += [row.pos_n_estimators, row.pos_n_estimators + row.vel_n_estimators, None]
                ys += [row.pos_max_depth, row.pos_max_depth + row.vel_max_depth, None]
                zs += [row.pos_min_samples_split, row.pos_min_samples_split + row.vel_min_samples_split, None]
            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="lines",
                    line={"color": MUTED, "width": 3},
                    name="velocity (arrival step)",
                    hoverinfo="skip",
                )
            )
        fig.add_trace(
            go.Scatter3d(
                x=finite.n_estimators,
                y=finite.max_depth,
                z=finite.min_samples_split,
                mode="markers",
                marker={
                    "size": 7,
                    "color": finite.fitness,
                    "colorscale": FITNESS_SCALE,
                    "showscale": True,
                    "colorbar": {"title": "fitness", "thickness": 12, "len": 0.6},
                    "line": {"width": 1, "color": "#ffffff"},
                },
                customdata=finite[["particle_id", "fitness"]].to_numpy() if "particle_id" in finite else None,
                name="particles now",
                hovertemplate="particle %{customdata[0]}<br>(%{x}, %{y}, %{z})<br>fitness %{customdata[1]:.4f}"
                "<extra></extra>"
                if "particle_id" in finite
                else None,
            )
        )
    if gbest:
        fig.add_trace(
            go.Scatter3d(
                x=[gbest["n_estimators"]],
                y=[gbest["max_depth"]],
                z=[gbest["min_samples_split"]],
                mode="markers",
                marker={
                    "size": 12,
                    "color": "#e34948",
                    "symbol": "diamond",
                    "line": {"width": 2, "color": "#ffffff"},
                },
                name="gbest",
                hovertemplate="gbest (%{x}, %{y}, %{z})<extra></extra>",
            )
        )
    if manual:
        fig.add_trace(
            go.Scatter3d(
                x=[manual["n_estimators"]],
                y=[manual["max_depth"]],
                z=[manual["min_samples_split"]],
                mode="markers",
                marker={"size": 10, "color": METHOD_COLOR["manual"], "symbol": "square"},
                name="your manual pick",
            )
        )
    axis = {"backgroundcolor": SURFACE, "gridcolor": GRID, "showbackground": True}
    fig.update_layout(
        scene={
            "xaxis": {**axis, "title": "n_estimators", "range": [45, 205]},
            "yaxis": {**axis, "title": "max_depth", "range": [1, 21]},
            "zaxis": {**axis, "title": "min_samples_split", "range": [1.5, 10.5]},
            "camera": {"eye": {"x": 1.6, "y": 1.5, "z": 0.9}},
            "aspectmode": "cube",
        },
        uirevision="swarm",
    )
    return _layout(fig, title, height=height)


def method_bars(rows: Sequence[Mapping[str, Any]], value: str, title: str, fmt: str = ".3f") -> go.Figure:
    """One bar per method (mean) with optional fold dots; rows need ``method``, ``value`` and optional ``folds``."""
    fig = go.Figure()
    for row in rows:
        method = row["method"]
        fig.add_trace(
            go.Bar(
                x=[METHOD_LABEL[method]],
                y=[row[value]],
                marker_color=METHOD_COLOR[method],
                name=METHOD_LABEL[method],
                width=0.55,
                text=[format(row[value], fmt)],
                textposition="outside",
                hovertemplate="%{x}<br>%{y:" + fmt + "}<extra></extra>",
            )
        )
        if row.get("folds"):
            folds = row["folds"]
            fig.add_trace(
                go.Scatter(
                    x=[METHOD_LABEL[method]] * len(folds),
                    y=folds,
                    mode="markers",
                    marker={"color": INK_2, "size": 8, "line": {"width": 1, "color": "#ffffff"}},
                    showlegend=False,
                    hovertemplate="fold value %{y:" + fmt + "}<extra></extra>",
                )
            )
    values = [r[value] for r in rows] + [v for r in rows for v in (r.get("folds") or [])]
    if values:
        low = min(values)
        fig.update_yaxes(range=[max(0.0, low - (1 - low) * 1.5 - 0.02), 1.0 + (1 - low) * 0.35 + 0.005])
    fig.update_layout(showlegend=False, bargap=0.4)
    fig.update_yaxes(tickformat=fmt)
    return _layout(fig, title, height=340)


def grouped_bars(frame: pd.DataFrame, title: str) -> go.Figure:
    """Test accuracy per dataset (x) and method (color); ``frame`` has dataset, method, mean, folds."""
    fig = go.Figure()
    for method in ("baseline", "random_search", "pso"):
        sub = frame[frame.method == method]
        if sub.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=sub.dataset_label,
                y=sub["mean"],
                name=METHOD_LABEL[method],
                marker_color=METHOD_COLOR[method],
                text=[f"{v:.3f}" for v in sub["mean"]],
                textposition="outside",
                hovertemplate="%{x}<br>" + METHOD_LABEL[method] + ": %{y:.4f}<extra></extra>",
            )
        )
    low = float(frame["mean"].min()) if not frame.empty else 0.8
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_yaxes(
        title="mean test accuracy (held-out folds)", tickformat=".2f", range=[max(0, low - 0.1), 1.02]
    )
    return _layout(fig, title, height=380)


def delta_dots(frame: pd.DataFrame, title: str) -> go.Figure:
    """Per-fold test-accuracy difference to the default RF, per dataset, for random search and PSO."""
    fig = go.Figure()
    for method in ("random_search", "pso"):
        sub = frame[frame.method == method]
        if sub.empty:
            continue
        fig.add_trace(
            go.Box(
                x=sub.dataset_label,
                y=sub.delta,
                name=METHOD_LABEL[method],
                boxpoints="all",
                jitter=0.35,
                pointpos=0,
                marker={"color": METHOD_COLOR[method], "size": 9},
                line={"color": METHOD_COLOR[method], "width": 1.5},
                fillcolor="rgba(0,0,0,0)",
                hovertemplate="%{x}<br>Δ %{y:+.4f}<extra>" + METHOD_LABEL[method] + "</extra>",
            )
        )
    fig.add_hline(y=0, line={"color": AXIS, "width": 1.5})
    fig.update_layout(boxmode="group")
    fig.update_yaxes(title="Δ test accuracy vs default RF (per fold)", tickformat="+.3f")
    return _layout(fig, title, height=380)


def confusion(matrix: Sequence[Sequence[int]], labels: Sequence[str], title: str) -> go.Figure:
    """Confusion-matrix heatmap (rows = true, columns = predicted) with counts."""
    z = np.asarray(matrix)
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=list(labels),
            y=list(labels),
            colorscale=[[0, "#fcfcfb"], [1, "#0d366b"]],
            showscale=False,
            text=z,
            texttemplate="%{text}",
            hovertemplate="true %{y}<br>predicted %{x}<br>%{z}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", title="true")
    fig.update_xaxes(title="predicted", side="bottom")
    return _layout(fig, title, height=320 if len(labels) <= 3 else 420)


def hyperparameter_paths(iterations: pd.DataFrame, title: str) -> go.Figure:
    """gbest hyperparameters vs iteration, normalized to the search range so the three share one axis."""
    fig = go.Figure()
    colors = {"n_estimators": "#2a78d6", "max_depth": "#eb6834", "min_samples_split": "#1baf7a"}
    for name in HYPERPARAMETERS:
        low, high = BOUNDS[name]
        values = iterations[f"gbest_{name}"]
        fig.add_trace(
            go.Scatter(
                x=iterations.iteration,
                y=(values - low) / (high - low),
                mode="lines+markers",
                line={"color": colors[name], "width": 2, "shape": "hv"},
                name=name,
                customdata=values,
                hovertemplate=name + " = %{customdata}<extra></extra>",
            )
        )
    fig.update_yaxes(title="position in range (0 = low bound, 1 = high)", range=[-0.05, 1.05])
    fig.update_xaxes(title="PSO iteration")
    return _layout(fig, title, height=320)


def ablation(curves: Mapping[str, Sequence[float]], title: str) -> go.Figure:
    """Distance of gbest to the true optimum per iteration, for true vs corrupted feedback (stub system)."""
    styles = {
        "true feedback": METHOD_COLOR["pso"],
        "constant feedback": MUTED,
        "shuffled feedback": "#e34948",
    }
    fig = go.Figure()
    for name, values in curves.items():
        fig.add_trace(
            go.Scatter(
                x=list(range(len(values))),
                y=values,
                mode="lines+markers",
                name=name,
                line={"color": styles.get(name, INK_2), "width": 2.5},
            )
        )
    fig.update_xaxes(title="PSO iteration")
    fig.update_yaxes(title="distance of gbest to the true optimum", rangemode="tozero")
    return _layout(fig, title, height=340)
