"""Demo layer (ADR-026): pure chart/HTML builders, and AppTest smoke runs of every page incl. a live loop."""

from __future__ import annotations

import json

import pandas as pd
import plotly.utils
import pytest

from pso_rf.app import charts, components

ITERATIONS = [
    {
        "iteration": t,
        "gbest_fitness": 0.8 + 0.01 * t,
        "mean_fitness": 0.78,
        "min_fitness": 0.7,
        "max_fitness": 0.8 + 0.01 * t,
        "gbest_n_estimators": 100,
        "gbest_max_depth": 5,
        "gbest_min_samples_split": 2,
    }
    for t in range(4)
]


def _json(fig) -> str:
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def test_figures_build_and_serialize_as_plain_lists() -> None:
    swarm_rows = [
        {
            "particle_id": i,
            "n_estimators": 60 + i,
            "max_depth": 3,
            "min_samples_split": 2,
            "fitness": 0.9,
            "pos_n_estimators": 60.2 + i,
            "pos_max_depth": 3.1,
            "pos_min_samples_split": 2.2,
            "vel_n_estimators": 1.0,
            "vel_max_depth": 0.1,
            "vel_min_samples_split": 0.0,
        }
        for i in range(3)
    ]
    frame = pd.DataFrame(
        {
            "dataset_label": ["Iris"] * 3,
            "method": ["baseline", "random_search", "pso"],
            "mean": [0.9, 0.91, 0.92],
            "delta": [0.0, 0.01, 0.02],
        }
    )
    figures = [
        charts.convergence(ITERATIONS),
        charts.anytime(
            {"pso": [0.8, 0.9], "random_search": [0.7, 0.85]}, bands={"pso": ([0.7, 0.8], [0.9, 0.95])}
        ),
        charts.swarm_3d(
            swarm_rows,
            swarm_rows,
            {"n_estimators": 60, "max_depth": 3, "min_samples_split": 2},
            {"n_estimators": 70, "max_depth": 4, "min_samples_split": 3},
        ),
        charts.method_bars([{"method": "pso", "value": 0.9, "folds": [0.88, 0.92]}], "value", "t"),
        charts.grouped_bars(frame, "t"),
        charts.delta_dots(frame, "t"),
        charts.confusion([[3, 1], [0, 4]], ["a", "b"], "t"),
        charts.hyperparameter_paths(pd.DataFrame(ITERATIONS), "t"),
        charts.ablation({"true feedback": [1.0, 0.5], "constant feedback": [1.0, 1.0]}, "t"),
    ]
    for fig in figures:
        text = _json(fig)
        assert "bdata" not in text  # Streamlit 1.28's plotly.js cannot decode typed arrays
        assert len(fig.data) >= 1
    assert [t.name for t in figures[0].data][-1] == "gbest"


def test_loop_diagram_shows_values_and_open_loop_variant() -> None:
    closed = components.loop_diagram(
        "evaluate", 3, 20, 7, {"n_estimators": 120, "max_depth": 8, "min_samples_split": 4}, 0.9123
    )
    assert (
        "(120, 8, 4)" in closed and "0.9123" in closed and "feedback" in closed and "stage active" in closed
    )
    opened = components.loop_diagram("evaluate", particle=5, open_loop=True)
    assert "no feedback" in opened and "Random search" in opened
    assert "SEALED" in components.seal_card("locked") and "UNSEALED" in components.seal_card("open")
    assert "<script" not in components.kpis([("a", "<script>", "b")])  # values are escaped


APP = "app.py"


@pytest.fixture
def app(repo_root, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(repo_root)
    return AppTest.from_file(str(repo_root / APP), default_timeout=300)


def test_every_page_renders_without_exceptions(app) -> None:
    at = app.run()
    assert not at.exception
    for page in list(at.sidebar.radio[0].options)[1:]:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, page


def test_isolation_proof_button_blocks_the_peek(app) -> None:
    at = app.run()
    at.sidebar.radio[0].set_value(at.sidebar.radio[0].options[-1]).run()
    at.button[0].click().run()
    assert any("TestSetAccessError" in s.value for s in at.success)


@pytest.mark.slow
def test_live_lab_runs_the_real_closed_loop(app, repo_root) -> None:
    at = app.run()
    at.sidebar.slider[0].set_value(4)
    at.sidebar.slider[1].set_value(2)
    at.run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    table = at.dataframe[0].value
    assert list(table["method"]) == [
        "Default RF",
        "Your manual pick",
        "Random search (open loop)",
        "PSO (closed loop)",
    ]
    assert (table["configs evaluated"].iloc[2:] == 12).all()  # equal budgets: 4 × (2 + 1)
    assert table["test accuracy"].between(0, 1).all()
