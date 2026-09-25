"""HTML building blocks for the demo: the live closed-loop diagram, the sealed test-fold card and KPI cards.

Each function returns an HTML string; the page renders it with ``st.markdown(..., unsafe_allow_html=True)``.
All values shown come from optimizer events or saved result files; nothing here computes a result.
"""

from __future__ import annotations

import html
from collections.abc import Mapping
from typing import Any

CSS = """
<style>
:root { --pso:#2a78d6; --rs:#eb6834; --base:#1baf7a; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
        --grid:#e1e0d9; --axis:#c3c2b7; --surface:#fcfcfb; --plane:#f4f3ef; --crit:#d03b3b; --good:#0ca30c; }
.block-container { padding-top: 1.6rem; }
.hero { padding: 0.2rem 0 0.8rem 0; }
.hero h1 { font-size: 1.9rem; margin: 0; color: var(--ink); letter-spacing: -0.01em; }
.hero p { color: var(--ink2); margin: 0.25rem 0 0 0; font-size: 1.0rem; }
.loop { border: 1px solid var(--grid); border-radius: 14px; padding: 14px 16px 10px 16px; background: #fff; }
.loop-row { display: flex; align-items: stretch; gap: 8px; }
.stage { flex: 1; border: 1.5px solid var(--grid); border-radius: 12px; padding: 10px 12px; background: var(--surface);
         transition: all .25s ease; min-height: 92px; }
.stage .k { font-size: 0.72rem; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
.stage .t { font-weight: 650; color: var(--ink); font-size: 0.98rem; margin-top: 2px; }
.stage .v { font-family: ui-monospace, Consolas, monospace; color: var(--ink2); font-size: 0.86rem; margin-top: 6px; }
.stage.active { border-color: var(--pso); background: #eef5fd; box-shadow: 0 0 0 3px rgba(42,120,214,.15); }
.stage.active .k { color: var(--pso); }
.arrow { align-self: center; color: var(--axis); font-size: 1.4rem; }
.arrow.active { color: var(--pso); }
.feedback { margin-top: 10px; border-top: 2px dashed var(--axis); padding-top: 8px; color: var(--ink2);
            font-size: 0.88rem; display: flex; justify-content: space-between; gap: 12px; }
.feedback.active { border-top-color: var(--pso); color: var(--ink); }
.feedback b { color: var(--pso); }
.seal { border-radius: 14px; padding: 14px 16px; border: 1.5px solid var(--grid); background: #fff; height: 100%; }
.seal .k { font-size: 0.72rem; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
.seal .t { font-weight: 700; font-size: 1.05rem; margin: 4px 0; }
.seal.locked { border-color: var(--crit); background: #fdf3f3; }
.seal.locked .t { color: var(--crit); }
.seal.open { border-color: var(--good); background: #f1f9f1; }
.seal.open .t { color: #006300; }
.seal .d { color: var(--ink2); font-size: 0.86rem; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 6px 0 4px 0; }
.kpi { border: 1px solid var(--grid); border-radius: 12px; padding: 10px 12px; background: #fff; }
.kpi .k { font-size: 0.72rem; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
.kpi .v { font-size: 1.35rem; font-weight: 700; color: var(--ink); margin-top: 2px; }
.kpi .s { font-size: 0.8rem; color: var(--ink2); }
.pill { display:inline-block; padding: 2px 9px; border-radius: 999px; font-size: 0.78rem; font-weight: 600; }
.pill.ok { background:#e6f4e6; color:#006300; } .pill.bad { background:#fbe6e6; color: var(--crit); }
.pill.info { background:#e7f0fb; color:#1c5cab; }
.caption { color: var(--muted); font-size: 0.82rem; }
</style>
"""


def _e(value: Any) -> str:
    return html.escape(str(value))


def config_text(config: Mapping[str, Any] | None) -> str:
    """``(n_estimators, max_depth, min_samples_split)`` as a compact tuple string."""
    if not config:
        return "–"
    return f"({config.get('n_estimators')}, {config.get('max_depth')}, {config.get('min_samples_split')})"


def loop_diagram(
    stage: str = "idle",
    iteration: int | None = None,
    total_iterations: int | None = None,
    particle: int | None = None,
    config: Mapping[str, Any] | None = None,
    fitness: float | None = None,
    gbest: Mapping[str, Any] | None = None,
    gbest_fitness: float | None = None,
    inner_folds: int = 5,
    open_loop: bool = False,
) -> str:
    """The closed loop, with the stage currently executing highlighted and the values flowing through it.

    ``stage``: ``idle`` | ``evaluate`` (a particle was just scored: forward path lit) | ``update`` (the
    feedback edge lit: PSO updates memory and moves the swarm) | ``done``. ``open_loop`` draws random search:
    the same plant and measurement, but the feedback edge is cut.
    """
    forward = stage == "evaluate"
    feedback = stage == "update"
    it = "–" if iteration is None else f"{iteration} / {total_iterations}"
    part = "–" if particle is None else f"particle {particle}"
    fit = "–" if fitness is None else f"{fitness:.4f}"
    gfit = "–" if gbest_fitness is None else f"{gbest_fitness:.4f}"

    def stage_box(key: str, title: str, value: str, active: bool) -> str:
        return (
            f'<div class="stage{" active" if active else ""}"><div class="k">{_e(key)}</div>'
            f'<div class="t">{_e(title)}</div><div class="v">{value}</div></div>'
        )

    arrow = f'<div class="arrow{" active" if forward else ""}">&#10140;</div>'
    if open_loop:
        first = stage_box(
            "① sampler", "Random search", f"sample {_e(part)}<br>(uniform, independent)", forward
        )
        tail = (
            f'<div class="feedback"><span>⑤ <b style="color:var(--crit)">no feedback</b>: the next configuration is '
            f"drawn independently of every fitness value (open loop)</span>"
            f"<span>best so far {_e(config_text(gbest))} = <b>{_e(gfit)}</b></span></div></div>"
        )
    else:
        first = stage_box(
            "① controller", "PSO swarm", f"iteration {_e(it)}<br>{_e(part)}", forward or feedback
        )
        tail = (
            f'<div class="feedback{" active" if feedback else ""}"><span>⑤ <b>feedback</b>: fitness → pbest / gbest '
            f"update → new velocities → new positions (next hyperparameters)</span>"
            f"<span>gbest {_e(config_text(gbest))} = <b>{_e(gfit)}</b></span></div></div>"
        )
    return (
        '<div class="loop"><div class="loop-row">'
        + first
        + arrow
        + stage_box(
            "② control action", "Hyperparameters", f"(n, depth, split)<br>{_e(config_text(config))}", forward
        )
        + arrow
        + stage_box(
            "③ plant",
            "Random Forest",
            f"trained on {inner_folds - 1}/{inner_folds}, validated on 1/{inner_folds}"
            f"<br>× {inner_folds} inner folds",
            forward,
        )
        + arrow
        + stage_box("④ measurement", "Fitness", f"mean CV accuracy<br><b>{_e(fit)}</b>", forward)
        + "</div>"
        + tail
    )


def seal_card(state: str, detail: str = "") -> str:
    """The held-out test fold: ``locked`` during optimization, ``open`` once, after the loop has ended."""
    if state == "open":
        return (
            f'<div class="seal open"><div class="k">held-out test fold</div><div class="t">UNSEALED · scored once'
            f'</div><div class="d">{detail or "Opened only after optimization finished."}</div></div>'
        )
    return (
        '<div class="seal locked"><div class="k">held-out test fold</div><div class="t">SEALED</div>'
        f'<div class="d">{detail or "Unreachable while PSO runs: reveal() raises TestSetAccessError."}</div></div>'
    )


def kpis(items: list[tuple[str, str, str]]) -> str:
    """A responsive row of KPI cards: ``(label, value, subtitle)``."""
    cards = "".join(
        f'<div class="kpi"><div class="k">{_e(k)}</div><div class="v">{_e(v)}</div><div class="s">{_e(s)}</div></div>'
        for k, v, s in items
    )
    return f'<div class="kpis">{cards}</div>'


def pill(text: str, kind: str = "info") -> str:
    """A small status pill (``ok`` | ``bad`` | ``info``); the text carries the meaning, not the color alone."""
    return f'<span class="pill {kind}">{_e(text)}</span>'


def hero(title: str, subtitle: str) -> str:
    return f'<div class="hero"><h1>{_e(title)}</h1><p>{_e(subtitle)}</p></div>'
