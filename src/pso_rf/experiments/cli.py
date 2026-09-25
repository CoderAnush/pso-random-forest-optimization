"""Command-line interface: ``python -m pso_rf {run,plot,verify,compare,audit}``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from pso_rf.experiments.config import ConfigError, load_config

DEFAULT_CONFIG = Path("configs") / "default.yaml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pso_rf", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run an experiment (closed-loop PSO + comparators)")
    run.add_argument(
        "--config", action="append", type=Path, help="YAML layer(s), in order; default configs/default.yaml"
    )
    run.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="override, e.g. pso.max_iter=30"
    )
    run.add_argument("--datasets", nargs="+", help="subset of datasets")
    run.add_argument("--folds", nargs="+", type=int, help="subset of outer folds")
    run.add_argument("--methods", nargs="+", help="subset of baseline random_search pso")
    run.add_argument("--name", help="experiment name (part of the results directory)")

    plot = commands.add_parser("plot", help="make the figures from a results directory")
    plot.add_argument("--results", type=Path, required=True)
    plot.add_argument("--out", type=Path, help="default: plots/<exp_id>")

    verify = commands.add_parser(
        "verify", help="audit a results directory (completeness, isolation, summaries)"
    )
    verify.add_argument("--results", type=Path, required=True)

    compare = commands.add_parser("compare", help="compare two results directories, ignoring timing fields")
    compare.add_argument("a", type=Path)
    compare.add_argument("b", type=Path)

    web = commands.add_parser("web", help="launch the interactive web frontend (live 3-D swarm, playground)")
    web.add_argument("--port", type=int, default=8600)
    web.add_argument("--no-browser", action="store_true", help="do not open a browser tab")

    demo = commands.add_parser("demo", help="launch the classic Streamlit demo")
    demo.add_argument("--port", type=int, default=8501)

    audit = commands.add_parser("audit", help="audit the datasets and write data/DATASET_AUDIT.md")
    audit.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    audit.add_argument("--data-dir", type=Path, default=Path("data"))
    audit.add_argument("--out", type=Path)
    return parser


def _overrides(args: argparse.Namespace) -> list[str]:
    overrides = list(args.set)
    if args.datasets:
        overrides.append(f"experiment.datasets=[{', '.join(args.datasets)}]")
    if args.folds is not None:
        overrides.append(f"experiment.folds=[{', '.join(map(str, args.folds))}]")
    if args.methods:
        overrides.append(f"experiment.methods=[{', '.join(args.methods)}]")
    if args.name:
        overrides.append(f"experiment.name={args.name}")
    return overrides


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = _parser().parse_args(argv)
    if args.command == "run":
        from pso_rf.experiments.runner import run_experiment

        try:
            cfg = load_config(args.config or [DEFAULT_CONFIG], _overrides(args))
        except ConfigError as exc:
            print(exc, file=sys.stderr)
            return 2
        root = run_experiment(cfg)
        print(root)
        return 0
    if args.command == "plot":
        from pso_rf.visualization import make_all_plots

        out = make_all_plots(args.results, args.out)
        print(out)
        return 0
    if args.command == "verify":
        from pso_rf.experiments.verify import verify_results

        problems = verify_results(args.results)
        for problem in problems:
            print(f"FAIL {problem}")
        print("OK: all checks passed" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0
    if args.command == "compare":
        from pso_rf.experiments.verify import compare_runs

        differences = compare_runs(args.a, args.b)
        for difference in differences:
            print(difference)
        print(
            "identical (excluding timing fields)" if not differences else f"{len(differences)} difference(s)"
        )
        return 1 if differences else 0
    if args.command == "web":
        from pso_rf.web.server import serve

        serve(args.port, open_browser=not args.no_browser)
        return 0
    if args.command == "demo":
        import subprocess

        app = Path(__file__).resolve().parents[3] / "app.py"
        command = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(args.port)]
        return subprocess.call(command, cwd=app.parent)
    if args.command == "audit":
        from pso_rf.datasets import LOADERS, load_dataset
        from pso_rf.datasets.audit import audit, write_audit_markdown

        cfg = load_config([args.config])
        audits, metas = [], {}
        for name in LOADERS:
            fitness = cfg.for_dataset(name).fitness
            bundle = load_dataset(name, args.data_dir)
            audits.append(audit(bundle, fitness.class_ratio_gate, fitness.metric))
            metas[name] = bundle.meta
        out = args.out or args.data_dir / "DATASET_AUDIT.md"
        write_audit_markdown(audits, out, metas, cfg.fitness.class_ratio_gate)
        print(out)
        return 0
    return 2
