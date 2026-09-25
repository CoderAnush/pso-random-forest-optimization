"""Interactive web frontend (Tornado + vanilla JS + three.js, ADR-027).

Live runs stream events from ``pso_rf.experiments.runner.run_fold`` (the experiment's own code path);
every other view reads saved result files. Launch with ``python -m pso_rf web``.
"""
