"""Figures F3–F11 (EXPERIMENT_PLAN §9), drawn only from saved result files (ET-05).

Nothing here trains a model or imports the optimizer: every figure reads CSV/JSON files under
``results/<exp_id>/`` and is recorded, with its source files and the config hash, in ``SOURCES.json``.
"""

from pso_rf.visualization.plots import make_all_plots

__all__ = ["make_all_plots"]
