"""Generic optimizers (PSO and random search) that know only numbers.

This package imports only NumPy and the standard library: no scikit-learn, no data and no dataset
names (ADR-023, enforced by UT-22).
"""

from pso_rf.optimization.events import (
    Callback,
    Evaluation,
    EvaluationEvent,
    IterationSummary,
    Objective,
    OptimizationResult,
)
from pso_rf.optimization.pso import PSOConfig, PSOOptimizer
from pso_rf.optimization.random_search import RandomSearch
from pso_rf.optimization.search_space import IntParam, SearchSpace

__all__ = [
    "Callback",
    "Evaluation",
    "EvaluationEvent",
    "IntParam",
    "IterationSummary",
    "Objective",
    "OptimizationResult",
    "PSOConfig",
    "PSOOptimizer",
    "RandomSearch",
    "SearchSpace",
]
