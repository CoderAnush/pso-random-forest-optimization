"""Particle Swarm Optimization, implemented from scratch (MATHEMATICAL_FORMULATION §7).

NumPy and the standard library only (UT-22). This module currently defines only ``PSOConfig``;
``PSOOptimizer`` is added in Phase 7.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PSOConfig:
    """PSO parameters; the defaults are the design values (ADR-012 to ADR-015).

    ``v_max_frac`` and ``v_init_frac`` are fractions of each dimension's range (upper - lower).
    The patience rule is off by default; when on, the run stops once gbest has improved by less than
    ``patience_tol`` over ``patience_iterations`` iterations.
    """

    n_particles: int = 10
    max_iter: int = 20
    w: float = 0.7298
    c1: float = 1.49618
    c2: float = 1.49618
    v_max_frac: float = 0.2
    v_init_frac: float = 0.1
    boundary: str = "absorb"
    topology: str = "gbest"
    update: str = "synchronous"
    patience_enabled: bool = False
    patience_tol: float = 1e-4
    patience_iterations: int = 5
