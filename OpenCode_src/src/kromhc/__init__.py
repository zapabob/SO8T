"""KromHC module."""
from .core import (
    SinkhornConfig,
    doubly_stochastic_projection,
    sinkhorn_knopp_iteration,
    DoublyStochasticProjector,
    KroneckerResidualConfig,
    KroneckerResidualMatrix,
    KroneckerResidualStream,
    verify_doubly_stochastic,
    compute_kron_dim,
    ManifoldConstraint,
    ManifoldConstraintConfig,
    enforce_birkhoff,
)
from .layers import KromHCAttentionLayer, KromHCMLPLayer
from .utils import init_kronecker_factors, init_kromhc_module

__all__ = [
    "SinkhornConfig",
    "doubly_stochastic_projection",
    "sinkhorn_knopp_iteration",
    "DoublyStochasticProjector",
    "KroneckerResidualConfig",
    "KroneckerResidualMatrix",
    "KroneckerResidualStream",
    "verify_doubly_stochastic",
    "compute_kron_dim",
    "ManifoldConstraint",
    "ManifoldConstraintConfig",
    "enforce_birkhoff",
    "KromHCAttentionLayer",
    "KromHCMLPLayer",
    "init_kronecker_factors",
    "init_kromhc_module",
]
