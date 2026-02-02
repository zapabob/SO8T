"""KromHC core modules."""
from .doubly_stochastic import (
    SinkhornConfig,
    doubly_stochastic_projection,
    sinkhorn_knopp_iteration,
    DoublyStochasticProjector,
)
from .kronecker_residual import (
    KroneckerResidualConfig,
    KroneckerResidualMatrix,
    KroneckerResidualStream,
    verify_doubly_stochastic,
    compute_kron_dim,
)
from .manifold_constraint import (
    ManifoldConstraint,
    ManifoldConstraintConfig,
    enforce_birkhoff,
)

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
]
