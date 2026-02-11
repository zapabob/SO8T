"""
Manifold constraints for KromHC.
Applies doubly-stochastic projection along tensor modes.
References:
    Zhou et al. (2026) arXiv:2601.21579
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from .doubly_stochastic import doubly_stochastic_projection
from ...utils.errors import MatrixConstraintError
from ...utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ManifoldConstraintConfig:
    """Configuration for manifold constraints.
    Attributes:
        max_iter: Sinkhorn iterations.
        tolerance: convergence tolerance.
        epsilon: numerical epsilon.
    """

    max_iter: int = 100
    tolerance: float = 1e-5
    epsilon: float = 1e-10


class ManifoldConstraint:
    """Applies manifold constraints to residual matrices."""

    def __init__(self, config: Optional[ManifoldConstraintConfig] = None) -> None:
        self.config = config or ManifoldConstraintConfig()

    def project(self, matrix: torch.Tensor) -> torch.Tensor:
        """Project a square matrix onto the Birkhoff polytope."""
        if matrix.dim() != 2 or matrix.shape[0] != matrix.shape[1]:
            raise MatrixConstraintError(
                "square_matrix",
                matrix.shape[0] if matrix.dim() == 2 else matrix.dim(),
                matrix.shape[1] if matrix.dim() == 2 else 0,
            )
        return doubly_stochastic_projection(
            matrix,
            max_iter=self.config.max_iter,
            tolerance=self.config.tolerance,
            epsilon=self.config.epsilon,
        )

    def project_modes(
        self,
        tensor: torch.Tensor,
        modes: Iterable[int],
    ) -> torch.Tensor:
        """Apply projection along specified modes (by flattening each mode)."""
        output = tensor
        for mode in modes:
            dim = output.shape[mode]
            if dim <= 1:
                continue
            perm = list(range(output.dim()))
            perm[0], perm[mode] = perm[mode], perm[0]
            reshaped = output.permute(perm).contiguous().view(dim, -1)
            projected = self.project(reshaped)
            output = projected.view([dim] + list(output.shape[:mode]) + list(output.shape[mode + 1 :]))
            output = output.permute(perm)
        return output


def enforce_birkhoff(matrix: torch.Tensor, config: Optional[ManifoldConstraintConfig] = None) -> torch.Tensor:
    """Convenience wrapper for projecting a matrix onto the Birkhoff polytope."""
    return ManifoldConstraint(config).project(matrix)
