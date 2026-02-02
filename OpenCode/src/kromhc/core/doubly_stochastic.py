"""
二重確率行列生成モジュール。

Sinkhorn-Knoppアルゴリズムを用いて、行列を二重確率多様体に射影する。

Features:
    - Sinkhorn-Knopp反復による二重確率化
    - 収束監視とエラー処理
    - GPUサポート

References:
    Zhou et al. (2026) arXiv:2601.21579
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ...utils.errors import (
    ConvergenceError,
    MatrixConstraintError,
    handle_kromhc_errors,
)
from ...utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SinkhornConfig:
    """Sinkhorn反復設定。

    Attributes:
        max_iter: 最大反復回数
        tolerance: 収束判定の許容誤差
        epsilon: ゼロ除算防止の微小値
        warn_on_non_convergence: 非収束時の警告出力
    """

    max_iter: int = 100
    tolerance: float = 1e-5
    epsilon: float = 1e-10
    warn_on_non_convergence: bool = True


def sinkhorn_knopp_iteration(
    matrix: torch.Tensor,
    *,
    config: Optional[SinkhornConfig] = None,
) -> torch.Tensor:
    """Sinkhorn-Knopp反復により行列を二重確率化する。

    二重確率行列とは、行和と列和がすべて1になる行列である。
    この関数は、入力行列をSinkhorn反復によって二重確率行列に射影する。

    Args:
        matrix: 入力行列 (n, n)
        config: Sinkhorn反復設定

    Returns:
        二重確率行列

    Raises:
        ConvergenceError: 反復が収束しなかった場合
        MatrixConstraintError: 入力行列が正方行列でない場合

    Example:
        >>> import torch
        >>> matrix = torch.rand(4, 4)
        >>> ds_matrix = sinkhorn_knopp_iteration(matrix)
        >>> assert torch.allclose(ds_matrix.sum(dim=-1), torch.ones(4), atol=1e-5)
    """
    if config is None:
        config = SinkhornConfig()

    if matrix.dim() != 2 or matrix.shape[0] != matrix.shape[1]:
        raise MatrixConstraintError(
            "square_matrix",
            matrix.shape[0] if matrix.dim() == 2 else matrix.dim(),
            matrix.shape[1] if matrix.dim() == 2 else 0,
        )

    if matrix.shape[0] == 1:
        return torch.ones(1, 1, device=matrix.device, dtype=matrix.dtype)

    eps = config.epsilon

    for i in range(config.max_iter):
        row_sum = matrix.sum(dim=-1, keepdim=True)
        row_sum = torch.where(row_sum < eps, eps, row_sum)
        matrix = matrix / row_sum

        col_sum = matrix.sum(dim=-2, keepdim=True)
        col_sum = torch.where(col_sum < eps, eps, col_sum)
        matrix = matrix / col_sum

        if (i + 1) % 10 == 0:
            row_error = torch.max(torch.abs(matrix.sum(dim=-1) - 1)).item()
            col_error = torch.max(torch.abs(matrix.sum(dim=-2) - 1)).item()
            max_error = max(row_error, col_error)

            if max_error < config.tolerance:
                logger.debug(f"Sinkhorn収束: iter={i + 1}, error={max_error:.2e}")
                return matrix

    row_error = torch.max(torch.abs(matrix.sum(dim=-1) - 1)).item()
    col_error = torch.max(torch.abs(matrix.sum(dim=-2) - 1)).item()
    final_error = max(row_error, col_error)

    if config.warn_on_non_convergence:
        logger.warning(
            f"Sinkhorn非収束: max_iter={config.max_iter}, final_error={final_error:.2e}"
        )

    raise ConvergenceError(
        config.max_iter,
        final_error,
        matrix_shape=matrix.shape,
    )


@handle_kromhc_errors
def doubly_stochastic_projection(
    matrix: torch.Tensor,
    *,
    max_iter: int = 100,
    tolerance: float = 1e-5,
    epsilon: float = 1e-10,
) -> torch.Tensor:
    """行列を二重確率多様体に射影する。

    Args:
        matrix: 入力行列
        max_iter: 最大反復回数
        tolerance: 収束判定の許容誤差
        epsilon: ゼロ除算防止の微小値

    Returns:
        二重確率行列
    """
    config = SinkhornConfig(
        max_iter=max_iter,
        tolerance=tolerance,
        epsilon=epsilon,
        warn_on_non_convergence=True,
    )

    return sinkhorn_knopp_iteration(matrix.abs(), config=config)


class DoublyStochasticProjector(nn.Module):
    """二重確率射影を行うニューラルネットワークモジュール。

    Attributes:
        config: Sinkhorn反復設定
        learnable: 射影行列が学習可能かどうか
    """

    def __init__(
        self,
        size: int,
        *,
        config: Optional[SinkhornConfig] = None,
        learnable: bool = False,
    ) -> None:
        super().__init__()
        self.size = size
        self.config = config or SinkhornConfig()
        self.learnable = learnable

        if learnable:
            self.matrix = nn.Parameter(torch.randn(size, size))
        else:
            self.register_buffer("matrix", torch.randn(size, size))

    def forward(self) -> torch.Tensor:
        """出力を計算する。

        Returns:
            二重確率行列
        """
        return doubly_stochastic_projection(
            self.matrix,
            max_iter=self.config.max_iter,
            tolerance=self.config.tolerance,
            epsilon=self.config.epsilon,
        )

    @property
    def is_valid(self) -> bool:
        """行列が二重確率性を持っているか確認する。

        Returns:
            妥当性
        """
        with torch.no_grad():
            mat = self()
            row_sum = mat.sum(dim=-1)
            col_sum = mat.sum(dim=-2)
            return torch.allclose(
                row_sum, torch.ones_like(row_sum), atol=1e-4
            ) and torch.allclose(col_sum, torch.ones_like(col_sum), atol=1e-4)
