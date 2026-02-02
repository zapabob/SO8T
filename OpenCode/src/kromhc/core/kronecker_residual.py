"""
クロネッカー積残差行列モジュール。

KromHCの中核となる、クロネッカー積を用いた残差行列の実装。

Features:
    - 因子行列のクロネッカー積による残差行列構築
    - 多様体制約の適用
    - パラメータ効率の良いO(n²C)複雑度

References:
    Zhou et al. (2026) arXiv:2601.21579
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .doubly_stochastic import (
    doubly_stochastic_projection,
    DoublyStochasticProjector,
    SinkhornConfig,
)
from ...utils.errors import ModelDimensionError, handle_kromhc_errors
from ...utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class KroneckerResidualConfig:
    """クロネッカー積残差行列設定。

    Attributes:
        n_streams: 残差ストリーム数
        hidden_dim: 隠れ次元
        factor_dim: 因子行列の次元（Noneの場合はn_streams）
        use_bias: バイアス項を使用するか
        learnable_projection: 射影を学習可能にするか
        sinkhorn_iter: Sinkhorn反復回数
    """

    n_streams: int
    hidden_dim: int
    factor_dim: Optional[int] = None
    use_bias: bool = True
    learnable_projection: bool = False
    sinkhorn_iter: int = 50


class KroneckerResidualMatrix(nn.Module):
    """クロネッカー積を用いた残差行列。

    因子行列A (k×k) とB (C×C) のクロネcker積を用いて、
    大きな残差行列R = A ⊗ B を構築する。

    Attributes:
        config: 設定
        factor_A: 行方向の因子行列
        factor_B: 列方向の因子行列
        bias: バイアス項
    """

    def __init__(
        self,
        config: KroneckerResidualConfig,
    ) -> None:
        super().__init__()
        self.config = config

        factor_dim = config.factor_dim or config.n_streams

        self.factor_A = nn.Parameter(torch.randn(config.n_streams, factor_dim))
        self.factor_B = nn.Parameter(torch.randn(config.hidden_dim, config.hidden_dim))

        if config.use_bias:
            self.bias = nn.Parameter(torch.zeros(config.n_streams * config.hidden_dim))
        else:
            self.register_parameter("bias", None)

        if config.learnable_projection:
            self.projector_A = DoublyStochasticProjector(
                config.n_streams,
                config=SinkhornConfig(max_iter=config.sinkhorn_iter),
                learnable=True,
            )
            self.projector_B = DoublyStochasticProjector(
                config.hidden_dim,
                config=SinkhornConfig(max_iter=config.sinkhorn_iter),
                learnable=True,
            )
        else:
            self.register_buffer(
                "projector_A",
                DoublyStochasticProjector(
                    config.n_streams,
                    config=SinkhornConfig(max_iter=config.sinkhorn_iter),
                    learnable=False,
                ).matrix,
            )
            self.register_buffer(
                "projector_B",
                DoublyStochasticProjector(
                    config.hidden_dim,
                    config=SinkhornConfig(max_iter=config.sinkhorn_iter),
                    learnable=False,
                ).matrix,
            )

    def forward(self) -> torch.Tensor:
        """残差行列を計算する。

        Returns:
            残差行列 (n*C, n*C)

        Example:
            >>> config = KroneckerResidualConfig(n_streams=4, hidden_dim=256)
            >>> residual = KroneckerResidualMatrix(config)
            >>> R = residual()
            >>> print(R.shape)
            torch.Size([1024, 1024])
        """
        A = self._get_projected_A()
        B = self._get_projected_B()

        R = torch.kron(A, B)

        if self.bias is not None:
            R = R + torch.diag(self.bias)

        return R

    def _get_projected_A(self) -> torch.Tensor:
        """射影された因子行列Aを取得する。

        Returns:
            射影された行列A
        """
        if hasattr(self, "projector_A"):
            return self.projector_A()
        return self.projector_A

    def _get_projected_B(self) -> torch.Tensor:
        """射影された因子行列Bを取得する。

        Returns:
            射影された行列B
        """
        if hasattr(self, "projector_B"):
            return self.projector_B()
        return self.projector_B

    @property
    def n_streams(self) -> int:
        """残差ストリーム数を返す。"""
        return self.config.n_streams

    @property
    def hidden_dim(self) -> int:
        """隠れ次元を返す。"""
        return self.config.hidden_dim

    @property
    def output_dim(self) -> int:
        """出力次元を返す。

        Returns:
            n_streams * hidden_dim
        """
        return self.n_streams * self.hidden_dim

    @property
    def complexity(self) -> str:
        """計算複雑度を返す。

        Returns:
            複雑度の文字列表現
        """
        return f"O(n²C) = O({self.n_streams}²×{self.hidden_dim})"


class KroneckerResidualStream(nn.Module):
    """クロネッカー積を用いた残差ストリーム。

    入力テンソルに対して残差接続を適用する。

    Attributes:
        config: 設定
        residual_matrix: 残差行列モジュール
    """

    def __init__(
        self,
        config: KroneckerResidualConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.residual_matrix = KroneckerResidualMatrix(config)

    def forward(
        self,
        x: torch.Tensor,
        *,
        identity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """残差接続を適用する。

        Args:
            x: 入力テンソル (batch, *, n_streams*hidden_dim)
            identity: アイデンティティマッピング（Noneの場合はx）

        Returns:
            残差適用後のテンソル

        Example:
            >>> config = KroneckerResidualConfig(n_streams=4, hidden_dim=256)
            >>> stream = KroneckerResidualStream(config)
            >>> x = torch.randn(2, 1024)
            >>> output = stream(x)
            >>> print(output.shape)
            torch.Size([2, 1024])
        """
        if identity is None:
            identity = x

        residual = self.residual_matrix()
        output = F.linear(x, residual, bias=None) + identity

        return output


def verify_doubly_stochastic(matrix: torch.Tensor, *, atol: float = 1e-4) -> bool:
    """行列が二重確率性を持っているか確認する。

    Args:
        matrix: 確認する行列
        atol: 許容誤差

    Returns:
        二重確率性を持っているか
    """
    row_sum = matrix.sum(dim=-1)
    col_sum = matrix.sum(dim=-2)

    is_row_stochastic = torch.allclose(
        row_sum,
        torch.ones_like(row_sum),
        atol=atol,
    )
    is_col_stochastic = torch.allclose(
        col_sum,
        torch.ones_like(col_sum),
        atol=atol,
    )

    return is_row_stochastic and is_col_stochastic


def compute_kron_dim(n: int, C: int) -> int:
    """クロネッカー積の次元を計算する。

    Args:
        n: ストリーム数
        C: 隠れ次元

    Returns:
        出力次元
    """
    return n * C
