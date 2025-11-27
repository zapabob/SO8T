"""
SO8T残差アダプタモジュール

Boreas-Phi3.5の挙動を保ったままSO(8)幾何構造を注入するための残差アダプタ実装。

理論的設計:
- 射影P: R^{hidden_size} -> R^8
- SO(8)回転R: so(8)パラメータから行列指数で生成
- 逆射影P^T: R^8 -> R^{hidden_size}
- 残差注入: h' = h + λ * (P^T R(α) P h - P^T P h)

λ=0で元モデル完全一致を保証しつつ、中間層のみに適用。
Alpha Gateとの連携でα=0で恒等、α=Φ^(-2)で目標値となる。
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
import math


class SO8TAdapter(nn.Module):
    """
    SO8T残差アダプタ

    射影・SO(8)回転・逆射影を組み合わせた残差アダプタ。
    λ=0で元モデルの出力と完全に一致することを保証。

    数式:
        h' = h + λ * Δh
        Δh = P^T * R(α) * P * h - P^T * P * h
        R(α) = exp(A) where A ∈ so(8) (skew-symmetric)

    Args:
        hidden_size: 隠れ層の次元 (Phi-3.5-mini: 2048)
        so8_dim: SO(8)の次元 (固定: 8)
        init_strength: 残差強度λの初期値 (デフォルト: 0.0)
        use_matrix_exp: 行列指数を使用するか (True: torch.matrix_exp, False: 近似)
    """

    def __init__(
        self,
        hidden_size: int,
        so8_dim: int = 8,
        init_strength: float = 0.0,
        use_matrix_exp: bool = True,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.so8_dim = so8_dim
        self.use_matrix_exp = use_matrix_exp

        # 射影 P: R^{hidden_size} -> R^8 (バイアスなし)
        self.proj = nn.Linear(hidden_size, so8_dim, bias=False)

        # so(8)パラメータ: skew-symmetric行列の自由パラメータ
        # A ∈ so(8): A^T = -A, Aの独立パラメータ数は 8*7/2 = 28
        # パラメータは上三角部分のみを学習
        self.A_params = nn.Parameter(torch.zeros(so8_dim, so8_dim))

        # 残差強度 λ: 初期値0で元モデル完全一致を保証
        self.strength = nn.Parameter(torch.tensor(init_strength, dtype=torch.float32))

    def _build_skew_symmetric_matrix(self) -> Tensor:
        """
        so(8)パラメータからskew-symmetric行列を構築

        A_paramsの上三角部分を使って A = A_params - A_params^T を構築。

        Returns:
            A: (8, 8) skew-symmetric行列
        """
        # A_paramsの上三角部分をコピー
        A_upper = torch.triu(self.A_params, diagonal=1)
        # skew-symmetric行列: A = A_upper - A_upper^T
        A = A_upper - A_upper.transpose(-2, -1)
        return A

    def _compute_rotation_matrix(self, alpha: Union[float, Tensor]) -> Tensor:
        """
        SO(8)回転行列を計算

        R(α) = exp(α * A) を計算。
        α=0で恒等行列となる。

        Args:
            alpha: Alpha Gate値 (0.0-1.0の範囲)

        Returns:
            R: (8, 8) SO(8)回転行列
        """
        A = self._build_skew_symmetric_matrix()

        # αがスカラーまたはテンソルの場合に対応
        if isinstance(alpha, (float, int)):
            alpha = torch.tensor(alpha, dtype=A.dtype, device=A.device)
        elif isinstance(alpha, Tensor) and alpha.dim() > 0:
            # バッチ次元がある場合、最初の要素を使用（Alpha Gateは通常スカラー）
            alpha = alpha.flatten()[0] if alpha.numel() > 0 else torch.tensor(0.0, dtype=A.dtype, device=A.device)

        # α * A
        alpha_A = alpha * A

        if self.use_matrix_exp:
            # 正確な行列指数
            R = torch.matrix_exp(alpha_A)
        else:
            # 近似: R ≈ I + α*A + (α*A)^2/2 (低次近似)
            I = torch.eye(self.so8_dim, dtype=A.dtype, device=A.device)
            alpha_A_sq = torch.matmul(alpha_A, alpha_A)
            R = I + alpha_A + 0.5 * alpha_A_sq

        return R

    def forward(self, h: Tensor, alpha: Union[float, Tensor, None] = None) -> Tensor:
        """
        残差アダプタ適用

        Args:
            h: 隠れ状態 (..., hidden_size)
            alpha: Alpha Gate値 (Noneの場合は0として扱う)

        Returns:
            h_prime: 残差適用後の隠れ状態 (..., hidden_size)
        """
        if alpha is None:
            alpha = 0.0

        # 1. 隠れ状態を8次元に射影: z = P * h
        z = self.proj(h)  # (..., 8)

        # 2. SO(8)回転行列を計算: R(α)
        R = self._compute_rotation_matrix(alpha)  # (8, 8)

        # 3. 回転適用: z_rot = R * z
        # 注意: PyTorchのmatmulでは最後の2次元が行列積
        if z.dim() == 1:
            # 1Dテンソルの場合: (8,) -> (8,)
            z_rot = torch.matmul(R, z)
        else:
            # バッチ/シーケンス次元がある場合: (..., 8)
            z_rot = torch.matmul(z, R.t())  # R.t()で転置

        # 4. 差分計算: Δz = R * z - z
        delta_z = z_rot - z  # (..., 8)

        # 5. 差分を元の隠れ空間に戻す: Δh = P^T * Δz
        # proj.weight: (8, hidden_size) の転置が逆射影
        delta_h = torch.matmul(delta_z, self.proj.weight)  # (..., hidden_size)

        # 6. 残差注入: h' = h + λ * Δh
        h_prime = h + self.strength * delta_h

        return h_prime

    def get_orthogonality_error(self) -> Tensor:
        """
        回転行列の直交性誤差を計算

        R^T * R - I のFrobeniusノルムを返す。

        Returns:
            直交性誤差 (スカラー)
        """
        if not hasattr(self, '_last_rotation_matrix'):
            # デフォルトのalpha=0で計算
            self._last_rotation_matrix = self._compute_rotation_matrix(0.0)

        R = self._last_rotation_matrix
        I = torch.eye(self.so8_dim, dtype=R.dtype, device=R.device)
        error = torch.norm(R.t() @ R - I, p='fro')
        return error

    def get_determinant_error(self) -> Tensor:
        """
        回転行列の行列式誤差を計算

        |det(R) - 1| を返す。SO(8)ではdet(R) = 1。

        Returns:
            行列式誤差 (スカラー)
        """
        if not hasattr(self, '_last_rotation_matrix'):
            # デフォルトのalpha=0で計算
            self._last_rotation_matrix = self._compute_rotation_matrix(0.0)

        R = self._last_rotation_matrix
        det = torch.det(R)
        error = torch.abs(det - 1.0)
        return error

    def update_rotation_matrix(self, alpha: Union[float, Tensor, None] = None):
        """
        回転行列を更新（ログ出力用）

        Args:
            alpha: Alpha Gate値
        """
        if alpha is None:
            alpha = 0.0
        self._last_rotation_matrix = self._compute_rotation_matrix(alpha)

    @torch.no_grad()
    def export_weights(self, original_weight: Tensor, layer_idx: int) -> Tensor:
        """
        アダプタの効果を重み行列に吸収

        推論時やエクスポート時に使用。
        残差効果を元の重み行列に組み込む。

        Args:
            original_weight: 元の重み行列
            layer_idx: レイヤーインデックス（ログ用）

        Returns:
            absorbed_weight: アダプタ効果を吸収した重み行列
        """
        # アダプタの効果を吸収した重み行列を計算
        # W' = W + λ * P^T * R * P
        R = self._compute_rotation_matrix(0.0)  # デフォルトalphaで計算
        P = self.proj.weight  # (8, hidden_size)
        absorbed_adapter = self.strength * P.t() @ R @ P  # (hidden_size, hidden_size)

        # 元の重みに追加
        absorbed_weight = original_weight + absorbed_adapter

        return absorbed_weight
