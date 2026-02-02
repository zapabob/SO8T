"""
厳密なSO(8)群回転ゲート実装

既存のSO8TRotationGateをベースに拡張し、右側からの作用と重み行列への吸収機能を実装。
これにより、既存エコシステム（HF Transformers, ONNX, TensorRT, llama.cpp, vLLM等）と
互換性を保ちつつ、理論的一貫性を確保する。

理論的設計:
- ブロック対角構造: D=8k次元をk個の8次元ブロックに分割
- 右側からの作用: Q = X R W_Q → W_Q' = R W_Qとして合成可能
- RoPEとの非可換性を保ちつつ、外部からは通常の線形変換に見える
- 訓練時: SO(8)パラメータを明示的に扱い、直交性・行列式制約を保持
- 推論/エクスポート時: 重み行列に吸収して標準形式に変換
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor


class StrictSO8RotationGate(nn.Module):
    """
    厳密なSO(8)群回転ゲート
    
    既存のSO8TRotationGateをベースに拡張し、以下を実装:
    1. 右側からの作用（X R → R W_Qとして合成可能）
    2. 重み行列への吸収機能（export_weights()）
    3. 既存エコシステムとの互換性
    
    理論的設計:
    - 隠れ次元Dを8k次元とみなし、各トークンの表現ベクトルx ∈ R^Dを
      k個の8次元ブロックの直和として分解: x = (x^(1), x^(2), ..., x^(k))
    - SO(8)群回転ゲートは、各ブロックに対しR^(j) ∈ SO(8)を作用させる
      ブロック対角行列: R = diag(R^(1), R^(2), ..., R^(k)) ∈ SO(D)
    - 右側からの作用により、Q = X R W_Q = X (R W_Q)となり、
      W_Q' = R W_Qとして重み行列に吸収可能
    """
    
    def __init__(
        self,
        hidden_size: int,
        use_cayley: bool = True,
        orthogonal_regularization: float = 1e-3,
    ):
        """
        Args:
            hidden_size: 隠れ層サイズ（8の倍数である必要がある）
            use_cayley: Cayley変換を使用するかどうか（True: Cayley, False: 指数写像）
            orthogonal_regularization: 直交性正則化の強度
        """
        super().__init__()
        
        if hidden_size % 8 != 0:
            raise ValueError(f"hidden_size must be divisible by 8, got {hidden_size}")
        
        self.hidden_size = hidden_size
        self.num_blocks = hidden_size // 8
        self.use_cayley = use_cayley
        self.orthogonal_reg = orthogonal_regularization
        
        # 歪対称行列パラメータ (各8x8ブロック)
        # 歪対称行列は28個の自由パラメータ (8x8の上三角 - 対角)
        self.theta = nn.Parameter(torch.randn(self.num_blocks, 8, 8) * 0.01)
    
    def _cayley_rotation(self, theta: Tensor) -> Tensor:
        """
        Cayley変換: 歪対称行列を直交行列に変換
        R = (I - 0.5*A)^{-1} (I + 0.5*A)
        
        Args:
            theta: [num_blocks, 8, 8] 歪対称行列パラメータ
        
        Returns:
            R: [num_blocks, 8, 8] 直交回転行列（det(R) = 1を満たす）
        """
        # 歪対称行列を作成: A = theta - theta^T
        skew = theta - theta.transpose(-1, -2)
        
        # I - 0.5*A と I + 0.5*A
        eye = torch.eye(8, device=theta.device, dtype=theta.dtype)
        eye = eye.unsqueeze(0).expand(self.num_blocks, 8, 8)
        
        a = eye - 0.5 * skew
        b = eye + 0.5 * skew
        
        # (I - 0.5*A)^{-1} (I + 0.5*A)
        rot = torch.linalg.solve(a, b)
        
        return rot
    
    def _exp_rotation(self, theta: Tensor) -> Tensor:
        """
        行列指数関数: exp(A) where A is skew-symmetric
        SO(8)群のリー代数（歪対称行列）からSO(8)群への指数写像
        
        Args:
            theta: [num_blocks, 8, 8] 歪対称行列パラメータ
        
        Returns:
            R: [num_blocks, 8, 8] 直交回転行列（det(R) = 1を満たす）
        """
        # 歪対称行列を作成
        skew = theta - theta.transpose(-1, -2)
        
        # 行列指数関数: exp(skew) ∈ SO(8)
        rot = torch.matrix_exp(skew)
        
        return rot
    
    def get_rotation_matrix(self) -> Tensor:
        """
        SO(8)回転行列を生成
        
        Returns:
            R: [num_blocks, 8, 8] 直交回転行列
        """
        if self.use_cayley:
            return self._cayley_rotation(self.theta)
        else:
            return self._exp_rotation(self.theta)
    
    def forward(self, x: Tensor, apply_right: bool = True) -> Tensor:
        """
        SO(8)回転を適用
        
        右側からの作用（apply_right=True）:
        - y = x @ R（ブロック単位で適用）
        - これにより、Q = X R W_Q = X (R W_Q)となり、W_Q' = R W_Qとして合成可能
        
        左側からの作用（apply_right=False）:
        - y = R @ x（既存実装との互換性のため）
        
        Args:
            x: [batch_size, seq_len, hidden_size] 入力テンソル
            apply_right: Trueなら右側からの作用、Falseなら左側からの作用
        
        Returns:
            y: [batch_size, seq_len, hidden_size] 回転後のテンソル
        """
        batch_size, seq_len, hidden_size = x.shape
        assert hidden_size == self.hidden_size
        
        # 回転行列を生成
        rot = self.get_rotation_matrix()  # [num_blocks, 8, 8]
        
        if apply_right:
            # 右側からの作用: y = x @ R
            # ブロック単位で回転適用
            x_blocks = x.view(batch_size, seq_len, self.num_blocks, 8)
            # アインシュタイン記法: [B, T, N, 8] @ [N, 8, 8] -> [B, T, N, 8]
            # 右側からの作用: x_blocks @ rot
            y_blocks = torch.einsum('btni,nio->btno', x_blocks, rot)
            y = y_blocks.reshape(batch_size, seq_len, hidden_size)
        else:
            # 左側からの作用: y = R @ x（既存実装との互換性）
            x_blocks = x.view(batch_size, seq_len, self.num_blocks, 8)
            # 左側からの作用: rot @ x_blocks
            y_blocks = torch.einsum('nio,btni->btno', rot, x_blocks)
            y = y_blocks.reshape(batch_size, seq_len, hidden_size)
        
        return y
    
    def export_weights(self, weight_matrix: Tensor) -> Tensor:
        """
        重み行列にSO(8)回転を吸収
        
        右側からの作用により、X R W = X (R W)となるため、
        W' = R Wとして重み行列に吸収可能。
        
        これにより、既存エコシステム（HF Transformers, ONNX, TensorRT等）からは
        単なる重み行列として扱える。
        
        Args:
            weight_matrix: [out_features, in_features] または [..., out_features, in_features]
                          元の重み行列
        
        Returns:
            absorbed_weight: [out_features, in_features] または [..., out_features, in_features]
                            SO(8)回転を吸収した重み行列
        """
        with torch.no_grad():
            rot = self.get_rotation_matrix()  # [num_blocks, 8, 8]
            
            # 重み行列の形状を確認
            original_shape = weight_matrix.shape
            if len(original_shape) == 2:
                # [out_features, in_features]
                out_features, in_features = original_shape
                if in_features != self.hidden_size:
                    raise ValueError(
                        f"weight_matrix in_features ({in_features}) must match "
                        f"hidden_size ({self.hidden_size})"
                    )
                
                # 重み行列をブロックに分割
                weight_blocks = weight_matrix.view(out_features, self.num_blocks, 8)
                # 各ブロックに右側から回転を適用: W' = W @ R
                # [out_features, num_blocks, 8] @ [num_blocks, 8, 8] -> [out_features, num_blocks, 8]
                absorbed_blocks = torch.einsum('oni,nio->ono', weight_blocks, rot)
                absorbed_weight = absorbed_blocks.reshape(out_features, in_features)
            else:
                # 高次元テンソルの場合（例: [..., out_features, in_features]）
                # 最後の2次元を処理
                *leading_dims, out_features, in_features = original_shape
                if in_features != self.hidden_size:
                    raise ValueError(
                        f"weight_matrix in_features ({in_features}) must match "
                        f"hidden_size ({self.hidden_size})"
                    )
                
                # フラット化して処理
                weight_flat = weight_matrix.view(-1, out_features, in_features)
                absorbed_flat = []
                for w in weight_flat:
                    w_blocks = w.view(out_features, self.num_blocks, 8)
                    absorbed_blocks = torch.einsum('oni,nio->ono', w_blocks, rot)
                    absorbed_flat.append(absorbed_blocks.reshape(out_features, in_features))
                absorbed_weight = torch.stack(absorbed_flat).view(*original_shape)
            
            return absorbed_weight
    
    def get_orthogonality_loss(self) -> Tensor:
        """
        直交性正則化損失
        ||R^T R - I||^2
        
        Returns:
            loss: スカラーテンソル
        """
        rot = self.get_rotation_matrix()
        
        # R^T @ R
        rtr = torch.einsum('nij,njk->nik', rot.transpose(-1, -2), rot)
        
        # I
        eye = torch.eye(8, device=rot.device, dtype=rot.dtype)
        eye = eye.unsqueeze(0).expand(self.num_blocks, 8, 8)
        
        # ||R^T R - I||^2
        loss = torch.mean((rtr - eye) ** 2)
        
        return loss * self.orthogonal_reg
    
    def get_determinant_loss(self) -> Tensor:
        """
        行列式制約損失
        (det(R) - 1)^2
        
        Returns:
            loss: スカラーテンソル
        """
        rot = self.get_rotation_matrix()
        
        # det(R) を計算
        det = torch.det(rot)  # [num_blocks]
        
        # (det(R) - 1)^2
        loss = torch.mean((det - 1.0) ** 2)
        
        return loss * self.orthogonal_reg
    
    def verify_orthogonality(self) -> Dict[str, float]:
        """
        直交性検証
        
        Returns:
            metrics: 検証メトリクス
        """
        with torch.no_grad():
            rot = self.get_rotation_matrix()
            
            # R^T @ R
            rtr = torch.einsum('nij,njk->nik', rot.transpose(-1, -2), rot)
            
            # I
            eye = torch.eye(8, device=rot.device, dtype=rot.dtype)
            eye = eye.unsqueeze(0).expand(self.num_blocks, 8, 8)
            
            # メトリクス
            max_error = torch.max(torch.abs(rtr - eye)).item()
            mean_error = torch.mean(torch.abs(rtr - eye)).item()
            
            # det(R) ≈ 1 の検証
            det = torch.det(rot)
            det_error = torch.mean(torch.abs(det - 1.0)).item()
            
            return {
                'max_orthogonality_error': max_error,
                'mean_orthogonality_error': mean_error,
                'determinant_error': det_error,
            }
    
    def verify_norm_preservation(self, x: Tensor) -> Dict[str, float]:
        """
        ノルム保存検証
        ||Rx|| = ||x||
        
        Args:
            x: [batch_size, seq_len, hidden_size] 入力テンソル
        
        Returns:
            metrics: 検証メトリクス
        """
        with torch.no_grad():
            y = self.forward(x, apply_right=True)
            
            # ノルム計算
            x_norm = torch.norm(x, dim=-1)
            y_norm = torch.norm(y, dim=-1)
            
            # 誤差
            norm_diff = torch.abs(x_norm - y_norm)
            max_error = torch.max(norm_diff).item()
            mean_error = torch.mean(norm_diff).item()
            
            return {
                'max_norm_error': max_error,
                'mean_norm_error': mean_error,
            }

