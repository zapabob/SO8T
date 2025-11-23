"""
SO8T回転ゲート層
8次元ブロック直交回転の実装
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class SO8TRotationGate(nn.Module):
    """
    SO(8)回転ゲート層
    
    主要機能:
    - 8次元ブロック直交回転の実装
    - 歪対称行列による回転生成
    - 等長写像保証（ノルム保存）
    - 直交性正則化損失
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
            use_cayley: Cayley変換を使用するかどうか
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
            R: [num_blocks, 8, 8] 直交回転行列
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
        
        Args:
            theta: [num_blocks, 8, 8] 歪対称行列パラメータ
        
        Returns:
            R: [num_blocks, 8, 8] 直交回転行列
        """
        # 歪対称行列を作成
        skew = theta - theta.transpose(-1, -2)
        
        # 行列指数関数
        rot = torch.matrix_exp(skew)
        
        return rot
    
    def forward(self, x: Tensor) -> Tensor:
        """
        SO(8)回転を適用
        
        Args:
            x: [batch_size, seq_len, hidden_size] 入力テンソル
        
        Returns:
            y: [batch_size, seq_len, hidden_size] 回転後のテンソル
        """
        batch_size, seq_len, hidden_size = x.shape
        assert hidden_size == self.hidden_size
        
        # 回転行列を生成
        if self.use_cayley:
            rot = self._cayley_rotation(self.theta)
        else:
            rot = self._exp_rotation(self.theta)
        
        # ブロック単位で回転適用
        x_blocks = x.view(batch_size, seq_len, self.num_blocks, 8)
        # アインシュタイン記法: [B, T, N, 8] @ [N, 8, 8] -> [B, T, N, 8]
        y_blocks = torch.einsum('btni,nio->btno', x_blocks, rot)
        y = y_blocks.reshape(batch_size, seq_len, hidden_size)
        
        return y
    
    def get_orthogonality_loss(self) -> Tensor:
        """
        直交性正則化損失
        ||R^T R - I||^2
        
        Returns:
            loss: スカラーテンソル
        """
        if self.use_cayley:
            rot = self._cayley_rotation(self.theta)
        else:
            rot = self._exp_rotation(self.theta)
        
        # R^T @ R
        rtr = torch.einsum('nij,njk->nik', rot.transpose(-1, -2), rot)
        
        # I
        eye = torch.eye(8, device=rot.device, dtype=rot.dtype)
        eye = eye.unsqueeze(0).expand(self.num_blocks, 8, 8)
        
        # ||R^T R - I||^2
        loss = torch.mean((rtr - eye) ** 2)
        
        return loss * self.orthogonal_reg
    
    def verify_orthogonality(self) -> dict:
        """
        直交性検証
        
        Returns:
            metrics: 検証メトリクス
        """
        with torch.no_grad():
            if self.use_cayley:
                rot = self._cayley_rotation(self.theta)
            else:
                rot = self._exp_rotation(self.theta)
            
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
    
    def verify_norm_preservation(self, x: Tensor) -> dict:
        """
        ノルム保存検証
        ||Rx|| = ||x||
        
        Args:
            x: [batch_size, seq_len, hidden_size] 入力テンソル
        
        Returns:
            metrics: 検証メトリクス
        """
        with torch.no_grad():
            y = self.forward(x)
            
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


class SO8TAttentionWrapper(nn.Module):
    """
    アテンション層にSO8T回転ゲートを追加するラッパー
    """
    
    def __init__(
        self,
        attention_module: nn.Module,
        hidden_size: int,
        apply_to_output: bool = True,
        apply_to_query: bool = False,
        apply_to_key: bool = False,
        apply_to_value: bool = False,
    ):
        """
        Args:
            attention_module: 元のアテンション層
            hidden_size: 隠れ層サイズ
            apply_to_output: 出力に回転を適用
            apply_to_query: クエリに回転を適用
            apply_to_key: キーに回転を適用
            apply_to_value: バリューに回転を適用
        """
        super().__init__()
        
        self.attention = attention_module
        self.apply_to_output = apply_to_output
        self.apply_to_query = apply_to_query
        self.apply_to_key = apply_to_key
        self.apply_to_value = apply_to_value
        
        # SO8T回転ゲート
        if apply_to_output:
            self.output_gate = SO8TRotationGate(hidden_size)
        if apply_to_query:
            self.query_gate = SO8TRotationGate(hidden_size)
        if apply_to_key:
            self.key_gate = SO8TRotationGate(hidden_size)
        if apply_to_value:
            self.value_gate = SO8TRotationGate(hidden_size)
    
    def forward(self, *args, **kwargs):
        """
        アテンション計算にSO8T回転を適用
        """
        # TODO: アテンション層の具体的な実装に合わせて調整が必要
        output = self.attention(*args, **kwargs)
        
        if self.apply_to_output:
            if isinstance(output, tuple):
                # (output, attention_weights, ...)形式
                hidden_states = output[0]
                hidden_states = self.output_gate(hidden_states)
                output = (hidden_states,) + output[1:]
            else:
                # outputのみ
                output = self.output_gate(output)
        
        return output
    
    def get_orthogonality_loss(self) -> Tensor:
        """
        全ゲートの直交性損失合計
        """
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        if self.apply_to_output:
            loss = loss + self.output_gate.get_orthogonality_loss()
        if self.apply_to_query:
            loss = loss + self.query_gate.get_orthogonality_loss()
        if self.apply_to_key:
            loss = loss + self.key_gate.get_orthogonality_loss()
        if self.apply_to_value:
            loss = loss + self.value_gate.get_orthogonality_loss()
        
        return loss

