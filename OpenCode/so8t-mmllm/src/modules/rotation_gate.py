"""
SO(8)群回転ゲート実装
SDPA出力後の8次元ブロック直交回転を適用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


def make_skew_symmetric(t: torch.Tensor) -> torch.Tensor:
    """
    歪対称行列を作成
    
    Args:
        t: 入力テンソル [*, 8, 8]
        
    Returns:
        歪対称行列 A - A^T
    """
    return t - t.transpose(-1, -2)


def block_expm(A: torch.Tensor, max_iter: int = 10) -> torch.Tensor:
    """
    ブロック単位での行列指数関数（数値安定化版）
    
    Args:
        A: 歪対称行列 [*, 8, 8]
        max_iter: 最大反復回数
        
    Returns:
        直交回転行列 R = exp(A)
    """
    device = A.device
    dtype = A.dtype
    
    # 単位行列
    I = torch.eye(8, device=device, dtype=dtype)
    
    # 数値安定化のための正則化
    A_reg = A + 1e-6 * I.unsqueeze(0)
    
    # スケール・スクエア法
    # ノルムが大きすぎる場合はスケールダウン
    norm = torch.norm(A_reg, dim=(-2, -1), keepdim=True)
    scale = torch.clamp(norm, max=1.0)
    A_scaled = A_reg / (scale + 1e-8)
    
    # テイラー展開による近似
    R = I.unsqueeze(0).expand_as(A_scaled)
    A_power = A_scaled.clone()
    
    for i in range(1, max_iter + 1):
        R = R + A_power / math.factorial(i)
        A_power = torch.matmul(A_power, A_scaled)
    
    # スケール・スクエア法の逆操作
    scale_int = scale.int()
    for _ in range(scale_int.max().item()):
        R = torch.matmul(R, R)
    
    return R


def apply_block_rotation(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    8次元ブロック単位での回転を適用
    
    Args:
        x: 入力テンソル [B, T, D] (D % 8 == 0)
        theta: 回転パラメータ [D//8, 8, 8]
        
    Returns:
        回転後のテンソル [B, T, D]
    """
    B, T, D = x.shape
    assert D % 8 == 0, f"Dimension {D} must be divisible by 8"
    
    # 8次元ブロックに分割
    xv = x.view(B, T, D // 8, 8)  # [B, T, D//8, 8]
    
    # 歪対称行列を作成
    A = make_skew_symmetric(theta)  # [D//8, 8, 8]
    
    # 回転行列を計算
    R = block_expm(A)  # [D//8, 8, 8]
    
    # 回転を適用
    y = torch.einsum('btne,neo->btno', xv, R)  # [B, T, D//8, 8]
    
    # 元の形状に戻す
    return y.reshape(B, T, D)


class SO8TRotationGate(nn.Module):
    """
    SO(8)群回転ゲートモジュール
    SDPA出力後に適用される回転変換
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_blocks: Optional[int] = None,
        init_scale: float = 0.1,
        learnable: bool = True
    ):
        """
        Args:
            hidden_size: 隠れ層のサイズ（8の倍数である必要がある）
            num_blocks: ブロック数（Noneの場合はhidden_size//8）
            init_scale: 初期化スケール
            learnable: 学習可能かどうか
        """
        super().__init__()
        
        assert hidden_size % 8 == 0, f"hidden_size {hidden_size} must be divisible by 8"
        
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks or hidden_size // 8
        self.init_scale = init_scale
        self.learnable = learnable
        
        # 回転パラメータ（歪対称行列の上三角部分）
        if learnable:
            # 上三角部分のみを学習（歪対称性を保証）
            self.theta = nn.Parameter(
                torch.randn(self.num_blocks, 8, 8) * init_scale
            )
        else:
            # 固定値
            self.register_buffer(
                'theta', 
                torch.randn(self.num_blocks, 8, 8) * init_scale
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向き計算
        
        Args:
            x: 入力テンソル [B, T, D]
            
        Returns:
            回転後のテンソル [B, T, D]
        """
        return apply_block_rotation(x, self.theta)
    
    def get_rotation_matrices(self) -> torch.Tensor:
        """
        現在の回転行列を取得
        
        Returns:
            回転行列 [num_blocks, 8, 8]
        """
        A = make_skew_symmetric(self.theta)
        return block_expm(A)
    
    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, num_blocks={self.num_blocks}, learnable={self.learnable}"


class SO8TMultiHeadAttention(nn.Module):
    """
    SO(8)群回転ゲート付きマルチヘッドアテンション
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        rotation_enabled: bool = True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout
        self.rotation_enabled = rotation_enabled
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        assert hidden_size % 8 == 0, "hidden_size must be divisible by 8"
        
        # 標準的なアテンション層
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # SO(8)回転ゲート
        if rotation_enabled:
            self.rotation_gate = SO8TRotationGate(hidden_size)
        else:
            self.rotation_gate = None
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        前向き計算
        
        Args:
            hidden_states: 入力状態 [B, T, D]
            attention_mask: アテンションマスク
            position_ids: 位置ID
            past_key_value: 過去のキー・バリュー
            output_attentions: アテンション重みを出力するか
            use_cache: キャッシュを使用するか
            
        Returns:
            (出力状態, アテンション重み, キー・バリュー)
        """
        B, T, D = hidden_states.shape
        
        # Q, K, Vの計算
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # ヘッドに分割
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # アテンション計算
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # アテンション適用
        attn_output = torch.matmul(attn_weights, v)
        
        # ヘッドを結合
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        
        # 出力投影
        attn_output = self.o_proj(attn_output)
        
        # SO(8)回転ゲート適用
        if self.rotation_gate is not None:
            attn_output = self.rotation_gate(attn_output)
        
        return attn_output, attn_weights if output_attentions else None, None
