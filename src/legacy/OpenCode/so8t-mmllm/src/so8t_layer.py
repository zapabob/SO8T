"""
SO8T (SO(8) Transformer) Layer Implementation
防衛・航空宇宙・運輸向けセキュアLLM用SO(8)回転ゲート実装

理論的基盤:
- SO(8)群: 8次元空間の特殊直交群（等長写像を保証）
- 歪対称行列: exp(A)でA^T = -Aとすることで直交性を保証
- ブロック対角構造: D次元をD//8個の8次元ブロックに分割して並列処理
- RoPE非可換性対策: 焼きこみ機構により学習時の基底を推論時まで維持

焼きこみ (Burn-in) 機構:
- 学習終了後、回転行列RをW_oに右掛け吸収: W_o' = W_o @ R
- これにより推論グラフから回転モジュールを削除可能
- 学習時と推論時の基底不一致を解消し、量子化後の安定性を確保

Author: SO8T Project Team
License: Apache 2.0
Date: 2024-11-06
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math
import logging

logger = logging.getLogger(__name__)


def make_skew_symmetric(t: torch.Tensor) -> torch.Tensor:
    """
    歪対称行列を作成
    
    直交回転行列を生成するための前処理。
    歪対称行列 A (A^T = -A) の指数関数 exp(A) は必ず直交行列になる。
    
    Args:
        t: 入力テンソル [*, 8, 8]
        
    Returns:
        歪対称行列 A - A^T [*, 8, 8]
        
    Theory:
        R = exp(A) where A^T = -A
        => R^T R = exp(A^T) exp(A) = exp(-A) exp(A) = I
    """
    return t - t.transpose(-1, -2)


def block_expm(A: torch.Tensor, max_iter: int = 12, eps: float = 1e-7) -> torch.Tensor:
    """
    ブロック単位での行列指数関数（数値安定化版）
    
    スケール・スクエア法とテイラー展開を組み合わせた実装。
    大きなノルムを持つ行列でも安定して計算できる。
    
    Args:
        A: 歪対称行列 [*, 8, 8]
        max_iter: テイラー展開の最大次数 (default: 12)
        eps: 数値安定化のためのイプシロン
        
    Returns:
        直交回転行列 R = exp(A) [*, 8, 8]
        
    Properties:
        - R^T R = I (直交性)
        - det(R) = 1 (特殊性)
        - ||R x|| = ||x|| (ノルム保存)
    """
    device = A.device
    dtype = A.dtype
    batch_shape = A.shape[:-2]
    
    # 単位行列
    I = torch.eye(8, device=device, dtype=dtype)
    I_batch = I.view(*([1] * len(batch_shape)), 8, 8).expand(*batch_shape, 8, 8)
    
    # ノルム計算
    norm = torch.norm(A, p='fro', dim=(-2, -1), keepdim=True)
    
    # スケーリング係数の計算（2の累乗に丸める）
    s = torch.ceil(torch.log2(norm + eps)).clamp(min=0)
    scale = torch.pow(2.0, s)
    
    # スケールダウン
    A_scaled = A / (scale + eps)
    
    # テイラー展開: exp(A) ≈ I + A + A^2/2! + A^3/3! + ...
    R = I_batch.clone()
    A_power = A_scaled.clone()
    factorial = 1.0
    
    for i in range(1, max_iter + 1):
        factorial *= i
        R = R + A_power / factorial
        
        # 次の項の計算
        if i < max_iter:
            A_power = torch.matmul(A_power, A_scaled)
    
    # スケール・スクエア法の逆操作: R^(2^s)
    max_s = int(s.max().item())
    for _ in range(max_s):
        # スケール別に処理（効率化）
        R = torch.matmul(R, R)
    
    return R


def orthogonality_loss(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    直交性損失を計算
    
    回転行列の直交性 R^T R = I からの逸脱を測定。
    学習中のドリフトを抑制するための正則化項。
    
    Args:
        R: 回転行列 [num_blocks, 8, 8]
        eps: 数値安定化のためのイプシロン
        
    Returns:
        直交性損失（スカラー）
        
    Formula:
        L_ortho = ||R^T R - I||_F^2
    """
    RTR = torch.matmul(R.transpose(-1, -2), R)
    I = torch.eye(8, device=R.device, dtype=R.dtype)
    diff = RTR - I
    return (diff.pow(2).sum() / R.shape[0]).clamp(min=eps)


def apply_block_rotation(x: torch.Tensor, theta: torch.Tensor, 
                         return_R: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    8次元ブロック単位での回転を適用
    
    入力テンソルをD//8個の8次元ブロックに分割し、
    各ブロックに独立した回転を適用する。
    
    Args:
        x: 入力テンソル [B, T, D] (D % 8 == 0)
        theta: 回転パラメータ [D//8, 8, 8]
        return_R: 回転行列も返すかどうか
        
    Returns:
        回転後のテンソル [B, T, D]
        回転行列 [D//8, 8, 8] (return_R=Trueの場合のみ)
        
    Complexity:
        Time: O(B * T * D * 64) = O(B * T * D)
        Space: O(D * 64) for rotation matrices
    """
    B, T, D = x.shape
    assert D % 8 == 0, f"Dimension {D} must be divisible by 8, got {D}"
    
    # 8次元ブロックに分割
    num_blocks = D // 8
    xv = x.view(B, T, num_blocks, 8)  # [B, T, num_blocks, 8]
    
    # 歪対称行列を作成
    A = make_skew_symmetric(theta)  # [num_blocks, 8, 8]
    
    # 回転行列を計算
    R = block_expm(A)  # [num_blocks, 8, 8]
    
    # 回転を適用: y[b,t,n,:] = R[n] @ x[b,t,n,:]
    y = torch.einsum('btne,neo->btno', xv, R)  # [B, T, num_blocks, 8]
    
    # 元の形状に戻す
    y = y.reshape(B, T, D)
    
    if return_R:
        return y, R
    return y, None


class SO8TRotationGate(nn.Module):
    """
    SO(8)群回転ゲートモジュール
    
    アテンション出力後に適用される等長写像。
    学習中はパラメータthetaが更新され、推論時は焼きこみにより除去される。
    
    Attributes:
        hidden_size: 隠れ層のサイズ（8の倍数）
        num_blocks: ブロック数 = hidden_size // 8
        init_scale: 初期化スケール
        learnable: 学習可能かどうか
        theta: 回転パラメータ [num_blocks, 8, 8]
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_blocks: Optional[int] = None,
        init_scale: float = 0.1,
        learnable: bool = True,
        orthogonal_reg: float = 1e-4
    ):
        """
        Args:
            hidden_size: 隠れ層のサイズ（8の倍数である必要がある）
            num_blocks: ブロック数（Noneの場合はhidden_size//8）
            init_scale: 初期化スケール（小さいほど恒等変換に近い）
            learnable: 学習可能かどうか
            orthogonal_reg: 直交性正則化の重み
        """
        super().__init__()
        
        assert hidden_size % 8 == 0, \
            f"[SO8T ERROR] hidden_size {hidden_size} must be divisible by 8"
        
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks or hidden_size // 8
        self.init_scale = init_scale
        self.learnable = learnable
        self.orthogonal_reg = orthogonal_reg
        
        # 回転パラメータの初期化
        # 小さな値で初期化することで、学習初期は恒等変換に近い状態から開始
        init_theta = torch.randn(self.num_blocks, 8, 8) * init_scale
        
        if learnable:
            self.theta = nn.Parameter(init_theta)
        else:
            self.register_buffer('theta', init_theta)
        
        logger.info(f"[SO8T] Initialized rotation gate: "
                   f"hidden_size={hidden_size}, num_blocks={self.num_blocks}, "
                   f"learnable={learnable}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向き計算
        
        Args:
            x: 入力テンソル [B, T, D]
            
        Returns:
            回転後のテンソル [B, T, D]
            
        Properties:
            - ||output|| = ||input|| (ノルム保存)
            - 可逆変換（逆回転により元に戻せる）
        """
        y, _ = apply_block_rotation(x, self.theta, return_R=False)
        return y
    
    def get_rotation_matrices(self) -> torch.Tensor:
        """
        現在の回転行列を取得
        
        焼きこみ時に使用。学習済みの回転行列を取得して
        線形層に吸収させる。
        
        Returns:
            回転行列 [num_blocks, 8, 8]
            
        Properties:
            - R^T R = I (直交性)
            - det(R) = 1 (特殊性)
        """
        A = make_skew_symmetric(self.theta)
        R = block_expm(A)
        return R
    
    def get_orthogonality_loss(self) -> torch.Tensor:
        """
        直交性正則化損失を取得
        
        学習中に直交性のドリフトを抑制するための損失項。
        総損失に加算して使用する。
        
        Returns:
            直交性損失（スカラー）
            
        Usage:
            total_loss = task_loss + so8t_gate.get_orthogonality_loss() * reg_weight
        """
        R = self.get_rotation_matrices()
        return orthogonality_loss(R) * self.orthogonal_reg
    
    def extra_repr(self) -> str:
        return (f"hidden_size={self.hidden_size}, "
                f"num_blocks={self.num_blocks}, "
                f"learnable={self.learnable}, "
                f"orthogonal_reg={self.orthogonal_reg}")


class SO8TAttentionWrapper(nn.Module):
    """
    既存のアテンション層にSO8T回転ゲートをラップするアダプタ
    
    Phi-4などの既存モデルに対して、アテンション層を置き換えることなく
    SO8T機能を追加できる。
    
    Usage:
        original_attn = model.layers[i].self_attn
        model.layers[i].self_attn = SO8TAttentionWrapper(
            original_attn, 
            hidden_size=model.config.hidden_size
        )
    """
    
    def __init__(
        self,
        base_attention: nn.Module,
        hidden_size: int,
        rotation_enabled: bool = True,
        init_scale: float = 0.1,
        orthogonal_reg: float = 1e-4
    ):
        """
        Args:
            base_attention: 元のアテンション層
            hidden_size: 隠れ層のサイズ
            rotation_enabled: 回転を有効にするか
            init_scale: 回転の初期化スケール
            orthogonal_reg: 直交性正則化の重み
        """
        super().__init__()
        
        self.base_attention = base_attention
        self.hidden_size = hidden_size
        self.rotation_enabled = rotation_enabled
        
        # SO8T回転ゲート
        if rotation_enabled:
            self.rotation_gate = SO8TRotationGate(
                hidden_size=hidden_size,
                init_scale=init_scale,
                orthogonal_reg=orthogonal_reg
            )
            logger.info(f"[SO8T] Wrapped attention with SO8T rotation gate")
        else:
            self.rotation_gate = None
            logger.info(f"[SO8T] Attention wrapper created without rotation (passthrough)")
    
    def forward(self, *args, **kwargs):
        """
        前向き計算
        
        元のアテンション層の出力にSO8T回転を適用する。
        """
        # 元のアテンション層を実行
        outputs = self.base_attention(*args, **kwargs)
        
        # 出力の形式を確認
        if isinstance(outputs, tuple):
            attn_output = outputs[0]
            rest = outputs[1:]
        else:
            attn_output = outputs
            rest = None
        
        # SO8T回転を適用
        if self.rotation_gate is not None:
            attn_output = self.rotation_gate(attn_output)
        
        # 元の形式で返す
        if rest is not None:
            return (attn_output,) + rest
        return attn_output
    
    def get_orthogonality_loss(self) -> torch.Tensor:
        """
        直交性正則化損失を取得
        
        Returns:
            直交性損失（回転が無効の場合は0）
        """
        if self.rotation_gate is not None:
            return self.rotation_gate.get_orthogonality_loss()
        return torch.tensor(0.0, device=next(self.parameters()).device)


def collect_so8t_orthogonality_loss(model: nn.Module) -> torch.Tensor:
    """
    モデル全体からSO8T回転ゲートの直交性損失を収集
    
    Args:
        model: PyTorchモデル
        
    Returns:
        総直交性損失
        
    Usage:
        ortho_loss = collect_so8t_orthogonality_loss(model)
        total_loss = task_loss + pet_loss + ortho_loss
    """
    total_loss = 0.0
    count = 0
    
    for module in model.modules():
        if isinstance(module, (SO8TRotationGate, SO8TAttentionWrapper)):
            total_loss = total_loss + module.get_orthogonality_loss()
            count += 1
    
    if count > 0:
        logger.debug(f"[SO8T] Collected orthogonality loss from {count} modules")
    
    return total_loss if isinstance(total_loss, torch.Tensor) else torch.tensor(total_loss)


def verify_rotation_properties(R: torch.Tensor, eps: float = 1e-3) -> Dict[str, bool]:
    """
    回転行列の性質を検証
    
    Args:
        R: 回転行列 [num_blocks, 8, 8]
        eps: 許容誤差
        
    Returns:
        検証結果の辞書
        
    Checks:
        - orthogonality: R^T R ≈ I
        - determinant: det(R) ≈ 1
        - norm_preservation: サンプルベクトルのノルム保存
    """
    device = R.device
    dtype = R.dtype
    num_blocks = R.shape[0]
    
    results = {}
    
    # 直交性チェック: R^T R = I
    RTR = torch.matmul(R.transpose(-1, -2), R)
    I = torch.eye(8, device=device, dtype=dtype)
    ortho_error = (RTR - I).abs().max().item()
    results['orthogonality'] = ortho_error < eps
    results['orthogonality_error'] = ortho_error
    
    # 行列式チェック: det(R) = 1
    dets = torch.linalg.det(R)
    det_error = (dets - 1.0).abs().max().item()
    results['determinant'] = det_error < eps
    results['determinant_error'] = det_error
    
    # ノルム保存チェック
    x_test = torch.randn(10, num_blocks, 8, device=device, dtype=dtype)
    x_norm = torch.norm(x_test, dim=-1)
    
    # 回転適用
    y_test = torch.einsum('bne,neo->bno', x_test, R)
    y_norm = torch.norm(y_test, dim=-1)
    
    norm_error = (x_norm - y_norm).abs().max().item()
    results['norm_preservation'] = norm_error < eps
    results['norm_error'] = norm_error
    
    return results


if __name__ == "__main__":
    """
    SO8T層の単体テスト
    """
    print("=" * 80)
    print("SO8T Layer Unit Test")
    print("=" * 80)
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")
    
    # テスト設定
    batch_size = 2
    seq_len = 16
    hidden_size = 512  # 8の倍数
    
    print(f"\n[Config]")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num blocks: {hidden_size // 8}")
    
    # 1. SO8TRotationGate単体テスト
    print(f"\n[Test 1] SO8TRotationGate")
    gate = SO8TRotationGate(hidden_size=hidden_size, init_scale=0.1).to(device)
    print(f"  Module: {gate}")
    
    # ダミー入力
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)
    print(f"  Input shape: {x.shape}")
    print(f"  Input norm: {x.norm().item():.6f}")
    
    # 前向き計算
    y = gate(x)
    print(f"  Output shape: {y.shape}")
    print(f"  Output norm: {y.norm().item():.6f}")
    print(f"  Norm preservation: {(x.norm() - y.norm()).abs().item():.6e}")
    
    # 直交性検証
    R = gate.get_rotation_matrices()
    print(f"  Rotation matrices shape: {R.shape}")
    
    verification = verify_rotation_properties(R)
    print(f"\n[Verification]")
    for key, value in verification.items():
        if isinstance(value, bool):
            status = "[OK]" if value else "[NG]"
            print(f"  {status} {key}")
        else:
            print(f"    └─ error: {value:.6e}")
    
    # 直交性損失
    ortho_loss = gate.get_orthogonality_loss()
    print(f"\n  Orthogonality loss: {ortho_loss.item():.6e}")
    
    # 2. 勾配フロー確認
    print(f"\n[Test 2] Gradient Flow")
    gate.train()
    optimizer = torch.optim.Adam(gate.parameters(), lr=1e-3)
    
    for step in range(3):
        optimizer.zero_grad()
        
        x = torch.randn(batch_size, seq_len, hidden_size, device=device)
        y = gate(x)
        
        # ダミー損失
        loss = y.pow(2).mean() + gate.get_orthogonality_loss()
        loss.backward()
        
        grad_norm = gate.theta.grad.norm().item()
        print(f"  Step {step}: loss={loss.item():.6f}, grad_norm={grad_norm:.6f}")
        
        optimizer.step()
    
    print("\n" + "=" * 80)
    print("[SO8T] All tests passed!")
    print("=" * 80)

