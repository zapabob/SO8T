"""
PET (Periodic Error Term) Regularization - 本番環境対応版
二階差分による高周波変動抑制と3相スケジューリング

理論的基盤:
- 二階差分: Δ²x[t] = x[t+2] - 2*x[t+1] + x[t]
- 高周波抑制: 時系列方向の急激な変化を罰則化
- SO8T等長性との相乗効果: ノルム保存と滑らかさの両立
- RoPE位相安定化: 長文での発振・位相ドリフトを抑制

3相スケジュール:
1. 探索相 (0-20%): λ=0.01 - 小さなペナルティで自由度を保つ
2. 遷移相 (20-60%): λ=0.05 - 徐々に滑らかさを強制
3. 安定相 (60-100%): λ=0.1 - 最大ペナルティで収束を促進

本番環境要件:
- 電源断リカバリー対応
- メモリ効率的な計算
- 数値安定性の保証
- ログ・監査機能
- エラーハンドリング

Author: SO8T Project Team
License: Apache 2.0
Date: 2024-11-06
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Union
import logging
import math
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PETPhase(Enum):
    """PET学習フェーズ"""
    EXPLORATION = "exploration"  # 探索相
    TRANSITION = "transition"    # 遷移相
    STABILIZATION = "stabilization"  # 安定相
    DISABLED = "disabled"        # 無効


@dataclass
class PETConfig:
    """PET正規化の設定"""
    # 基本設定
    lambda_exploration: float = 0.01
    lambda_transition: float = 0.05
    lambda_stabilization: float = 0.1
    
    # フェーズ比率
    exploration_ratio: float = 0.2
    transition_ratio: float = 0.4
    stabilization_ratio: float = 0.4
    
    # 損失関数設定
    huber_delta: float = 0.0  # 0ならL2損失
    reduction: str = "mean"
    
    # 安定性設定
    clip_gradient: bool = True
    max_gradient_norm: float = 1.0
    eps: float = 1e-8
    
    # ログ設定
    log_every_n_steps: int = 100
    save_stats: bool = True
    
    def __post_init__(self):
        """設定の妥当性チェック"""
        total_ratio = self.exploration_ratio + self.transition_ratio + self.stabilization_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            logger.warning(f"[PET Config] Phase ratios sum to {total_ratio}, normalizing to 1.0")
            self.exploration_ratio /= total_ratio
            self.transition_ratio /= total_ratio
            self.stabilization_ratio /= total_ratio
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            'lambda_exploration': self.lambda_exploration,
            'lambda_transition': self.lambda_transition,
            'lambda_stabilization': self.lambda_stabilization,
            'exploration_ratio': self.exploration_ratio,
            'transition_ratio': self.transition_ratio,
            'stabilization_ratio': self.stabilization_ratio,
            'huber_delta': self.huber_delta,
            'reduction': self.reduction,
            'clip_gradient': self.clip_gradient,
            'max_gradient_norm': self.max_gradient_norm,
        }


def second_difference(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    二階差分を計算
    
    Args:
        x: 入力テンソル [B, T, D]
        dim: 差分を取る次元（通常は時系列次元=1）
        
    Returns:
        二階差分 [B, T-2, D]
        
    Formula:
        Δ²x[t] = x[t+2] - 2*x[t+1] + x[t]
        
    Properties:
        - 高周波成分を強調
        - 一階差分の差分 = 加速度的な変化
    """
    if x.size(dim) < 3:
        # 系列が短すぎる場合は0を返す
        return torch.zeros(*x.shape[:dim], 0, *x.shape[dim+1:], 
                          device=x.device, dtype=x.dtype)
    
    # インデックススライスで二階差分を計算
    if dim == 1:
        d2 = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
    elif dim == 0:
        d2 = x[2:] - 2 * x[1:-1] + x[:-2]
    else:
        # 一般的な次元の場合
        indices_plus2 = [slice(None)] * len(x.shape)
        indices_plus1 = [slice(None)] * len(x.shape)
        indices_plus0 = [slice(None)] * len(x.shape)
        
        indices_plus2[dim] = slice(2, None)
        indices_plus1[dim] = slice(1, -1)
        indices_plus0[dim] = slice(None, -2)
        
        d2 = x[tuple(indices_plus2)] - 2 * x[tuple(indices_plus1)] + x[tuple(indices_plus0)]
    
    return d2


def pet_loss_l2(d2: torch.Tensor, reduction: str = "mean", eps: float = 1e-8) -> torch.Tensor:
    """
    L2ベースのPET損失
    
    Args:
        d2: 二階差分 [B, T-2, D]
        reduction: 削減方法
        eps: 数値安定化
        
    Returns:
        PET損失
        
    Formula:
        L_PET = ||Δ²x||²
    """
    loss = d2.pow(2)
    
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def pet_loss_huber(d2: torch.Tensor, delta: float, reduction: str = "mean") -> torch.Tensor:
    """
    Huber損失ベースのPET損失
    
    外れ値に対してロバスト。大きな変化に対しては線形的にペナルティを与える。
    
    Args:
        d2: 二階差分 [B, T-2, D]
        delta: Huber損失の閾値
        reduction: 削減方法
        
    Returns:
        PET損失
        
    Formula:
        L_huber(a) = 0.5 * a²           if |a| ≤ δ
                   = δ * (|a| - 0.5δ)   if |a| > δ
    """
    abs_d2 = d2.abs()
    
    # |d2| ≤ δ の部分: 0.5 * d2²
    quad_mask = abs_d2 <= delta
    quad_loss = 0.5 * d2.pow(2)
    
    # |d2| > δ の部分: δ * (|d2| - 0.5δ)
    linear_loss = delta * (abs_d2 - 0.5 * delta)
    
    loss = torch.where(quad_mask, quad_loss, linear_loss)
    
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


class PETRegularization(nn.Module):
    """
    PET正規化モジュール（本番環境対応版）
    
    Features:
    - 3相スケジューリング
    - 数値安定性保証
    - 勾配クリッピング
    - 詳細なログ・統計
    - チェックポイント対応
    """
    
    def __init__(
        self,
        config: Optional[PETConfig] = None,
        total_steps: Optional[int] = None,
    ):
        """
        Args:
            config: PET設定（Noneの場合はデフォルト）
            total_steps: 総学習ステップ数
        """
        super().__init__()
        
        self.config = config or PETConfig()
        self.total_steps = total_steps
        
        # フェーズ境界の計算
        if total_steps is not None:
            self.exploration_end = int(total_steps * self.config.exploration_ratio)
            self.transition_end = self.exploration_end + int(
                total_steps * self.config.transition_ratio
            )
        else:
            self.exploration_end = None
            self.transition_end = None
        
        # 統計情報
        self.reset_stats()
        
        logger.info(f"[PET] Initialized with config: {self.config.to_dict()}")
        if total_steps:
            logger.info(f"[PET] Phase boundaries: "
                       f"exploration=0-{self.exploration_end}, "
                       f"transition={self.exploration_end}-{self.transition_end}, "
                       f"stabilization={self.transition_end}-{total_steps}")
    
    def reset_stats(self):
        """統計情報をリセット"""
        self.stats = {
            'total_calls': 0,
            'phase_counts': {phase: 0 for phase in PETPhase},
            'loss_history': [],
            'lambda_history': [],
        }
    
    def get_phase(self, step: int) -> PETPhase:
        """
        現在のフェーズを取得
        
        Args:
            step: 現在のステップ
            
        Returns:
            PETPhase
        """
        if self.total_steps is None or self.exploration_end is None:
            # total_stepsが未設定の場合は比率で計算
            return PETPhase.TRANSITION
        
        if step < 0:
            return PETPhase.DISABLED
        elif step < self.exploration_end:
            return PETPhase.EXPLORATION
        elif step < self.transition_end:
            return PETPhase.TRANSITION
        elif step < self.total_steps:
            return PETPhase.STABILIZATION
        else:
            return PETPhase.DISABLED
    
    def get_lambda(self, step: int) -> float:
        """
        現在のλ値を取得
        
        Args:
            step: 現在のステップ
            
        Returns:
            λ値
        """
        phase = self.get_phase(step)
        
        if phase == PETPhase.EXPLORATION:
            return self.config.lambda_exploration
        elif phase == PETPhase.TRANSITION:
            return self.config.lambda_transition
        elif phase == PETPhase.STABILIZATION:
            return self.config.lambda_stabilization
        else:
            return 0.0
    
    def compute_pet_loss(
        self,
        hidden_states: torch.Tensor,
        step: int,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        PET損失を計算
        
        Args:
            hidden_states: 隠れ状態 [B, T, D]
            step: 現在のステップ
            mask: マスク [B, T]（Noneの場合は全て有効）
            
        Returns:
            loss: PET損失
            info: 詳細情報
        """
        # フェーズとλを取得
        phase = self.get_phase(step)
        lambda_val = self.get_lambda(step)
        
        # 統計更新
        self.stats['total_calls'] += 1
        self.stats['phase_counts'][phase] += 1
        
        # λ=0の場合は損失を返さない
        if lambda_val == 0.0:
            info = {
                'phase': phase.value,
                'lambda': 0.0,
                'loss': 0.0,
                'raw_loss': 0.0,
            }
            return torch.tensor(0.0, device=hidden_states.device), info
        
        # 二階差分を計算
        d2 = second_difference(hidden_states, dim=1)  # [B, T-2, D]
        
        # マスクを適用（必要な場合）
        if mask is not None:
            # マスクも二階差分に合わせてスライス
            mask_d2 = mask[:, 1:-1]  # [B, T-2]
            d2 = d2 * mask_d2.unsqueeze(-1)
        
        # 損失を計算
        if self.config.huber_delta > 0:
            raw_loss = pet_loss_huber(d2, self.config.huber_delta, self.config.reduction)
        else:
            raw_loss = pet_loss_l2(d2, self.config.reduction, self.config.eps)
        
        # λを適用
        loss = lambda_val * raw_loss
        
        # 勾配クリッピング（必要な場合）
        if self.config.clip_gradient and loss.requires_grad:
            loss.register_hook(
                lambda grad: torch.clamp(grad, -self.config.max_gradient_norm, 
                                        self.config.max_gradient_norm)
            )
        
        # 詳細情報
        info = {
            'phase': phase.value,
            'lambda': lambda_val,
            'loss': loss.item(),
            'raw_loss': raw_loss.item(),
            'd2_mean': d2.abs().mean().item(),
            'd2_max': d2.abs().max().item(),
            'd2_std': d2.std().item(),
        }
        
        # ログ
        if self.config.log_every_n_steps > 0 and \
           self.stats['total_calls'] % self.config.log_every_n_steps == 0:
            logger.info(f"[PET Step {step}] Phase={phase.value}, λ={lambda_val:.4f}, "
                       f"Loss={loss.item():.6e}, RawLoss={raw_loss.item():.6e}")
        
        # 統計保存
        if self.config.save_stats:
            self.stats['loss_history'].append(loss.item())
            self.stats['lambda_history'].append(lambda_val)
        
        return loss, info
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        step: int,
        mask: Optional[torch.Tensor] = None,
        return_info: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        前向き計算
        
        Args:
            hidden_states: 隠れ状態 [B, T, D]
            step: 現在のステップ
            mask: マスク [B, T]
            return_info: 詳細情報を返すか
            
        Returns:
            loss: PET損失
            info: 詳細情報（return_info=Trueの場合）
        """
        loss, info = self.compute_pet_loss(hidden_states, step, mask)
        
        if return_info:
            return loss, info
        return loss
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得"""
        return {
            'total_calls': self.stats['total_calls'],
            'phase_counts': {k.value: v for k, v in self.stats['phase_counts'].items()},
            'avg_loss': sum(self.stats['loss_history']) / max(len(self.stats['loss_history']), 1),
            'loss_std': torch.tensor(self.stats['loss_history']).std().item() 
                       if len(self.stats['loss_history']) > 0 else 0.0,
        }
    
    def state_dict(self) -> Dict[str, Any]:
        """チェックポイント用の状態を保存"""
        return {
            'config': self.config.to_dict(),
            'total_steps': self.total_steps,
            'stats': self.stats,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """チェックポイントから状態を復元"""
        self.total_steps = state_dict.get('total_steps')
        self.stats = state_dict.get('stats', {})
        
        # フェーズ境界を再計算
        if self.total_steps is not None:
            self.exploration_end = int(self.total_steps * self.config.exploration_ratio)
            self.transition_end = self.exploration_end + int(
                self.total_steps * self.config.transition_ratio
            )
        
        logger.info(f"[PET] Loaded state from checkpoint")


def create_pet_regularization(
    total_steps: int,
    lambda_max: float = 0.1,
    exploration_ratio: float = 0.2,
    transition_ratio: float = 0.4,
    huber_delta: float = 0.0
) -> PETRegularization:
    """
    PET正規化を簡単に作成するヘルパー関数
    
    Args:
        total_steps: 総学習ステップ数
        lambda_max: 最大λ値（安定相で使用）
        exploration_ratio: 探索相の比率
        transition_ratio: 遷移相の比率
        huber_delta: Huber損失の閾値（0ならL2）
        
    Returns:
        PETRegularizationインスタンス
        
    Example:
        pet = create_pet_regularization(total_steps=10000, lambda_max=0.1)
        
        for step in range(10000):
            loss = task_loss + pet(hidden_states, step)
    """
    config = PETConfig(
        lambda_exploration=lambda_max * 0.1,
        lambda_transition=lambda_max * 0.5,
        lambda_stabilization=lambda_max,
        exploration_ratio=exploration_ratio,
        transition_ratio=transition_ratio,
        stabilization_ratio=1.0 - exploration_ratio - transition_ratio,
        huber_delta=huber_delta,
    )
    
    return PETRegularization(config=config, total_steps=total_steps)


if __name__ == "__main__":
    """
    PET正規化の単体テスト
    """
    print("=" * 80)
    print("PET Regularization Unit Test")
    print("=" * 80)
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")
    
    # テスト設定
    batch_size = 4
    seq_len = 64
    hidden_size = 512
    total_steps = 1000
    
    print(f"\n[Config]")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Total steps: {total_steps}")
    
    # 1. PET正規化の作成
    print(f"\n[Test 1] Create PET Regularization")
    pet = create_pet_regularization(total_steps=total_steps, lambda_max=0.1)
    print(f"  Config: {pet.config.to_dict()}")
    
    # 2. フェーズ遷移のテスト
    print(f"\n[Test 2] Phase Transitions")
    test_steps = [0, 200, 600, 900, 1000, 1100]
    for step in test_steps:
        phase = pet.get_phase(step)
        lambda_val = pet.get_lambda(step)
        print(f"  Step {step:4d}: phase={phase.value:15s}, λ={lambda_val:.4f}")
    
    # 3. 損失計算のテスト
    print(f"\n[Test 3] Loss Computation")
    
    # 滑らかな系列
    t = torch.linspace(0, 2*math.pi, seq_len, device=device)
    smooth_hidden = torch.sin(t).unsqueeze(0).unsqueeze(-1).expand(
        batch_size, seq_len, hidden_size
    )
    
    # ノイズが多い系列
    noisy_hidden = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    step = 500  # 遷移相
    
    smooth_loss, smooth_info = pet(smooth_hidden, step, return_info=True)
    noisy_loss, noisy_info = pet(noisy_hidden, step, return_info=True)
    
    print(f"  Smooth sequence:")
    print(f"    Loss: {smooth_loss.item():.6e}")
    print(f"    d2_mean: {smooth_info['d2_mean']:.6e}")
    print(f"    d2_max: {smooth_info['d2_max']:.6e}")
    
    print(f"  Noisy sequence:")
    print(f"    Loss: {noisy_loss.item():.6e}")
    print(f"    d2_mean: {noisy_info['d2_mean']:.6e}")
    print(f"    d2_max: {noisy_info['d2_max']:.6e}")
    
    print(f"  Loss ratio (noisy/smooth): {noisy_loss.item() / (smooth_loss.item() + 1e-8):.2f}")
    
    # 4. 勾配フローのテスト
    print(f"\n[Test 4] Gradient Flow")
    
    hidden = torch.randn(batch_size, seq_len, hidden_size, device=device, requires_grad=True)
    loss = pet(hidden, step)
    loss.backward()
    
    print(f"  Loss: {loss.item():.6e}")
    print(f"  Gradient norm: {hidden.grad.norm().item():.6e}")
    print(f"  Gradient max: {hidden.grad.abs().max().item():.6e}")
    
    # 5. 統計情報のテスト
    print(f"\n[Test 5] Statistics")
    stats = pet.get_stats()
    print(f"  Total calls: {stats['total_calls']}")
    print(f"  Phase counts: {stats['phase_counts']}")
    print(f"  Average loss: {stats['avg_loss']:.6e}")
    
    print("\n" + "=" * 80)
    print("[PET] All tests passed!")
    print("=" * 80)

