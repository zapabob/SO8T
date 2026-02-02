"""
PET (Periodic Error Term) 損失実装
二階差分による高周波変動抑制
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import math


def pet_penalty(
    sequence: torch.Tensor, 
    huber_delta: float = 0.0,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    PET（二階差分）ペナルティの計算
    
    Args:
        sequence: 入力系列 [B, T, D]
        huber_delta: Huber損失の閾値（0の場合はL2損失）
        reduction: 削減方法 ("mean", "sum", "none")
        
    Returns:
        PET損失値
    """
    if sequence.size(1) < 3:
        # 系列が短すぎる場合は0を返す
        return torch.tensor(0.0, device=sequence.device, dtype=sequence.dtype)
    
    # 二階差分の計算: Δ²y = y[t+1] - 2*y[t] + y[t-1]
    d2 = sequence[:, 2:] - 2 * sequence[:, 1:-1] + sequence[:, :-2]
    
    if huber_delta > 0:
        # Huber損失
        abs_d2 = torch.abs(d2)
        quad = torch.minimum(abs_d2, torch.tensor(huber_delta, device=sequence.device))
        lin = abs_d2 - quad
        loss = 0.5 * quad**2 + huber_delta * lin
    else:
        # L2損失
        loss = d2.pow(2)
    
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


class PETLoss(nn.Module):
    """
    PET損失モジュール
    3相スケジュール（warmup→main→anneal）に対応
    """
    
    def __init__(
        self,
        max_lambda: float = 0.1,
        warmup_steps: int = 100,
        main_steps: int = 1000,
        anneal_steps: int = 200,
        huber_delta: float = 0.0,
        reduction: str = "mean"
    ):
        """
        Args:
            max_lambda: 最大λ値
            warmup_steps: ウォームアップステップ数
            main_steps: メインステップ数
            anneal_steps: アニールステップ数
            huber_delta: Huber損失の閾値
            reduction: 削減方法
        """
        super().__init__()
        
        self.max_lambda = max_lambda
        self.warmup_steps = warmup_steps
        self.main_steps = main_steps
        self.anneal_steps = anneal_steps
        self.huber_delta = huber_delta
        self.reduction = reduction
        
        self.total_steps = warmup_steps + main_steps + anneal_steps
        
    def get_lambda(self, step: int) -> float:
        """
        現在のステップでのλ値を取得
        
        Args:
            step: 現在のステップ数
            
        Returns:
            λ値
        """
        if step < self.warmup_steps:
            # ウォームアップ: 0 → max_lambda
            return self.max_lambda * (step / self.warmup_steps)
        elif step < self.warmup_steps + self.main_steps:
            # メイン: max_lambda
            return self.max_lambda
        elif step < self.total_steps:
            # アニール: max_lambda → 0
            anneal_progress = (step - self.warmup_steps - self.main_steps) / self.anneal_steps
            return self.max_lambda * (1 - anneal_progress)
        else:
            # 終了後: 0
            return 0.0
    
    def forward(
        self, 
        sequence: torch.Tensor, 
        step: int,
        return_lambda: bool = False
    ) -> torch.Tensor:
        """
        前向き計算
        
        Args:
            sequence: 入力系列 [B, T, D]
            step: 現在のステップ数
            return_lambda: λ値を返すかどうか
            
        Returns:
            PET損失値（λ値を含む場合もある）
        """
        # λ値を取得
        lambda_val = self.get_lambda(step)
        
        # PETペナルティを計算
        penalty = pet_penalty(sequence, self.huber_delta, self.reduction)
        
        # λ値を適用
        loss = lambda_val * penalty
        
        if return_lambda:
            return loss, lambda_val
        else:
            return loss
    
    def extra_repr(self) -> str:
        return (
            f"max_lambda={self.max_lambda}, "
            f"warmup_steps={self.warmup_steps}, "
            f"main_steps={self.main_steps}, "
            f"anneal_steps={self.anneal_steps}, "
            f"huber_delta={self.huber_delta}"
        )


class CombinedLoss(nn.Module):
    """
    タスク損失 + PET損失の組み合わせ
    """
    
    def __init__(
        self,
        task_loss_fn: nn.Module,
        pet_loss: PETLoss,
        pet_weight: float = 1.0
    ):
        """
        Args:
            task_loss_fn: タスク損失関数
            pet_loss: PET損失
            pet_weight: PET損失の重み
        """
        super().__init__()
        
        self.task_loss_fn = task_loss_fn
        self.pet_loss = pet_loss
        self.pet_weight = pet_weight
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        hidden_states: torch.Tensor,
        step: int,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        前向き計算
        
        Args:
            logits: 予測ロジット [B, T, V]
            labels: 正解ラベル [B, T]
            hidden_states: 隠れ状態 [B, T, D]
            step: 現在のステップ数
            return_components: 各成分を返すかどうか
            
        Returns:
            合計損失（成分を含む場合もある）
        """
        # タスク損失
        task_loss = self.task_loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # PET損失
        pet_loss_val = self.pet_loss(hidden_states, step)
        
        # 合計損失
        total_loss = task_loss + self.pet_weight * pet_loss_val
        
        if return_components:
            return {
                "total_loss": total_loss,
                "task_loss": task_loss,
                "pet_loss": pet_loss_val,
                "pet_lambda": self.pet_loss.get_lambda(step)
            }
        else:
            return total_loss


def create_pet_schedule(
    max_lambda: float = 0.1,
    total_steps: int = 1000,
    warmup_ratio: float = 0.1,
    anneal_ratio: float = 0.2
) -> PETLoss:
    """
    PET損失スケジュールを作成
    
    Args:
        max_lambda: 最大λ値
        total_steps: 総ステップ数
        warmup_ratio: ウォームアップ比率
        anneal_ratio: アニール比率
        
    Returns:
        PET損失インスタンス
    """
    warmup_steps = int(total_steps * warmup_ratio)
    anneal_steps = int(total_steps * anneal_ratio)
    main_steps = total_steps - warmup_steps - anneal_steps
    
    return PETLoss(
        max_lambda=max_lambda,
        warmup_steps=warmup_steps,
        main_steps=main_steps,
        anneal_steps=anneal_steps
    )
