#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
損失関数実装
- L_total = L_task + λ_pet(t)*L_PET + L_reg
- λ_pet(t): 3相スケジュール（0-20%:0.01, 20-60%:0.05, 60-100%:0.1）
- Label Smoothing (ε=0.1)
- 勾配ノイズ注入（σ=0.01）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class PETScheduler:
    """PET λスケジューラー（3相）"""
    
    def __init__(self, 
                 phase1_ratio: float = 0.2,  # 探索フェーズ
                 phase2_ratio: float = 0.4,  # 遷移フェーズ（0.2-0.6）
                 lambda_phase1: float = 0.01,
                 lambda_phase2: float = 0.05,
                 lambda_phase3: float = 0.1):
        """
        Args:
            phase1_ratio: フェーズ1終了時点（0-20%）
            phase2_ratio: フェーズ2の長さ（20-60%なので0.4）
            lambda_phase1: 探索フェーズλ
            lambda_phase2: 遷移フェーズλ
            lambda_phase3: 安定フェーズλ
        """
        self.phase1_end = phase1_ratio
        self.phase2_end = phase1_ratio + phase2_ratio
        self.lambda_phase1 = lambda_phase1
        self.lambda_phase2 = lambda_phase2
        self.lambda_phase3 = lambda_phase3
    
    def get_lambda(self, progress: float) -> float:
        """
        進捗率に応じたλ値取得
        
        Args:
            progress: 学習進捗率（0.0-1.0）
        
        Returns:
            現在のλ値
        """
        if progress < self.phase1_end:
            # フェーズ1: 探索（0-20%）
            return self.lambda_phase1
        elif progress < self.phase2_end:
            # フェーズ2: 遷移（20-60%）
            return self.lambda_phase2
        else:
            # フェーズ3: 安定（60-100%）
            return self.lambda_phase3
    
    def get_phase_name(self, progress: float) -> str:
        """フェーズ名取得"""
        if progress < self.phase1_end:
            return "exploration"
        elif progress < self.phase2_end:
            return "transition"
        else:
            return "stabilization"


class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing付きCross Entropy Loss"""
    
    def __init__(self, epsilon: float = 0.1, reduction: str = 'mean'):
        """
        Args:
            epsilon: smoothing parameter（0.1推奨）
            reduction: 'mean', 'sum', 'none'
        """
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch, seq_len, vocab_size]
            targets: [batch, seq_len]
        
        Returns:
            loss: scalar or [batch, seq_len]
        """
        vocab_size = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # One-hot targets
        targets_one_hot = F.one_hot(targets, num_classes=vocab_size).float()
        
        # Label smoothing
        smooth_targets = (1.0 - self.epsilon) * targets_one_hot + self.epsilon / vocab_size
        
        # Cross entropy
        loss = -(smooth_targets * log_probs).sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class PETLoss(nn.Module):
    """PET正規化損失（二階差分）"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        二階差分ペナルティ計算
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        
        Returns:
            pet_loss: scalar
        """
        if hidden_states.size(1) < 3:
            # シーケンス長が3未満なら計算不可
            return torch.tensor(0.0, device=hidden_states.device)
        
        # 時系列方向の二階差分
        # Δ²x[t] = x[t+2] - 2*x[t+1] + x[t]
        x_t = hidden_states[:, :-2, :]      # [batch, seq_len-2, hidden_dim]
        x_t1 = hidden_states[:, 1:-1, :]    # [batch, seq_len-2, hidden_dim]
        x_t2 = hidden_states[:, 2:, :]      # [batch, seq_len-2, hidden_dim]
        
        second_diff = x_t2 - 2.0 * x_t1 + x_t
        
        # L2ノルム（二乗和平均）
        pet_loss = torch.mean(second_diff ** 2)
        
        return pet_loss


class GradientNoiseInjector:
    """勾配ノイズ注入器"""
    
    def __init__(self, std: float = 0.01):
        """
        Args:
            std: ノイズの標準偏差
        """
        self.std = std
    
    def inject(self, model: nn.Module):
        """
        モデルの勾配にノイズ注入
        
        Args:
            model: PyTorchモデル
        """
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * self.std
                    param.grad.add_(noise)


class SO8TCompositeLoss(nn.Module):
    """SO8T統合損失関数"""
    
    def __init__(self,
                 label_smoothing: float = 0.1,
                 pet_scheduler: Optional[PETScheduler] = None,
                 gradient_noise_std: float = 0.01,
                 weight_decay: float = 0.05):
        """
        Args:
            label_smoothing: Label smoothing parameter
            pet_scheduler: PET λスケジューラー
            gradient_noise_std: 勾配ノイズ標準偏差
            weight_decay: Weight decay（正則化）
        """
        super().__init__()
        
        self.task_loss_fn = LabelSmoothingCrossEntropy(epsilon=label_smoothing)
        self.pet_loss_fn = PETLoss()
        self.pet_scheduler = pet_scheduler or PETScheduler()
        self.gradient_noise_injector = GradientNoiseInjector(std=gradient_noise_std)
        self.weight_decay = weight_decay
        
        # 統計追跡
        self.loss_history = {
            'total': [],
            'task': [],
            'pet': [],
            'reg': [],
            'lambda': []
        }
    
    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor,
                hidden_states: Optional[torch.Tensor] = None,
                progress: float = 0.0,
                model: Optional[nn.Module] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        総合損失計算
        
        Args:
            logits: [batch, seq_len, vocab_size]
            targets: [batch, seq_len]
            hidden_states: [batch, seq_len, hidden_dim]（PET用）
            progress: 学習進捗率（0.0-1.0）
            model: モデル（weight decay用）
        
        Returns:
            total_loss: 総合損失
            loss_dict: 損失内訳辞書
        """
        # Task loss（Label smoothing付きCE）
        L_task = self.task_loss_fn(logits, targets)
        
        # PET loss
        L_pet = torch.tensor(0.0, device=logits.device)
        if hidden_states is not None:
            L_pet = self.pet_loss_fn(hidden_states)
        
        # PET λスケジューリング
        lambda_pet = self.pet_scheduler.get_lambda(progress)
        
        # Regularization loss（Weight decay）
        L_reg = torch.tensor(0.0, device=logits.device)
        if model is not None and self.weight_decay > 0:
            for param in model.parameters():
                if param.requires_grad:
                    L_reg += torch.sum(param ** 2)
            L_reg = self.weight_decay * L_reg
        
        # Total loss
        L_total = L_task + lambda_pet * L_pet + L_reg
        
        # 統計記録
        loss_dict = {
            'loss': L_total.item(),
            'task_loss': L_task.item(),
            'pet_loss': L_pet.item(),
            'reg_loss': L_reg.item(),
            'lambda_pet': lambda_pet,
            'pet_phase': self.pet_scheduler.get_phase_name(progress)
        }
        
        self.loss_history['total'].append(L_total.item())
        self.loss_history['task'].append(L_task.item())
        self.loss_history['pet'].append(L_pet.item())
        self.loss_history['reg'].append(L_reg.item())
        self.loss_history['lambda'].append(lambda_pet)
        
        return L_total, loss_dict
    
    def get_statistics(self) -> Dict[str, float]:
        """損失統計取得"""
        if not self.loss_history['total']:
            return {}
        
        stats = {}
        for key, values in self.loss_history.items():
            if values:
                stats[f'{key}_mean'] = np.mean(values)
                stats[f'{key}_std'] = np.std(values)
                stats[f'{key}_min'] = np.min(values)
                stats[f'{key}_max'] = np.max(values)
        
        return stats
    
    def reset_statistics(self):
        """統計リセット"""
        for key in self.loss_history:
            self.loss_history[key] = []


class StochasticWeightAveraging:
    """SWA（Stochastic Weight Averaging）実装"""
    
    def __init__(self, model: nn.Module, swa_start: float = 0.75):
        """
        Args:
            model: PyTorchモデル
            swa_start: SWA開始時点（進捗率0.75 = 75%）
        """
        self.model = model
        self.swa_start = swa_start
        self.swa_n = 0
        
        # SWAモデル（平均重み保持）
        self.swa_model = None
        self.swa_active = False
    
    def update(self, progress: float):
        """
        SWA更新
        
        Args:
            progress: 学習進捗率（0.0-1.0）
        """
        if progress < self.swa_start:
            return
        
        if not self.swa_active:
            # SWA初期化
            self.swa_active = True
            self.swa_model = {}
            for name, param in self.model.named_parameters():
                self.swa_model[name] = param.data.clone()
            self.swa_n = 1
            print(f"[SWA] Started at progress {progress:.2%}")
        else:
            # 移動平均更新
            self.swa_n += 1
            for name, param in self.model.named_parameters():
                self.swa_model[name] += (param.data - self.swa_model[name]) / self.swa_n
    
    def apply_swa_weights(self):
        """SWA重みをモデルに適用"""
        if not self.swa_active:
            print("[WARNING] SWA not active, skipping")
            return
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data.copy_(self.swa_model[name])
        
        print(f"[SWA] Applied averaged weights (n={self.swa_n})")


# [OK] テスト用ユーティリティ
def test_loss_functions():
    """損失関数テスト"""
    print("\n[TEST] Testing loss functions...")
    
    # ダミーデータ
    batch_size = 2
    seq_len = 10
    vocab_size = 1000
    hidden_dim = 768
    
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    
    # PETスケジューラーテスト
    scheduler = PETScheduler()
    for progress in [0.1, 0.3, 0.7]:
        lambda_val = scheduler.get_lambda(progress)
        phase = scheduler.get_phase_name(progress)
        print(f"Progress {progress:.0%}: λ={lambda_val}, phase={phase}")
    
    # Label smoothing test
    ls_loss = LabelSmoothingCrossEntropy(epsilon=0.1)
    loss = ls_loss(logits, targets)
    print(f"\nLabel Smoothing Loss: {loss.item():.4f}")
    
    # PET loss test
    pet_loss_fn = PETLoss()
    pet_loss = pet_loss_fn(hidden_states)
    print(f"PET Loss: {pet_loss.item():.6f}")
    
    # Composite loss test
    composite_loss = SO8TCompositeLoss()
    total_loss, loss_dict = composite_loss(
        logits=logits,
        targets=targets,
        hidden_states=hidden_states,
        progress=0.5
    )
    print(f"\nComposite Loss:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value}")
    
    print("\n[OK] Loss functions test passed!")


if __name__ == "__main__":
    test_loss_functions()

