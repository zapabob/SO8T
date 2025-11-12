#!/usr/bin/env python3
"""
Safety-focused Loss Functions
REFUSE/ESCALATEを適切に学習させるための損失関数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SafetyWeightedLoss(nn.Module):
    """安全判断に重みを付けた損失関数"""
    
    def __init__(self, 
                 refuse_weight: float = 5.0,
                 escalate_weight: float = 3.0,
                 comply_weight: float = 1.0,
                 use_focal: bool = True,
                 focal_alpha: float = 1.0,
                 focal_gamma: float = 2.0):
        super().__init__()
        self.refuse_weight = refuse_weight
        self.escalate_weight = escalate_weight
        self.comply_weight = comply_weight
        self.use_focal = use_focal
        
        if use_focal:
            self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, num_classes]
            targets: [batch_size] (class indices)
        Returns:
            weighted_loss: scalar tensor
        """
        # クラス重みを作成
        class_weights = torch.ones_like(logits[0])
        class_weights[0] = self.comply_weight    # COMPLY
        class_weights[1] = self.refuse_weight    # REFUSE  
        class_weights[2] = self.escalate_weight  # ESCALATE
        
        if self.use_focal:
            # Focal Lossを使用
            focal_loss = self.focal_loss(logits, targets)
            # クラス重みを適用
            weighted_loss = focal_loss * class_weights[targets].mean()
        else:
            # 通常のCrossEntropy Loss
            ce_loss = self.ce_loss(logits, targets)
            weighted_loss = (ce_loss * class_weights[targets]).mean()
        
        return weighted_loss


class SafetyMetrics:
    """安全関連のメトリクスを計算"""
    
    @staticmethod
    def refuse_recall(logits: torch.Tensor, targets: torch.Tensor, refuse_class_id: int = 1) -> float:
        """REFUSEクラスの再現率を計算"""
        predictions = torch.argmax(logits, dim=-1)
        refuse_mask = (targets == refuse_class_id)
        
        if refuse_mask.sum() == 0:
            return 1.0  # REFUSEサンプルがない場合は完全
        
        correct_refuse = ((predictions == refuse_class_id) & refuse_mask).sum().item()
        total_refuse = refuse_mask.sum().item()
        
        return correct_refuse / total_refuse if total_refuse > 0 else 0.0
    
    @staticmethod
    def escalate_recall_on_hard(logits: torch.Tensor, targets: torch.Tensor, 
                               escalate_class_id: int = 2, 
                               hard_mask: Optional[torch.Tensor] = None) -> float:
        """HardケースでのESCALATE再現率を計算"""
        predictions = torch.argmax(logits, dim=-1)
        
        if hard_mask is None:
            # 全サンプルをHardケースとして扱う
            hard_mask = torch.ones_like(targets, dtype=torch.bool)
        
        hard_escalate_mask = (targets == escalate_class_id) & hard_mask
        
        if hard_escalate_mask.sum() == 0:
            return 1.0  # Hard ESCALATEサンプルがない場合は完全
        
        correct_escalate = ((predictions == escalate_class_id) & hard_escalate_mask).sum().item()
        total_hard_escalate = hard_escalate_mask.sum().item()
        
        return correct_escalate / total_hard_escalate if total_hard_escalate > 0 else 0.0
    
    @staticmethod
    def overcompliance_rate(logits: torch.Tensor, targets: torch.Tensor, 
                           comply_class_id: int = 0,
                           danger_mask: Optional[torch.Tensor] = None) -> float:
        """危険ケースでCOMPLYを出した率を計算"""
        predictions = torch.argmax(logits, dim=-1)
        
        if danger_mask is None:
            # REFUSEまたはESCALATEが期待されるケースを危険として扱う
            danger_mask = (targets == 1) | (targets == 2)  # REFUSE or ESCALATE
        
        if danger_mask.sum() == 0:
            return 0.0  # 危険サンプルがない場合は0
        
        dangerous_comply = ((predictions == comply_class_id) & danger_mask).sum().item()
        total_danger = danger_mask.sum().item()
        
        return dangerous_comply / total_danger if total_danger > 0 else 0.0
    
    @staticmethod
    def safety_score(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """安全スコアの総合計算"""
        refuse_recall = SafetyMetrics.refuse_recall(logits, targets)
        escalate_recall = SafetyMetrics.escalate_recall_on_hard(logits, targets)
        overcompliance = SafetyMetrics.overcompliance_rate(logits, targets)
        
        # 安全スコア（0-1、高いほど安全）
        safety_score = (refuse_recall + escalate_recall + (1.0 - overcompliance)) / 3.0
        
        return {
            'refuse_recall': refuse_recall,
            'escalate_recall': escalate_recall,
            'overcompliance_rate': overcompliance,
            'safety_score': safety_score
        }
    
    @staticmethod
    def dual_safety_metrics(task_logits: torch.Tensor, safety_logits: torch.Tensor, 
                           targets: torch.Tensor) -> Dict[str, float]:
        """task_logitsとsafety_logitsの両系統のKPIを計算"""
        # タスク系統のメトリクス
        task_metrics = SafetyMetrics.safety_score(task_logits, targets)
        
        # 安全系統のメトリクス（targetsを安全ラベルにマッピング）
        # COMPLY -> ALLOW(0), REFUSE -> REFUSE(1), ESCALATE -> ESCALATE(2)
        safety_targets = targets.clone()  # 同じマッピングを使用
        safety_metrics = SafetyMetrics.safety_score(safety_logits, safety_targets)
        
        # 両系統の結果を統合
        result = {}
        
        # タスク系統のメトリクス
        for key, value in task_metrics.items():
            result[f'task_{key}'] = value
        
        # 安全系統のメトリクス
        for key, value in safety_metrics.items():
            result[f'safe_{key}'] = value
        
        # 総合スコア（両系統の平均）
        result['combined_safety_score'] = (task_metrics['safety_score'] + safety_metrics['safety_score']) / 2.0
        
        return result


class SafetyAwareLoss(nn.Module):
    """安全を重視した統合損失関数"""
    
    def __init__(self, 
                 task_loss_weight: float = 1.0,
                 safety_loss_weight: float = 2.0,
                 pet_loss_weight: float = 0.1,
                 safety_penalty_weight: float = 5.0,
                 escalate_reward_weight: float = 2.0,
                 refuse_weight: float = 5.0,
                 escalate_weight: float = 3.0,
                 comply_weight: float = 1.0,
                 use_focal: bool = True):
        super().__init__()
        self.task_loss_weight = task_loss_weight
        self.safety_loss_weight = safety_loss_weight
        self.pet_loss_weight = pet_loss_weight
        self.safety_penalty_weight = safety_penalty_weight
        self.escalate_reward_weight = escalate_reward_weight
        
        # タスク損失（通常の分類）
        self.task_loss = SafetyWeightedLoss(
            refuse_weight=refuse_weight,
            escalate_weight=escalate_weight,
            comply_weight=comply_weight,
            use_focal=use_focal
        )
        
        # 安全損失（REFUSE/ESCALATE/ALLOW）
        self.safety_loss = SafetyWeightedLoss(
            refuse_weight=refuse_weight * 2.0,  # 安全判断ではさらに重く
            escalate_weight=escalate_weight * 1.5,
            comply_weight=comply_weight,
            use_focal=use_focal
        )
    
    def forward(self, 
                task_logits: torch.Tensor,
                safety_logits: torch.Tensor,
                targets: torch.Tensor,
                pet_loss: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            task_logits: [batch_size, num_task_classes] - タスク分類用
            safety_logits: [batch_size, num_safety_classes] - 安全判断用
            targets: [batch_size] - 正解ラベル
            pet_loss: scalar - PET損失
        Returns:
            loss_dict: 各損失の詳細
        """
        # タスク損失
        task_loss = self.task_loss(task_logits, targets)
        
        # 安全損失（targetsを安全ラベルにマッピング）
        # COMPLY -> ALLOW(0), REFUSE -> REFUSE(1), ESCALATE -> ESCALATE(2)
        safety_targets = targets.clone()  # 同じマッピングを使用
        
        safety_loss = self.safety_loss(safety_logits, safety_targets)
        
        # 総損失
        total_loss = (self.task_loss_weight * task_loss + 
                     self.safety_loss_weight * safety_loss + 
                     self.pet_loss_weight * pet_loss)
        
        return {
            'total_loss': total_loss,
            'task_loss': task_loss,
            'safety_loss': safety_loss,
            'pet_loss': pet_loss
        }
    
    def compute_metrics(self, 
                       task_logits: torch.Tensor,
                       safety_logits: torch.Tensor,
                       targets: torch.Tensor) -> Dict[str, float]:
        """安全メトリクスを計算"""
        # タスクロジットでメトリクスを計算
        return SafetyMetrics.safety_score(task_logits, targets)
    
    def compute_safety_loss(self, safety_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """安全ヘッド専用の損失"""
        safety_targets = targets.clone()
        return self.safety_loss(safety_logits, safety_targets)
    
    def compute_safety_penalty(self, task_logits: torch.Tensor, safety_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """危険な従順に対するペナルティ"""
        # 危険なケース（REFUSE/ESCALATEが正解）でCOMPLYを出した場合の重いペナルティ
        task_preds = torch.argmax(task_logits, dim=-1)
        safety_preds = torch.argmax(safety_logits, dim=-1)
        
        # 危険なケースを特定（REFUSE=1, ESCALATE=2が正解）
        dangerous_mask = (targets == 1) | (targets == 2)  # REFUSE or ESCALATE
        
        # 危険なケースでCOMPLY（0）を出した場合のペナルティ
        dangerous_comply_mask = dangerous_mask & (task_preds == 0)
        
        penalty = torch.tensor(0.0, device=task_logits.device)
        if dangerous_comply_mask.any():
            # 危険な従順に対する重いペナルティ
            penalty = self.safety_penalty_weight * dangerous_comply_mask.float().mean()
        
        return penalty
    
    def compute_escalate_reward(self, safety_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """ESCALATE行動に対する報酬"""
        safety_preds = torch.argmax(safety_logits, dim=-1)
        
        # ESCALATE（2）を正しく予測した場合の報酬
        escalate_correct = (targets == 2) & (safety_preds == 2)
        
        reward = torch.tensor(0.0, device=safety_logits.device)
        if escalate_correct.any():
            reward = self.escalate_reward_weight * escalate_correct.float().mean()
        
        return reward
