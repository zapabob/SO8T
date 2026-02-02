#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T Gradient Management Utilities
勾配の適切な管理による重み崩壊防止
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class GradientManager(ABC):
    """勾配管理の抽象基底クラス"""
    
    @abstractmethod
    def apply_gradient_modification(self, model: nn.Module, optimizer: optim.Optimizer) -> Dict[str, Any]:
        """勾配の修正を適用"""
        pass

class GradientClippingManager(GradientManager):
    """勾配クリッピング管理クラス"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        logger.info("GradientClippingManager initialized")
    
    def _get_default_config(self) -> Dict:
        """デフォルト設定を取得"""
        return {
            'max_norm': 1.0,
            'norm_type': 2,
            'adaptive_clipping': True,
            'clipping_history_size': 100,
            'target_percentile': 90,
            'min_clip_norm': 0.1,
            'max_clip_norm': 10.0
        }
    
    def apply_gradient_modification(self, model: nn.Module, optimizer: optim.Optimizer) -> Dict[str, Any]:
        """勾配クリッピングを適用"""
        if self.config['adaptive_clipping']:
            return self._apply_adaptive_clipping(model, optimizer)
        else:
            return self._apply_fixed_clipping(model, optimizer)
    
    def _apply_fixed_clipping(self, model: nn.Module, optimizer: optim.Optimizer) -> Dict[str, Any]:
        """固定クリッピングを適用"""
        total_norm = 0.0
        param_count = 0
        
        # 勾配ノルムを計算
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(self.config['norm_type'])
                total_norm += param_norm.item() ** self.config['norm_type']
                param_count += 1
        
        total_norm = total_norm ** (1. / self.config['norm_type'])
        
        # クリッピングを適用
        clip_coef = 1.0
        if total_norm > self.config['max_norm']:
            clip_coef = self.config['max_norm'] / (total_norm + 1e-6)
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
        
        return {
            'original_norm': total_norm,
            'clipped_norm': total_norm * clip_coef,
            'clip_coef': clip_coef,
            'was_clipped': clip_coef < 1.0
        }
    
    def _apply_adaptive_clipping(self, model: nn.Module, optimizer: optim.Optimizer) -> Dict[str, Any]:
        """適応的クリッピングを適用"""
        # 勾配ノルムを計算
        total_norm = 0.0
        norm_type = self.config.get('norm_type', 2)
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
        
        total_norm = total_norm ** (1. / norm_type)
        
        # 適応的クリッピング閾値を計算
        adaptive_norm = self._calculate_adaptive_norm(total_norm)
        
        # クリッピングを適用
        clip_coef = 1.0
        if total_norm > adaptive_norm:
            clip_coef = adaptive_norm / (total_norm + 1e-6)
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
        
        return {
            'original_norm': total_norm,
            'adaptive_norm': adaptive_norm,
            'clipped_norm': total_norm * clip_coef,
            'clip_coef': clip_coef,
            'was_clipped': clip_coef < 1.0
        }
    
    def _calculate_adaptive_norm(self, current_norm: float) -> float:
        """適応的クリッピング閾値を計算"""
        # 履歴に追加（実際の実装では履歴を保存）
        # ここでは簡略化
        min_clip_norm = self.config.get('min_clip_norm', 0.1)
        max_clip_norm = self.config.get('max_clip_norm', 10.0)
        return max(min_clip_norm, 
                  min(max_clip_norm, 
                      self.config['max_norm']))

class GradientNoiseManager(GradientManager):
    """勾配ノイズ管理クラス"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        logger.info("GradientNoiseManager initialized")
    
    def _get_default_config(self) -> Dict:
        """デフォルト設定を取得"""
        return {
            'noise_scale': 0.01,
            'noise_decay': 0.99,
            'min_noise_scale': 0.001,
            'noise_type': 'gaussian',  # 'gaussian' or 'uniform'
            'per_parameter_noise': True
        }
    
    def apply_gradient_modification(self, model: nn.Module, optimizer: optim.Optimizer) -> Dict[str, Any]:
        """勾配ノイズを適用"""
        noise_stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                # パラメータごとのノイズスケールを計算
                if self.config['per_parameter_noise']:
                    noise_scale = self._calculate_per_parameter_noise_scale(param)
                else:
                    noise_scale = self.config['noise_scale']
                
                # ノイズを生成
                noise_type = self.config.get('noise_type', 'gaussian')
                if noise_type == 'gaussian':
                    noise = torch.randn_like(param.grad) * noise_scale
                else:  # uniform
                    noise = (torch.rand_like(param.grad) - 0.5) * 2 * noise_scale
                
                # 勾配にノイズを追加
                param.grad.data.add_(noise)
                
                noise_stats[name] = {
                    'noise_scale': noise_scale,
                    'noise_norm': torch.norm(noise).item(),
                    'grad_norm_before': torch.norm(param.grad.data - noise).item(),
                    'grad_norm_after': torch.norm(param.grad.data).item()
                }
        
        return noise_stats
    
    def _calculate_per_parameter_noise_scale(self, param: torch.Tensor) -> float:
        """パラメータごとのノイズスケールを計算"""
        # パラメータのサイズに基づいてノイズスケールを調整
        param_size = param.numel()
        base_scale = self.config['noise_scale']
        
        # パラメータサイズが大きいほどノイズスケールを小さく
        size_factor = 1.0 / math.sqrt(param_size)
        
        return base_scale * size_factor

class SO8TGradientManager(GradientManager):
    """SO8T専用の勾配管理クラス"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.clipping_manager = GradientClippingManager(self.config.get('clipping', {}))
        self.noise_manager = GradientNoiseManager(self.config.get('noise', {}))
        logger.info("SO8TGradientManager initialized")
    
    def _get_default_config(self) -> Dict:
        """デフォルト設定を取得"""
        return {
            'rotation_gradient_scale': 0.5,
            'triality_gradient_scale': 1.0,
            'safety_gradient_scale': 2.0,
            'clipping': {
                'max_norm': 1.0,
                'adaptive_clipping': True
            },
            'noise': {
                'noise_scale': 0.01,
                'per_parameter_noise': True
            },
            'layer_specific_scaling': True,
            'preserve_so8_structure': True
        }
    
    def apply_gradient_modification(self, model: nn.Module, optimizer: optim.Optimizer) -> Dict[str, Any]:
        """SO8T専用の勾配修正を適用"""
        results = {
            'layer_specific_scaling': {},
            'clipping_results': {},
            'noise_results': {}
        }
        
        # レイヤー固有のスケーリング
        if self.config['layer_specific_scaling']:
            results['layer_specific_scaling'] = self._apply_layer_specific_scaling(model)
        
        # 勾配クリッピング
        results['clipping_results'] = self.clipping_manager.apply_gradient_modification(model, optimizer)
        
        # 勾配ノイズ
        results['noise_results'] = self.noise_manager.apply_gradient_modification(model, optimizer)
        
        return results
    
    def _apply_layer_specific_scaling(self, model: nn.Module) -> Dict[str, Any]:
        """レイヤー固有のスケーリングを適用"""
        scaling_results = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                # レイヤータイプに基づいてスケーリング係数を決定
                if 'rotation' in name.lower():
                    scale = self.config['rotation_gradient_scale']
                elif 'triality' in name.lower() or 'task_head' in name.lower() or 'safety_head' in name.lower() or 'authority_head' in name.lower():
                    scale = self.config['triality_gradient_scale']
                elif 'safety' in name.lower():
                    scale = self.config['safety_gradient_scale']
                else:
                    scale = 1.0
                
                # 勾配をスケーリング
                if scale != 1.0:
                    param.grad.data.mul_(scale)
                
                scaling_results[name] = {
                    'scale': scale,
                    'grad_norm_before': torch.norm(param.grad.data / scale).item() if scale != 0 else 0,
                    'grad_norm_after': torch.norm(param.grad.data).item()
                }
        
        return scaling_results

class LearningRateScheduler:
    """学習率スケジューラー"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.step_count = 0
        logger.info("LearningRateScheduler initialized")
    
    def _get_default_config(self) -> Dict:
        """デフォルト設定を取得"""
        return {
            'base_lr': 0.001,
            'warmup_steps': 1000,
            'decay_steps': 10000,
            'decay_rate': 0.95,
            'min_lr': 1e-6,
            'scheduler_type': 'exponential',  # 'exponential', 'cosine', 'linear'
            'warmup_type': 'linear'  # 'linear', 'exponential'
        }
    
    def get_learning_rate(self, step: int) -> float:
        """現在のステップでの学習率を取得"""
        self.step_count = step
        
        if step < self.config['warmup_steps']:
            return self._warmup_lr(step)
        else:
            return self._decay_lr(step)
    
    def _warmup_lr(self, step: int) -> float:
        """ウォームアップ期間の学習率"""
        warmup_type = self.config.get('warmup_type', 'linear')
        if warmup_type == 'linear':
            return self.config['base_lr'] * (step / self.config['warmup_steps'])
        else:  # exponential
            return self.config['base_lr'] * (step / self.config['warmup_steps']) ** 2
    
    def _decay_lr(self, step: int) -> float:
        """減衰期間の学習率"""
        decay_step = step - self.config['warmup_steps']
        scheduler_type = self.config.get('scheduler_type', 'exponential')
        
        if scheduler_type == 'exponential':
            lr = self.config['base_lr'] * (self.config['decay_rate'] ** (decay_step / self.config['decay_steps']))
        elif scheduler_type == 'cosine':
            lr = self.config['base_lr'] * 0.5 * (1 + math.cos(math.pi * decay_step / self.config['decay_steps']))
        else:  # linear
            lr = self.config['base_lr'] * (1 - decay_step / self.config['decay_steps'])
        
        return max(lr, self.config.get('min_lr', 1e-6))
    
    def update_optimizer_lr(self, optimizer: optim.Optimizer, step: int):
        """オプティマイザーの学習率を更新"""
        lr = self.get_learning_rate(step)
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

class WeightDecayManager:
    """重み減衰管理クラス"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        logger.info("WeightDecayManager initialized")
    
    def _get_default_config(self) -> Dict:
        """デフォルト設定を取得"""
        return {
            'base_decay': 1e-4,
            'rotation_decay': 1e-5,
            'triality_decay': 1e-4,
            'safety_decay': 1e-3,
            'adaptive_decay': True,
            'decay_schedule': 'constant'  # 'constant', 'linear', 'exponential'
        }
    
    def apply_weight_decay(self, model: nn.Module, step: int = 0) -> Dict[str, Any]:
        """重み減衰を適用"""
        decay_results = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                # レイヤータイプに基づいて減衰率を決定
                if 'rotation' in name.lower():
                    decay_rate = self.config['rotation_decay']
                elif 'triality' in name.lower() or 'task_head' in name.lower() or 'safety_head' in name.lower() or 'authority_head' in name.lower():
                    decay_rate = self.config['triality_decay']
                elif 'safety' in name.lower():
                    decay_rate = self.config['safety_decay']
                else:
                    decay_rate = self.config['base_decay']
                
                # 適応的減衰
                if self.config['adaptive_decay']:
                    decay_rate = self._calculate_adaptive_decay(decay_rate, param, step)
                
                # 重み減衰を適用
                param.data.mul_(1 - decay_rate)
                
                decay_results[name] = {
                    'decay_rate': decay_rate,
                    'weight_norm_before': torch.norm(param.data / (1 - decay_rate)).item() if decay_rate != 1 else 0,
                    'weight_norm_after': torch.norm(param.data).item()
                }
        
        return decay_results
    
    def _calculate_adaptive_decay(self, base_decay: float, param: torch.Tensor, step: int) -> float:
        """適応的重み減衰率を計算"""
        # パラメータのノルムに基づいて減衰率を調整
        param_norm = torch.norm(param.data).item()
        
        # ノルムが大きいほど減衰率を大きく
        norm_factor = min(2.0, max(0.5, param_norm))
        
        return base_decay * norm_factor

def create_gradient_manager(manager_type: str = "so8t", config: Optional[Dict] = None) -> GradientManager:
    """
    勾配マネージャーを作成
    
    Args:
        manager_type: マネージャータイプ ("so8t", "clipping", "noise")
        config: 設定パラメータ
        
    Returns:
        勾配マネージャー
    """
    if manager_type.lower() == "so8t":
        return SO8TGradientManager(config)
    elif manager_type.lower() == "clipping":
        return GradientClippingManager(config)
    elif manager_type.lower() == "noise":
        return GradientNoiseManager(config)
    else:
        raise ValueError(f"Unknown manager type: {manager_type}")

if __name__ == "__main__":
    # テスト用のダミーモデル
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.rotation_layer = nn.Linear(8, 8)
            self.task_head = nn.Linear(10, 3)
            self.safety_head = nn.Linear(10, 3)
    
    model = DummyModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # ダミーの勾配を設定
    for param in model.parameters():
        if param.requires_grad:
            param.grad = torch.randn_like(param.data)
    
    # 勾配マネージャーのテスト
    gradient_manager = create_gradient_manager("so8t")
    results = gradient_manager.apply_gradient_modification(model, optimizer)
    
    print("Gradient Management Test")
    print("=" * 50)
    print(f"Layer specific scaling: {len(results['layer_specific_scaling'])} layers")
    print(f"Clipping applied: {results['clipping_results']['was_clipped']}")
    print(f"Noise applied to: {len(results['noise_results'])} parameters")
    
    # 学習率スケジューラーのテスト
    lr_scheduler = LearningRateScheduler()
    lr = lr_scheduler.get_learning_rate(100)
    print(f"Learning rate at step 100: {lr:.6f}")
    
    # 重み減衰マネージャーのテスト
    weight_decay_manager = WeightDecayManager()
    decay_results = weight_decay_manager.apply_weight_decay(model, 100)
    print(f"Weight decay applied to: {len(decay_results)} parameters")
