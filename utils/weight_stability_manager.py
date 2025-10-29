#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T Weight Stability Manager
重み崩壊を防ぐための包括的な管理システム
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)

class WeightStabilityManager:
    """
    重みの安定性を管理し、崩壊を防ぐためのマネージャー
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: Optional[Dict] = None):
        """
        Args:
            model: 監視対象のモデル
            config: 設定パラメータ
        """
        self.model = model
        self.config = config or self._get_default_config()
        
        # 重みの履歴を保存
        self.weight_history = []
        self.gradient_history = []
        self.stability_metrics = []
        
        # 重みの統計情報
        self.weight_stats = {
            'mean': [],
            'std': [],
            'min': [],
            'max': [],
            'norm': []
        }
        
        # 勾配の統計情報
        self.gradient_stats = {
            'mean': [],
            'std': [],
            'min': [],
            'max': [],
            'norm': []
        }
        
        logger.info("WeightStabilityManager initialized")
    
    def _get_default_config(self) -> Dict:
        """デフォルト設定を取得"""
        return {
            'gradient_clip_norm': 1.0,
            'weight_decay': 1e-4,
            'learning_rate_warmup': 1000,
            'learning_rate_decay': 0.95,
            'stability_threshold': 0.1,
            'explosion_threshold': 10.0,
            'monitoring_frequency': 100,
            'save_frequency': 1000,
            'checkpoint_dir': 'checkpoints/weight_stability'
        }
    
    def monitor_weights(self, step: int) -> Dict[str, Any]:
        """
        重みの状態を監視し、統計情報を記録
        
        Args:
            step: 現在のステップ数
            
        Returns:
            監視結果の辞書
        """
        if step % self.config['monitoring_frequency'] != 0:
            return {}
        
        current_stats = {}
        
        # 各レイヤーの重みを監視
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # 重みの統計
                weight_data = param.data.cpu().numpy()
                weight_mean = np.mean(weight_data)
                weight_std = np.std(weight_data)
                weight_min = np.min(weight_data)
                weight_max = np.max(weight_data)
                weight_norm = np.linalg.norm(weight_data)
                
                # 勾配の統計
                grad_data = param.grad.data.cpu().numpy()
                grad_mean = np.mean(grad_data)
                grad_std = np.std(grad_data)
                grad_min = np.min(grad_data)
                grad_max = np.max(grad_data)
                grad_norm = np.linalg.norm(grad_data)
                
                # 統計情報を保存（float32をfloatに変換）
                layer_stats = {
                    'weight_mean': float(weight_mean),
                    'weight_std': float(weight_std),
                    'weight_min': float(weight_min),
                    'weight_max': float(weight_max),
                    'weight_norm': float(weight_norm),
                    'grad_mean': float(grad_mean),
                    'grad_std': float(grad_std),
                    'grad_min': float(grad_min),
                    'grad_max': float(grad_max),
                    'grad_norm': float(grad_norm),
                    'step': step,
                    'timestamp': datetime.now().isoformat()
                }
                
                current_stats[name] = layer_stats
                
                # 履歴に追加（float32をfloatに変換）
                self.weight_stats['mean'].append(float(weight_mean))
                self.weight_stats['std'].append(float(weight_std))
                self.weight_stats['min'].append(float(weight_min))
                self.weight_stats['max'].append(float(weight_max))
                self.weight_stats['norm'].append(float(weight_norm))
                
                self.gradient_stats['mean'].append(float(grad_mean))
                self.gradient_stats['std'].append(float(grad_std))
                self.gradient_stats['min'].append(float(grad_min))
                self.gradient_stats['max'].append(float(grad_max))
                self.gradient_stats['norm'].append(float(grad_norm))
        
        # 安定性メトリクスを計算
        stability_metrics = self._calculate_stability_metrics(current_stats)
        current_stats['stability_metrics'] = stability_metrics
        
        # 履歴に追加
        self.stability_metrics.append(stability_metrics)
        
        # 異常検知
        anomalies = self._detect_anomalies(current_stats)
        if anomalies:
            logger.warning(f"Weight anomalies detected at step {step}: {anomalies}")
            current_stats['anomalies'] = anomalies
        
        # 定期的にチェックポイントを保存
        if step % self.config.get('save_frequency', 100) == 0:
            self._save_checkpoint(step)
        
        return current_stats
    
    def _calculate_stability_metrics(self, current_stats: Dict) -> Dict[str, float]:
        """安定性メトリクスを計算"""
        metrics = {}
        
        if not current_stats:
            return metrics
        
        # 重みの分散を計算
        weight_stds = [stats['weight_std'] for stats in current_stats.values() 
                      if isinstance(stats, dict) and 'weight_std' in stats]
        if weight_stds:
            metrics['weight_std_mean'] = float(np.mean(weight_stds))
            metrics['weight_std_max'] = float(np.max(weight_stds))
        
        # 勾配の分散を計算
        grad_stds = [stats['grad_std'] for stats in current_stats.values() 
                    if isinstance(stats, dict) and 'grad_std' in stats]
        if grad_stds:
            metrics['grad_std_mean'] = float(np.mean(grad_stds))
            metrics['grad_std_max'] = float(np.max(grad_stds))
        
        # 重みのノルムを計算
        weight_norms = [stats['weight_norm'] for stats in current_stats.values() 
                       if isinstance(stats, dict) and 'weight_norm' in stats]
        if weight_norms:
            metrics['weight_norm_mean'] = float(np.mean(weight_norms))
            metrics['weight_norm_max'] = float(np.max(weight_norms))
        
        # 勾配のノルムを計算
        grad_norms = [stats['grad_norm'] for stats in current_stats.values() 
                     if isinstance(stats, dict) and 'grad_norm' in stats]
        if grad_norms:
            metrics['grad_norm_mean'] = float(np.mean(grad_norms))
            metrics['grad_norm_max'] = float(np.max(grad_norms))
        
        return metrics
    
    def _detect_anomalies(self, current_stats: Dict) -> List[str]:
        """重みの異常を検知"""
        anomalies = []
        
        for name, stats in current_stats.items():
            if not isinstance(stats, dict):
                continue
            
            # 重みの爆発を検知
            if 'weight_norm' in stats:
                if stats['weight_norm'] > self.config['explosion_threshold']:
                    anomalies.append(f"Weight explosion in {name}: {stats['weight_norm']:.4f}")
            
            # 重みの消失を検知
            if 'weight_std' in stats:
                if stats['weight_std'] < self.config['stability_threshold']:
                    anomalies.append(f"Weight collapse in {name}: {stats['weight_std']:.4f}")
            
            # 勾配の爆発を検知
            if 'grad_norm' in stats:
                if stats['grad_norm'] > self.config['explosion_threshold']:
                    anomalies.append(f"Gradient explosion in {name}: {stats['grad_norm']:.4f}")
        
        return anomalies
    
    def apply_gradient_clipping(self, optimizer: torch.optim.Optimizer) -> float:
        """
        勾配クリッピングを適用
        
        Args:
            optimizer: 最適化器
            
        Returns:
            クリッピング前の勾配ノルム
        """
        total_norm = 0.0
        param_count = 0
        
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        total_norm = total_norm ** (1. / 2)
        
        # 勾配クリッピングを適用
        if total_norm > self.config['gradient_clip_norm']:
            clip_coef = self.config['gradient_clip_norm'] / (total_norm + 1e-6)
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
            
            logger.info(f"Gradient clipped: {total_norm:.4f} -> {self.config['gradient_clip_norm']:.4f}")
        
        return total_norm
    
    def apply_weight_decay(self, optimizer: torch.optim.Optimizer):
        """重み減衰を適用"""
        for param in self.model.parameters():
            if param.requires_grad:
                param.data.mul_(1 - self.config['weight_decay'])
    
    def get_learning_rate_schedule(self, step: int, base_lr: float) -> float:
        """
        学習率スケジュールを取得
        
        Args:
            step: 現在のステップ数
            base_lr: ベース学習率
            
        Returns:
            調整された学習率
        """
        # ウォームアップ
        if step < self.config['learning_rate_warmup']:
            return base_lr * (step / self.config['learning_rate_warmup'])
        
        # 指数減衰
        decay_steps = step - self.config['learning_rate_warmup']
        return base_lr * (self.config['learning_rate_decay'] ** decay_steps)
    
    def _save_checkpoint(self, step: int):
        """チェックポイントを保存"""
        os.makedirs(self.config.get('checkpoint_dir', 'checkpoints/weight_stability'), exist_ok=True)
        
        checkpoint = {
            'step': step,
            'weight_stats': self.weight_stats,
            'gradient_stats': self.gradient_stats,
            'stability_metrics': self.stability_metrics,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = os.path.join(
            self.config['checkpoint_dir'], 
            f'weight_stability_checkpoint_{step}.json'
        )
        
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Weight stability checkpoint saved: {checkpoint_path}")
    
    def get_stability_report(self) -> Dict[str, Any]:
        """安定性レポートを生成"""
        if not self.stability_metrics:
            return {"error": "No stability metrics available"}
        
        report = {
            'total_steps': len(self.stability_metrics),
            'weight_stats_summary': {},
            'gradient_stats_summary': {},
            'stability_trends': {},
            'recommendations': []
        }
        
        # 重み統計の要約
        for key, values in self.weight_stats.items():
            if values:
                report['weight_stats_summary'][key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # 勾配統計の要約
        for key, values in self.gradient_stats.items():
            if values:
                report['gradient_stats_summary'][key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # 安定性トレンドの分析
        if len(self.stability_metrics) > 1:
            for key in self.stability_metrics[0].keys():
                values = [m[key] for m in self.stability_metrics if key in m]
                if values:
                    report['stability_trends'][key] = {
                        'trend': 'increasing' if values[-1] > values[0] else 'decreasing',
                        'volatility': np.std(values),
                        'current': values[-1]
                    }
        
        # 推奨事項を生成
        report['recommendations'] = self._generate_recommendations(report)
        
        return report
    
    def _generate_recommendations(self, report: Dict) -> List[str]:
        """推奨事項を生成"""
        recommendations = []
        
        # 重みの分散が小さい場合
        if 'weight_std_mean' in report.get('stability_trends', {}):
            weight_std = report['stability_trends']['weight_std_mean']['current']
            if weight_std < 0.01:
                recommendations.append("重みの分散が非常に小さいです。学習率を上げるか、重み初期化を見直してください。")
        
        # 勾配の分散が大きい場合
        if 'grad_std_mean' in report.get('stability_trends', {}):
            grad_std = report['stability_trends']['grad_std_mean']['current']
            if grad_std > 1.0:
                recommendations.append("勾配の分散が大きすぎます。勾配クリッピングを強化してください。")
        
        # 重みのノルムが大きい場合
        if 'weight_norm_mean' in report.get('stability_trends', {}):
            weight_norm = report['stability_trends']['weight_norm_mean']['current']
            if weight_norm > 10.0:
                recommendations.append("重みのノルムが大きすぎます。重み減衰を強化してください。")
        
        return recommendations

class SO8TWeightStabilityManager(WeightStabilityManager):
    """
    SO8T専用の重み安定性マネージャー
    SO(8)群の特性を考慮した重み管理
    """
    
    def __init__(self, model: nn.Module, config: Optional[Dict] = None):
        super().__init__(model, config)
        self.so8_config = self._get_so8_config()
    
    def _get_so8_config(self) -> Dict:
        """SO8T専用設定を取得"""
        return {
            'rotation_weight_decay': 1e-5,  # 回転重みの減衰率
            'triality_weight_decay': 1e-4,  # 三性重みの減衰率
            'safety_weight_decay': 1e-3,    # 安全性重みの減衰率
            'rotation_gradient_clip': 0.5,  # 回転勾配のクリッピング
            'triality_gradient_clip': 1.0,  # 三性勾配のクリッピング
            'safety_gradient_clip': 2.0,    # 安全性勾配のクリッピング
        }
    
    def monitor_so8_weights(self, step: int) -> Dict[str, Any]:
        """SO8T専用の重み監視"""
        base_stats = self.monitor_weights(step)
        
        # SO8T特有の重みを監視
        so8_stats = {}
        
        for name, param in self.model.named_parameters():
            if 'rotation' in name.lower():
                so8_stats[f"{name}_rotation"] = self._analyze_rotation_weights(param)
            elif 'triality' in name.lower() or 'task_head' in name.lower() or 'safety_head' in name.lower() or 'authority_head' in name.lower():
                so8_stats[f"{name}_triality"] = self._analyze_triality_weights(param)
            elif 'safety' in name.lower():
                so8_stats[f"{name}_safety"] = self._analyze_safety_weights(param)
        
        base_stats['so8_specific'] = so8_stats
        return base_stats
    
    def _analyze_rotation_weights(self, param: torch.Tensor) -> Dict[str, float]:
        """回転重みの分析"""
        data = param.data.cpu().numpy()
        
        # 回転行列の特性をチェック
        if data.ndim >= 2:
            # 直交性のチェック
            if data.shape[-1] == data.shape[-2]:  # 正方行列の場合
                orthogonality = np.linalg.norm(data @ data.T - np.eye(data.shape[-1]))
            else:
                orthogonality = 0.0
        else:
            orthogonality = 0.0
        
        return {
            'orthogonality_error': orthogonality,
            'determinant': np.linalg.det(data) if data.ndim == 2 and data.shape[0] == data.shape[1] else 0.0,
            'condition_number': np.linalg.cond(data) if data.ndim == 2 else 0.0,
            'norm': np.linalg.norm(data)
        }
    
    def _analyze_triality_weights(self, param: torch.Tensor) -> Dict[str, float]:
        """三性重みの分析"""
        data = param.data.cpu().numpy()
        
        return {
            'sparsity': np.mean(np.abs(data) < 1e-6),  # スパース性
            'entropy': -np.sum(data * np.log(np.abs(data) + 1e-8)),  # エントロピー
            'norm': np.linalg.norm(data),
            'max_activation': np.max(np.abs(data))
        }
    
    def _analyze_safety_weights(self, param: torch.Tensor) -> Dict[str, float]:
        """安全性重みの分析"""
        data = param.data.cpu().numpy()
        
        return {
            'bias_toward_safety': np.mean(data > 0),  # 安全性への偏り
            'confidence': np.std(data),  # 信頼度
            'norm': np.linalg.norm(data),
            'max_weight': np.max(np.abs(data))
        }
    
    def apply_so8_gradient_clipping(self, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """SO8T専用の勾配クリッピング"""
        clipping_results = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                
                if 'rotation' in name.lower():
                    clip_norm = self.so8_config['rotation_gradient_clip']
                elif 'triality' in name.lower() or 'task_head' in name.lower() or 'safety_head' in name.lower() or 'authority_head' in name.lower():
                    clip_norm = self.so8_config['triality_gradient_clip']
                elif 'safety' in name.lower():
                    clip_norm = self.so8_config['safety_gradient_clip']
                else:
                    clip_norm = self.config['gradient_clip_norm']
                
                if grad_norm > clip_norm:
                    clip_coef = clip_norm / (grad_norm + 1e-6)
                    param.grad.data.mul_(clip_coef)
                    clipping_results[name] = {'original': grad_norm, 'clipped': clip_norm}
        
        return clipping_results

def create_weight_stability_manager(model: nn.Module, 
                                  model_type: str = "standard",
                                  config: Optional[Dict] = None) -> WeightStabilityManager:
    """
    重み安定性マネージャーを作成
    
    Args:
        model: 監視対象のモデル
        model_type: モデルタイプ ("standard" or "so8t")
        config: 設定パラメータ
        
    Returns:
        重み安定性マネージャー
    """
    if model_type.lower() == "so8t":
        return SO8TWeightStabilityManager(model, config)
    else:
        return WeightStabilityManager(model, config)

if __name__ == "__main__":
    # テスト用のダミーモデル
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)
            self.rotation = nn.Linear(8, 8)
            self.safety_head = nn.Linear(5, 3)
    
    model = DummyModel()
    manager = create_weight_stability_manager(model, "so8t")
    
    # テスト実行
    print("Weight Stability Manager Test")
    print("=" * 50)
    
    # ダミーの勾配を設定
    for param in model.parameters():
        if param.requires_grad:
            param.grad = torch.randn_like(param.data)
    
    # 重み監視
    stats = manager.monitor_so8_weights(100)
    print(f"Monitoring results: {len(stats)} layers monitored")
    
    # 勾配クリッピング
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    clipping_results = manager.apply_so8_gradient_clipping(optimizer)
    print(f"Gradient clipping applied to {len(clipping_results)} parameters")
    
    # 安定性レポート
    report = manager.get_stability_report()
    print(f"Stability report generated with {len(report.get('recommendations', []))} recommendations")
