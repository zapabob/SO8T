#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T Weight Stability Demo
重み安定性管理システムのデモンストレーション
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from datetime import datetime

from utils.weight_stability_manager import create_weight_stability_manager
from utils.weight_initialization import initialize_so8t_model
from utils.gradient_management import create_gradient_manager, LearningRateScheduler, WeightDecayManager

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DummySO8TModel(nn.Module):
    """テスト用のSO8Tモデル"""
    
    def __init__(self, input_size=10, hidden_size=20, output_size=3):
        super().__init__()
        
        # 標準的な層
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # SO8T特有の層
        self.rotation_layer = nn.Linear(8, 8)  # SO(8)回転層
        self.task_head = nn.Linear(hidden_size, 3)  # タスクヘッド
        self.safety_head = nn.Linear(hidden_size, 3)  # 安全性ヘッド
        self.authority_head = nn.Linear(hidden_size, 3)  # 権威ヘッド
        self.safety_judgment = nn.Linear(output_size, 1)  # 安全性判定
        
        # 活性化関数
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # 標準的な前向き伝播
        x = self.relu(self.input_layer(x))
        x = self.dropout(x)
        x = self.relu(self.hidden_layer(x))
        x = self.dropout(x)
        
        # 出力層
        output = self.output_layer(x)
        
        # SO8T特有の処理
        rotation_output = self.rotation_layer(x[:, :8])  # 最初の8次元を回転
        task_output = self.task_head(x)
        safety_output = self.safety_head(x)
        authority_output = self.authority_head(x)
        safety_judgment = self.safety_judgment(output)
        
        return {
            'output': output,
            'rotation': rotation_output,
            'task': task_output,
            'safety': safety_output,
            'authority': authority_output,
            'safety_judgment': safety_judgment
        }

def demo_weight_initialization():
    """重み初期化のデモ"""
    print("\n" + "="*60)
    print("[INIT] SO8T Weight Initialization Demo")
    print("="*60)
    
    # モデル作成
    model = DummySO8TModel()
    
    # 初期化前の重み統計
    print("\n[INITIAL] Before initialization:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            weight_data = param.data.cpu().numpy()
            print(f"  {name}: mean={np.mean(weight_data):.6f}, std={np.std(weight_data):.6f}, norm={np.linalg.norm(weight_data):.6f}")
    
    # SO8T初期化を適用
    initialized_model = initialize_so8t_model(model)
    
    # 初期化後の重み統計
    print("\n[AFTER] After SO8T initialization:")
    for name, param in initialized_model.named_parameters():
        if param.requires_grad:
            weight_data = param.data.cpu().numpy()
            print(f"  {name}: mean={np.mean(weight_data):.6f}, std={np.std(weight_data):.6f}, norm={np.linalg.norm(weight_data):.6f}")
    
    print("\n[OK] Weight initialization completed!")

def demo_weight_monitoring():
    """重み監視のデモ"""
    print("\n" + "="*60)
    print("[MONITOR] SO8T Weight Monitoring Demo")
    print("="*60)
    
    # モデル作成と初期化
    model = DummySO8TModel()
    model = initialize_so8t_model(model)
    
    # 重み安定性マネージャーを作成
    stability_manager = create_weight_stability_manager(model, "so8t")
    
    # オプティマイザーを作成
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # ダミーのデータ
    x = torch.randn(32, 10)
    y = torch.randn(32, 3)
    
    print("\n[STEP] Simulating training steps...")
    
    for step in range(0, 1000, 100):
        # 前向き伝播
        outputs = model(x)
        loss = nn.MSELoss()(outputs['output'], y)
        
        # 逆伝播
        optimizer.zero_grad()
        loss.backward()
        
        # 重み監視
        if step % 100 == 0:
            stats = stability_manager.monitor_so8_weights(step)
            print(f"\n  Step {step}:")
            print(f"    Loss: {loss.item():.6f}")
            
            if 'so8_specific' in stats:
                for layer_name, layer_stats in stats['so8_specific'].items():
                    print(f"    {layer_name}: {layer_stats}")
            
            if 'anomalies' in stats and stats['anomalies']:
                print(f"    [WARNING] Anomalies: {stats['anomalies']}")
        
        # 勾配クリッピング
        stability_manager.apply_gradient_clipping(optimizer)
        
        # 重み減衰
        stability_manager.apply_weight_decay(optimizer)
        
        # オプティマイザーステップ
        optimizer.step()
    
    # 安定性レポートを生成
    report = stability_manager.get_stability_report()
    print(f"\n[REPORT] Stability report generated:")
    print(f"  Total steps monitored: {report['total_steps']}")
    print(f"  Recommendations: {len(report.get('recommendations', []))}")
    
    for rec in report.get('recommendations', []):
        print(f"    - {rec}")
    
    print("\n[OK] Weight monitoring completed!")

def demo_gradient_management():
    """勾配管理のデモ"""
    print("\n" + "="*60)
    print("[GRADIENT] SO8T Gradient Management Demo")
    print("="*60)
    
    # モデル作成
    model = DummySO8TModel()
    model = initialize_so8t_model(model)
    
    # 勾配マネージャーを作成
    gradient_manager = create_gradient_manager("so8t")
    
    # 学習率スケジューラーを作成
    lr_scheduler = LearningRateScheduler({
        'base_lr': 0.001,
        'warmup_steps': 100,
        'decay_steps': 500,
        'decay_rate': 0.95
    })
    
    # 重み減衰マネージャーを作成
    weight_decay_manager = WeightDecayManager()
    
    # オプティマイザーを作成
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # ダミーのデータ
    x = torch.randn(16, 10)
    y = torch.randn(16, 3)
    
    print("\n[STEP] Simulating training with gradient management...")
    
    for step in range(500):
        # 学習率を更新
        lr_scheduler.update_optimizer_lr(optimizer, step)
        
        # 前向き伝播
        outputs = model(x)
        loss = nn.MSELoss()(outputs['output'], y)
        
        # 逆伝播
        optimizer.zero_grad()
        loss.backward()
        
        # 勾配管理を適用
        if step % 50 == 0:
            grad_results = gradient_manager.apply_gradient_modification(model, optimizer)
            print(f"\n  Step {step}:")
            print(f"    Loss: {loss.item():.6f}")
            print(f"    Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"    Clipping applied: {grad_results['clipping_results']['was_clipped']}")
            print(f"    Noise applied to: {len(grad_results['noise_results'])} parameters")
        
        # 重み減衰を適用
        if step % 10 == 0:
            decay_results = weight_decay_manager.apply_weight_decay(model, step)
        
        # オプティマイザーステップ
        optimizer.step()
    
    print("\n[OK] Gradient management completed!")

def demo_weight_collapse_prevention():
    """重み崩壊防止のデモ"""
    print("\n" + "="*60)
    print("[PREVENTION] SO8T Weight Collapse Prevention Demo")
    print("="*60)
    
    # モデル作成
    model = DummySO8TModel()
    model = initialize_so8t_model(model)
    
    # 重み安定性マネージャーを作成
    stability_manager = create_weight_stability_manager(model, "so8t", {
        'gradient_clip_norm': 0.5,
        'weight_decay': 1e-4,
        'stability_threshold': 0.01,
        'explosion_threshold': 5.0,
        'monitoring_frequency': 50
    })
    
    # 勾配マネージャーを作成
    gradient_manager = create_gradient_manager("so8t", {
        'rotation_gradient_scale': 0.3,
        'triality_gradient_scale': 0.8,
        'safety_gradient_scale': 1.5,
        'clipping': {'max_norm': 0.5, 'adaptive_clipping': True},
        'noise': {'noise_scale': 0.005, 'per_parameter_noise': True}
    })
    
    # 学習率スケジューラーを作成
    lr_scheduler = LearningRateScheduler({
        'base_lr': 0.0005,
        'warmup_steps': 200,
        'decay_steps': 1000,
        'decay_rate': 0.9,
        'min_lr': 1e-6
    })
    
    # オプティマイザーを作成
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    # ダミーのデータ
    x = torch.randn(64, 10)
    y = torch.randn(64, 3)
    
    print("\n[STEP] Simulating training with collapse prevention...")
    
    collapse_detected = False
    
    for step in range(2000):
        # 学習率を更新
        lr_scheduler.update_optimizer_lr(optimizer, step)
        
        # 前向き伝播
        outputs = model(x)
        loss = nn.MSELoss()(outputs['output'], y)
        
        # 逆伝播
        optimizer.zero_grad()
        loss.backward()
        
        # 重み監視
        if step % 50 == 0:
            stats = stability_manager.monitor_so8_weights(step)
            
            if 'anomalies' in stats and stats['anomalies']:
                print(f"\n  [WARNING] Step {step}: Anomalies detected!")
                for anomaly in stats['anomalies']:
                    print(f"    - {anomaly}")
                collapse_detected = True
        
        # 勾配管理を適用
        grad_results = gradient_manager.apply_gradient_modification(model, optimizer)
        
        # 重み減衰を適用
        stability_manager.apply_weight_decay(optimizer)
        
        # オプティマイザーステップ
        optimizer.step()
        
        # 定期的にステータスを表示
        if step % 200 == 0:
            print(f"\n  Step {step}:")
            print(f"    Loss: {loss.item():.6f}")
            print(f"    Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"    Clipping applied: {grad_results['clipping_results']['was_clipped']}")
            
            # 重みの統計を表示
            total_norm = 0
            for param in model.parameters():
                if param.requires_grad:
                    total_norm += param.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            print(f"    Total weight norm: {total_norm:.6f}")
    
    # 最終レポート
    final_report = stability_manager.get_stability_report()
    print(f"\n[FINAL REPORT] Training completed:")
    print(f"  Total steps: {final_report['total_steps']}")
    print(f"  Collapse detected: {collapse_detected}")
    print(f"  Recommendations: {len(final_report.get('recommendations', []))}")
    
    for rec in final_report.get('recommendations', []):
        print(f"    - {rec}")
    
    print("\n[OK] Weight collapse prevention completed!")

def main():
    """メイン関数"""
    print("[START] SO8T Weight Stability Management System Demo")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # デモを実行
        demo_weight_initialization()
        demo_weight_monitoring()
        demo_gradient_management()
        demo_weight_collapse_prevention()
        
        print("\n" + "="*80)
        print("[SUCCESS] All demos completed successfully!")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n[ERROR] Demo failed: {e}")
        raise

if __name__ == "__main__":
    main()
