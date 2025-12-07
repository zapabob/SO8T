#!/usr/bin/env python3
"""
Autosave and Recovery Tests
オートセーブと復旧機能の軽量テスト
"""

import tempfile
import time
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import Mock

from shared.data_backup import SessionCheckpointManager


class DummyModel(nn.Module):
    """テスト用のダミーモデル"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)


def test_session_checkpoint_manager():
    """セッション管理の基本機能をテスト"""
    print("🧪 Testing SessionCheckpointManager...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        
        # セッション管理を初期化
        session_id = "test_session_123"
        mgr = SessionCheckpointManager(output_dir, session_id=session_id)
        
        # セッション情報を確認
        info = mgr.get_session_info()
        assert info['session_id'] == session_id
        assert not info['has_checkpoint']
        print("✅ Session initialization successful")
        
        # ダミーモデルとオプティマイザーを作成
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scaler = Mock()
        scaler.state_dict.return_value = {'scale': 1.0}
        scheduler = Mock()
        scheduler.state_dict.return_value = {'last_epoch': 0}
        
        # チェックポイントを保存
        meta = {'epoch': 1, 'step': 100, 'test': True}
        checkpoint_path = mgr.save(model, optimizer, scaler, scheduler, meta)
        
        assert checkpoint_path.exists()
        print(f"✅ Checkpoint saved: {checkpoint_path}")
        
        # チェックポイントを読み込み
        loaded_data = mgr.load_latest()
        assert loaded_data is not None
        assert loaded_data['session_id'] == session_id
        assert loaded_data['meta']['epoch'] == 1
        assert loaded_data['meta']['step'] == 100
        print("✅ Checkpoint loading successful")
        
        # 新しいモデルに状態を読み込み
        new_model = DummyModel()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        
        new_model.load_state_dict(loaded_data['model_state_dict'])
        new_optimizer.load_state_dict(loaded_data['optimizer_state_dict'])
        print("✅ Model state restoration successful")
        
        # セッション情報を再確認
        info_after = mgr.get_session_info()
        assert info_after['has_checkpoint']
        assert info_after['latest_timestamp'] is not None
        print("✅ Session info updated correctly")


def test_backup_rotation():
    """バックアップローテーションをテスト"""
    print("🧪 Testing backup rotation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        
        # 最大3個のバックアップでテスト
        mgr = SessionCheckpointManager(output_dir, max_backups=3)
        
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scaler = Mock()
        scaler.state_dict.return_value = {'scale': 1.0}
        scheduler = Mock()
        scheduler.state_dict.return_value = {'last_epoch': 0}
        
        # 5個のチェックポイントを保存
        for i in range(5):
            meta = {'epoch': i, 'step': i * 100}
            mgr.save(model, optimizer, scaler, scheduler, meta)
            time.sleep(0.1)  # タイムスタンプを確実に変える
        
        # バックアップファイル数を確認
        autosave_dir = output_dir / "autosave"
        checkpoint_files = list(autosave_dir.glob(f"autosave_{mgr.session_id}_*.pt"))
        
        # 最大3個までしか残っていないはず
        assert len(checkpoint_files) <= 3, f"Expected ≤3 backups, got {len(checkpoint_files)}"
        print(f"✅ Backup rotation working: {len(checkpoint_files)} files remaining")
        
        # 最新のチェックポイントが存在することを確認
        latest = mgr.load_latest()
        assert latest is not None
        assert latest['meta']['epoch'] == 4  # 最後に保存したエポック
        print("✅ Latest checkpoint accessible after rotation")


def test_emergency_save():
    """緊急保存機能をテスト"""
    print("🧪 Testing emergency save...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        
        mgr = SessionCheckpointManager(output_dir)
        
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scaler = Mock()
        scaler.state_dict.return_value = {'scale': 1.0}
        scheduler = Mock()
        scheduler.state_dict.return_value = {'last_epoch': 0}
        
        # 緊急保存をシミュレート
        meta = {'epoch': 999, 'step': 9999, 'emergency': True}
        
        # 緊急保存フラグを手動で設定
        with mgr._emergency_save_lock:
            mgr._emergency_save_requested = True
        
        # 緊急保存を実行
        emergency_path = mgr.emergency_save(model, optimizer, scaler, scheduler, meta)
        
        assert emergency_path is not None
        assert emergency_path.exists()
        print(f"✅ Emergency save successful: {emergency_path}")
        
        # 緊急保存フラグがリセットされていることを確認
        assert not mgr.check_emergency_save()
        print("✅ Emergency save flag reset correctly")


def test_dual_metrics():
    """両系統KPIの計算をテスト"""
    print("🧪 Testing dual metrics calculation...")
    
    from safety_losses import SafetyMetrics
    
    # ダミーデータを作成
    batch_size = 10
    num_classes = 3
    
    # ランダムなロジットとターゲット
    task_logits = torch.randn(batch_size, num_classes)
    safety_logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # 両系統のメトリクスを計算
    dual_metrics = SafetyMetrics.dual_safety_metrics(task_logits, safety_logits, targets)
    
    # 必要なキーが存在することを確認
    expected_keys = [
        'task_refuse_recall', 'task_escalate_recall', 'task_overcompliance_rate', 'task_safety_score',
        'safe_refuse_recall', 'safe_escalate_recall', 'safe_overcompliance_rate', 'safe_safety_score',
        'combined_safety_score'
    ]
    
    for key in expected_keys:
        assert key in dual_metrics, f"Missing key: {key}"
        assert isinstance(dual_metrics[key], (int, float)), f"Invalid type for {key}: {type(dual_metrics[key])}"
    
    print("✅ Dual metrics calculation successful")
    print(f"   Combined safety score: {dual_metrics['combined_safety_score']:.4f}")


def main():
    """全テストを実行"""
    print("[START] Running Safety-Aware SO8T Autosave/Recovery Tests")
    print("=" * 60)
    
    try:
        test_session_checkpoint_manager()
        print()
        
        test_backup_rotation()
        print()
        
        test_emergency_save()
        print()
        
        test_dual_metrics()
        print()
        
        print("🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
