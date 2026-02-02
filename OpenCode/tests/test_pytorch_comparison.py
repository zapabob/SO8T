"""
PyTorchモデルとの精度比較テスト

SO8Tモデルと標準PyTorch実装の比較:
- フォワードパスの数値精度
- 勾配計算の正確性
- 損失関数の一貫性
- バックプロパゲーションの検証
- 数値安定性の比較
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Any
import math
import logging
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import seaborn as sns

# SO8T関連のインポート
from models.so8t_group_structure import SO8Rotation, NonCommutativeGate, PETRegularization
from so8t-mmllm.src.modules.rotation_gate import SO8TRotationGate, apply_block_rotation
from so8t-mmllm.src.losses.pet import PETLoss, pet_penalty
from models.so8t_mlp import SO8TMLP
from models.so8t_attention import SO8TAttention, SO8TRotaryEmbedding

logger = logging.getLogger(__name__)


class StandardPyTorchImplementation:
    """標準PyTorch実装（比較用）"""
    
    def __init__(self, hidden_size: int = 64):
        self.hidden_size = hidden_size
        
        # 標準線形層
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # 標準MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # 標準アテンション
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        
        # 標準正則化
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """標準フォワードパス"""
        # 線形変換
        x = self.linear(x)
        
        # アテンション
        x_attn, _ = self.attention(x, x, x)
        x = x + x_attn
        
        # レイヤーノーマライゼーション
        x = self.layer_norm(x)
        
        # MLP
        x_mlp = self.mlp(x)
        x = x + x_mlp
        
        # ドロップアウト
        x = self.dropout(x)
        
        return x


class TestForwardPassComparison:
    """フォワードパス比較テスト"""
    
    def test_linear_transformation_accuracy(self):
        """線形変換の精度比較"""
        print("\n[TEST] 線形変換精度比較...")
        
        hidden_size = 64
        batch_size, seq_len = 4, 16
        
        # SO8T回転
        so8t_rotation = SO8Rotation(hidden_size=hidden_size, rotation_dim=8)
        
        # 標準PyTorch線形層
        standard_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # テスト用の入力
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        with torch.no_grad():
            # SO8T回転
            R = so8t_rotation._generate_rotation_matrix()
            x_so8t = torch.matmul(x, R.T)
            
            # 標準線形変換
            x_standard = standard_linear(x)
            
            # 出力の形状確認
            assert x_so8t.shape == x.shape
            assert x_standard.shape == x.shape
            
            # 数値的安定性の確認
            assert torch.isfinite(x_so8t).all()
            assert torch.isfinite(x_standard).all()
            
            # ノルムの比較
            norm_so8t = torch.norm(x_so8t, dim=-1)
            norm_standard = torch.norm(x_standard, dim=-1)
            norm_original = torch.norm(x, dim=-1)
            
            # SO8T回転はノルムを保持する
            norm_preservation_error = torch.mean(torch.abs(norm_so8t - norm_original))
            assert norm_preservation_error < 1e-5, f"SO8T回転のノルム保持エラー: {norm_preservation_error}"
            
        print("[OK] 線形変換精度比較完了")
    
    def test_mlp_comparison(self):
        """MLP比較テスト"""
        print("\n[TEST] MLP比較テスト...")
        
        hidden_size = 64
        batch_size, seq_len = 4, 16
        
        # SO8T MLP
        so8t_mlp = SO8TMLP(
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
            group_structure=True
        )
        
        # 標準PyTorch MLP
        standard_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # テスト用の入力
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        with torch.no_grad():
            # SO8T MLP
            x_so8t = so8t_mlp(x)
            
            # 標準MLP
            x_standard = standard_mlp(x)
            
            # 出力の形状確認
            assert x_so8t.shape == x.shape
            assert x_standard.shape == x.shape
            
            # 数値的安定性の確認
            assert torch.isfinite(x_so8t).all()
            assert torch.isfinite(x_standard).all()
            
        print("[OK] MLP比較テスト完了")
    
    def test_attention_comparison(self):
        """アテンション比較テスト"""
        print("\n[TEST] アテンション比較テスト...")
        
        hidden_size = 64
        batch_size, seq_len = 4, 16
        
        # SO8Tアテンション
        so8t_attention = SO8TAttention(
            hidden_size=hidden_size,
            num_attention_heads=8,
            attention_dropout=0.1
        )
        
        # 標準PyTorchアテンション
        standard_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # テスト用の入力
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        with torch.no_grad():
            # SO8Tアテンション
            x_so8t, _ = so8t_attention(x)
            
            # 標準アテンション
            x_standard, _ = standard_attention(x, x, x)
            
            # 出力の形状確認
            assert x_so8t.shape == x.shape
            assert x_standard.shape == x.shape
            
            # 数値的安定性の確認
            assert torch.isfinite(x_so8t).all()
            assert torch.isfinite(x_standard).all()
            
        print("[OK] アテンション比較テスト完了")


class TestGradientComparison:
    """勾配計算比較テスト"""
    
    def test_gradient_accuracy(self):
        """勾配計算の精度比較"""
        print("\n[TEST] 勾配計算精度比較...")
        
        hidden_size = 64
        batch_size, seq_len = 4, 16
        
        # SO8T回転
        so8t_rotation = SO8Rotation(hidden_size=hidden_size, rotation_dim=8)
        
        # 標準線形層
        standard_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # テスト用の入力
        x = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
        
        # SO8T回転の勾配計算
        R = so8t_rotation._generate_rotation_matrix()
        y_so8t = torch.matmul(x, R.T)
        loss_so8t = torch.mean(y_so8t ** 2)
        loss_so8t.backward()
        grad_so8t = x.grad.clone()
        
        # 勾配をリセット
        x.grad.zero_()
        
        # 標準線形層の勾配計算
        y_standard = standard_linear(x)
        loss_standard = torch.mean(y_standard ** 2)
        loss_standard.backward()
        grad_standard = x.grad.clone()
        
        # 勾配の数値的安定性確認
        assert torch.isfinite(grad_so8t).all()
        assert torch.isfinite(grad_standard).all()
        
        # 勾配のノルム比較
        grad_norm_so8t = torch.norm(grad_so8t)
        grad_norm_standard = torch.norm(grad_standard)
        
        print(f"[INFO] SO8T勾配ノルム: {grad_norm_so8t:.6f}")
        print(f"[INFO] 標準勾配ノルム: {grad_norm_standard:.6f}")
        
        print("[OK] 勾配計算精度比較完了")
    
    def test_gradient_flow_analysis(self):
        """勾配フロー分析"""
        print("\n[TEST] 勾配フロー分析...")
        
        hidden_size = 64
        batch_size, seq_len = 4, 16
        
        # SO8Tモデル
        so8t_rotation = SO8Rotation(hidden_size=hidden_size, rotation_dim=8)
        so8t_gate = NonCommutativeGate(hidden_size=hidden_size)
        
        # テスト用の入力
        x = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
        
        # フォワードパス
        R = so8t_rotation._generate_rotation_matrix()
        x_rotated = torch.matmul(x, R.T)
        cmd_output, safe_output, _ = so8t_gate(x_rotated)
        
        # 損失計算
        loss = torch.mean(cmd_output ** 2) + torch.mean(safe_output ** 2)
        
        # バックプロパゲーション
        loss.backward()
        
        # 勾配の分析
        grad_norm = torch.norm(x.grad)
        grad_max = torch.max(torch.abs(x.grad))
        grad_mean = torch.mean(torch.abs(x.grad))
        
        print(f"[INFO] 勾配ノルム: {grad_norm:.6f}")
        print(f"[INFO] 最大勾配: {grad_max:.6f}")
        print(f"[INFO] 平均勾配: {grad_mean:.6f}")
        
        # 勾配の数値的安定性確認
        assert torch.isfinite(x.grad).all()
        assert not torch.isnan(x.grad).any()
        
        # 勾配消失/爆発のチェック
        assert grad_norm > 1e-8, "勾配消失の可能性"
        assert grad_norm < 1e3, "勾配爆発の可能性"
        
        print("[OK] 勾配フロー分析完了")
    
    def test_second_order_derivatives(self):
        """2階微分のテスト"""
        print("\n[TEST] 2階微分テスト...")
        
        hidden_size = 64
        batch_size, seq_len = 4, 16
        
        # SO8T回転
        so8t_rotation = SO8Rotation(hidden_size=hidden_size, rotation_dim=8)
        
        # テスト用の入力
        x = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
        
        # 1階微分
        R = so8t_rotation._generate_rotation_matrix()
        y = torch.matmul(x, R.T)
        loss = torch.mean(y ** 2)
        
        # 1階勾配計算
        grad1 = torch.autograd.grad(loss, x, create_graph=True)[0]
        
        # 2階微分（ヘッセ行列の対角要素）
        grad2_diag = torch.autograd.grad(
            grad1.sum(), x, retain_graph=True, create_graph=True
        )[0]
        
        # 2階微分の数値的安定性確認
        assert torch.isfinite(grad2_diag).all()
        assert not torch.isnan(grad2_diag).any()
        
        # 2階微分のノルム
        grad2_norm = torch.norm(grad2_diag)
        print(f"[INFO] 2階微分ノルム: {grad2_norm:.6f}")
        
        print("[OK] 2階微分テスト完了")


class TestLossFunctionComparison:
    """損失関数比較テスト"""
    
    def test_pet_loss_vs_standard_regularization(self):
        """PET損失と標準正則化の比較"""
        print("\n[TEST] PET損失と標準正則化比較...")
        
        # PET損失
        pet_loss = PETLoss(max_lambda=0.1, warmup_steps=100, main_steps=1000)
        
        # 標準L2正則化
        l2_regularization = nn.MSELoss()
        
        # テスト用の系列
        batch_size, seq_len, hidden_size = 4, 20, 64
        sequence = torch.randn(batch_size, seq_len, hidden_size)
        
        # 滑らかな系列
        smooth_sequence = torch.linspace(0, 1, seq_len).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, hidden_size)
        
        with torch.no_grad():
            # PET損失
            pet_loss_val = pet_loss(sequence, step=500)
            
            # 標準L2正則化
            l2_loss_val = l2_regularization(sequence, torch.zeros_like(sequence))
            
            # 滑らかな系列でのPET損失
            pet_smooth = pet_loss(smooth_sequence, step=500)
            
            # 滑らかな系列でのL2正則化
            l2_smooth = l2_regularization(smooth_sequence, torch.zeros_like(smooth_sequence))
            
            print(f"[INFO] PET損失 (ノイジー): {pet_loss_val:.6f}")
            print(f"[INFO] L2正則化 (ノイジー): {l2_loss_val:.6f}")
            print(f"[INFO] PET損失 (滑らか): {pet_smooth:.6f}")
            print(f"[INFO] L2正則化 (滑らか): {l2_smooth:.6f}")
            
            # PET損失は滑らかさを重視するため、滑らかな系列でより低い損失
            assert pet_smooth < pet_loss_val, "PET損失が滑らかさを正しく評価していません"
            
        print("[OK] PET損失と標準正則化比較完了")
    
    def test_loss_consistency_across_batches(self):
        """バッチ間での損失一貫性テスト"""
        print("\n[TEST] バッチ間損失一貫性テスト...")
        
        pet_loss = PETLoss(max_lambda=0.1)
        
        # 異なるバッチサイズでのテスト
        batch_sizes = [1, 2, 4, 8]
        seq_len, hidden_size = 16, 64
        
        losses = []
        
        for batch_size in batch_sizes:
            sequence = torch.randn(batch_size, seq_len, hidden_size)
            
            with torch.no_grad():
                loss = pet_loss(sequence, step=500)
                losses.append(loss.item())
                
                # 損失が非負であることを確認
                assert loss >= 0, f"バッチサイズ{batch_size}で負の損失: {loss}"
                
        print(f"[INFO] 各バッチサイズでの損失: {losses}")
        
        # 損失の一貫性確認（完全に一致する必要はないが、極端に異なってはいけない）
        loss_std = np.std(losses)
        assert loss_std < 1.0, f"バッチ間で損失のばらつきが大きすぎます: {loss_std}"
        
        print("[OK] バッチ間損失一貫性テスト完了")


class TestNumericalStability:
    """数値安定性テスト"""
    
    def test_extreme_values_handling(self):
        """極値処理のテスト"""
        print("\n[TEST] 極値処理テスト...")
        
        hidden_size = 64
        batch_size, seq_len = 4, 16
        
        # SO8T回転
        so8t_rotation = SO8Rotation(hidden_size=hidden_size, rotation_dim=8)
        
        # 極値でのテスト
        extreme_values = [
            torch.zeros(batch_size, seq_len, hidden_size),
            torch.ones(batch_size, seq_len, hidden_size) * 1e6,
            torch.ones(batch_size, seq_len, hidden_size) * -1e6,
            torch.randn(batch_size, seq_len, hidden_size) * 1e3
        ]
        
        for i, x in enumerate(extreme_values):
            with torch.no_grad():
                R = so8t_rotation._generate_rotation_matrix()
                y = torch.matmul(x, R.T)
                
                # 数値的安定性確認
                assert torch.isfinite(y).all(), f"極値{i}で無限値が発生"
                assert not torch.isnan(y).any(), f"極値{i}でNaNが発生"
                
        print("[OK] 極値処理テスト完了")
    
    def test_precision_comparison(self):
        """精度比較テスト"""
        print("\n[TEST] 精度比較テスト...")
        
        hidden_size = 64
        batch_size, seq_len = 4, 16
        
        # 異なる精度でのテスト
        precisions = [torch.float32, torch.float64]
        
        for precision in precisions:
            # SO8T回転
            so8t_rotation = SO8Rotation(hidden_size=hidden_size, rotation_dim=8)
            
            # テスト用の入力
            x = torch.randn(batch_size, seq_len, hidden_size, dtype=precision)
            
            with torch.no_grad():
                R = so8t_rotation._generate_rotation_matrix().to(precision)
                y = torch.matmul(x, R.T)
                
                # 数値的安定性確認
                assert torch.isfinite(y).all()
                assert not torch.isnan(y).any()
                
                # ノルム保持の確認
                norm_before = torch.norm(x, dim=-1)
                norm_after = torch.norm(y, dim=-1)
                norm_error = torch.mean(torch.abs(norm_before - norm_after))
                
                print(f"[INFO] {precision}でのノルム誤差: {norm_error:.2e}")
                
        print("[OK] 精度比較テスト完了")


class TestPerformanceComparison:
    """パフォーマンス比較テスト"""
    
    def test_computation_speed(self):
        """計算速度比較"""
        print("\n[TEST] 計算速度比較...")
        
        hidden_size = 64
        batch_size, seq_len = 16, 32
        
        # SO8T回転
        so8t_rotation = SO8Rotation(hidden_size=hidden_size, rotation_dim=8)
        
        # 標準線形層
        standard_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # テスト用の入力
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        # ウォームアップ
        for _ in range(10):
            with torch.no_grad():
                R = so8t_rotation._generate_rotation_matrix()
                _ = torch.matmul(x, R.T)
                _ = standard_linear(x)
        
        # SO8T回転の速度測定
        num_iterations = 100
        start_time = time.time()
        
        for _ in tqdm(range(num_iterations), desc="SO8T回転速度テスト"):
            with torch.no_grad():
                R = so8t_rotation._generate_rotation_matrix()
                _ = torch.matmul(x, R.T)
        
        so8t_time = time.time() - start_time
        
        # 標準線形層の速度測定
        start_time = time.time()
        
        for _ in tqdm(range(num_iterations), desc="標準線形層速度テスト"):
            with torch.no_grad():
                _ = standard_linear(x)
        
        standard_time = time.time() - start_time
        
        print(f"[INFO] SO8T回転時間: {so8t_time:.4f}秒")
        print(f"[INFO] 標準線形層時間: {standard_time:.4f}秒")
        print(f"[INFO] 速度比: {so8t_time/standard_time:.2f}x")
        
        # 速度が許容範囲内であることを確認
        assert so8t_time < 1.0, f"SO8T回転が遅すぎます: {so8t_time}秒"
        assert standard_time < 1.0, f"標準線形層が遅すぎます: {standard_time}秒"
        
        print("[OK] 計算速度比較完了")
    
    def test_memory_efficiency(self):
        """メモリ効率比較"""
        print("\n[TEST] メモリ効率比較...")
        
        hidden_size = 256
        batch_size, seq_len = 8, 64
        
        # SO8T回転
        so8t_rotation = SO8Rotation(hidden_size=hidden_size, rotation_dim=8)
        
        # 標準線形層
        standard_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # テスト用の入力
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        with torch.no_grad():
            # SO8T回転
            R = so8t_rotation._generate_rotation_matrix()
            y_so8t = torch.matmul(x, R.T)
            
            # 標準線形層
            y_standard = standard_linear(x)
            
            # メモリ使用量の推定
            so8t_memory = x.numel() * x.element_size() + R.numel() * R.element_size() + y_so8t.numel() * y_so8t.element_size()
            standard_memory = x.numel() * x.element_size() + standard_linear.weight.numel() * standard_linear.weight.element_size() + y_standard.numel() * y_standard.element_size()
            
            print(f"[INFO] SO8T回転メモリ使用量: {so8t_memory / 1024:.2f} KB")
            print(f"[INFO] 標準線形層メモリ使用量: {standard_memory / 1024:.2f} KB")
            
        print("[OK] メモリ効率比較完了")


def run_pytorch_comparison_tests():
    """PyTorch比較テストの実行"""
    print("=" * 80)
    print("PyTorchモデルとの精度比較テスト開始")
    print("=" * 80)
    
    # テストクラスのインスタンス化
    test_classes = [
        TestForwardPassComparison(),
        TestGradientComparison(),
        TestLossFunctionComparison(),
        TestNumericalStability(),
        TestPerformanceComparison()
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for test_class in test_classes:
        print(f"\n[CLASS] {test_class.__class__.__name__}")
        print("-" * 60)
        
        # テストメソッドの取得
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            try:
                print(f"\n[TEST] {test_method}")
                getattr(test_class, test_method)()
                passed_tests += 1
                print(f"[OK] {test_method} 成功")
            except Exception as e:
                failed_tests += 1
                print(f"[NG] {test_method} 失敗: {str(e)}")
    
    # 結果サマリー
    print("\n" + "=" * 80)
    print("PyTorch比較テスト結果サマリー")
    print("=" * 80)
    print(f"総テスト数: {total_tests}")
    print(f"成功: {passed_tests}")
    print(f"失敗: {failed_tests}")
    print(f"成功率: {passed_tests/total_tests*100:.1f}%")
    
    if failed_tests == 0:
        print("\n[SUCCESS] 全てのPyTorch比較テストが成功しました！")
    else:
        print(f"\n[WARNING] {failed_tests}個のPyTorch比較テストが失敗しました")
    
    return failed_tests == 0


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # PyTorch比較テストの実行
    success = run_pytorch_comparison_tests()
    
    # 終了コード
    exit(0 if success else 1)
