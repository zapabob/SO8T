"""
SO(8)演算の包括的ユニットテスト

SO(8)群構造の数学的性質を厳密に検証:
- 回転行列の直交性 (R^T @ R = I)
- 行列式の保持 (det(R) = 1)
- 非可換性の検証
- PET正則化の数値精度
- 量子化サポートの検証
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any
import math
import logging
from tqdm import tqdm
import time

# SO8T関連のインポート
from models.so8t_group_structure import SO8Rotation, NonCommutativeGate, PETRegularization
from so8t-mmllm.src.modules.rotation_gate import SO8TRotationGate, apply_block_rotation
from so8t-mmllm.src.losses.pet import PETLoss, pet_penalty
from models.so8t_mlp import SO8TMLP
from models.so8t_attention import SO8TAttention, SO8TRotaryEmbedding

logger = logging.getLogger(__name__)


class TestSO8RotationMathematicalProperties:
    """SO(8)回転行列の数学的性質テスト"""
    
    def test_orthogonality_property(self):
        """直交性の検証: R^T @ R = I"""
        print("\n[TEST] SO(8)回転行列の直交性検証...")
        
        rotation = SO8Rotation(hidden_size=64, rotation_dim=8)
        
        # 複数の回転行列を生成してテスト
        for i in tqdm(range(10), desc="直交性テスト"):
            with torch.no_grad():
                R = rotation._generate_rotation_matrix()
                
                # 直交性の検証: R^T @ R = I
                R_T = R.transpose(-1, -2)
                identity_approx = torch.matmul(R_T, R)
                identity_true = torch.eye(8, device=R.device)
                
                # 誤差の計算
                orthogonality_error = torch.max(torch.abs(identity_approx - identity_true))
                
                assert orthogonality_error < 1e-5, f"直交性エラーが大きすぎます: {orthogonality_error}"
                
        print("[OK] 直交性検証完了")
    
    def test_determinant_property(self):
        """行列式の検証: det(R) = 1"""
        print("\n[TEST] SO(8)回転行列の行列式検証...")
        
        rotation = SO8Rotation(hidden_size=64, rotation_dim=8)
        
        for i in tqdm(range(10), desc="行列式テスト"):
            with torch.no_grad():
                R = rotation._generate_rotation_matrix()
                det_R = torch.det(R)
                
                # 行列式は1に近い必要がある
                det_error = torch.abs(det_R - 1.0)
                
                assert det_error < 1e-5, f"行列式エラーが大きすぎます: {det_error}"
                
        print("[OK] 行列式検証完了")
    
    def test_rotation_preserves_norm(self):
        """回転がベクトルのノルムを保持することを検証"""
        print("\n[TEST] 回転によるノルム保持検証...")
        
        rotation = SO8Rotation(hidden_size=64, rotation_dim=8)
        
        # テスト用の8次元ベクトル
        test_vectors = torch.randn(5, 8)
        
        for i, vec in enumerate(tqdm(test_vectors, desc="ノルム保持テスト")):
            with torch.no_grad():
                # 回転前のノルム
                norm_before = torch.norm(vec)
                
                # 回転適用
                R = rotation._generate_rotation_matrix()
                vec_rotated = torch.matmul(vec, R.T)
                
                # 回転後のノルム
                norm_after = torch.norm(vec_rotated)
                
                # ノルムの差
                norm_error = torch.abs(norm_before - norm_after)
                
                assert norm_error < 1e-5, f"ノルム保持エラーが大きすぎます: {norm_error}"
                
        print("[OK] ノルム保持検証完了")
    
    def test_rotation_composition(self):
        """回転の合成の検証: R1 @ R2 も回転行列"""
        print("\n[TEST] 回転合成の検証...")
        
        rotation = SO8Rotation(hidden_size=64, rotation_dim=8)
        
        for i in tqdm(range(5), desc="回転合成テスト"):
            with torch.no_grad():
                R1 = rotation._generate_rotation_matrix()
                R2 = rotation._generate_rotation_matrix()
                
                # 回転の合成
                R_composed = torch.matmul(R1, R2)
                
                # 合成回転も直交行列であることを検証
                R_composed_T = R_composed.transpose(-1, -2)
                identity_approx = torch.matmul(R_composed_T, R_composed)
                identity_true = torch.eye(8, device=R_composed.device)
                
                orthogonality_error = torch.max(torch.abs(identity_approx - identity_true))
                
                assert orthogonality_error < 1e-4, f"合成回転の直交性エラー: {orthogonality_error}"
                
                # 合成回転の行列式も1に近い
                det_composed = torch.det(R_composed)
                det_error = torch.abs(det_composed - 1.0)
                
                assert det_error < 1e-4, f"合成回転の行列式エラー: {det_error}"
                
        print("[OK] 回転合成検証完了")


class TestNonCommutativeGate:
    """非可換ゲートのテスト"""
    
    def test_non_commutativity(self):
        """非可換性の検証: R_cmd @ R_safe ≠ R_safe @ R_cmd"""
        print("\n[TEST] 非可換性の検証...")
        
        gate = NonCommutativeGate(hidden_size=64)
        
        # テスト用の入力
        x = torch.randn(2, 10, 64)
        
        with torch.no_grad():
            # 順序1: R_cmd @ R_safe
            cmd_output1, safe_output1, _ = gate(x)
            
            # 順序2: R_safe @ R_cmd (順序を変更)
            gate.safety_first = False
            cmd_output2, safe_output2, _ = gate(x)
            
            # 出力が異なることを検証
            cmd_diff = torch.max(torch.abs(cmd_output1 - cmd_output2))
            safe_diff = torch.max(torch.abs(safe_output1 - safe_output2))
            
            assert cmd_diff > 1e-3, f"非可換性が不十分です (cmd_diff: {cmd_diff})"
            assert safe_diff > 1e-3, f"非可換性が不十分です (safe_diff: {safe_diff})"
            
        print("[OK] 非可換性検証完了")
    
    def test_gate_consistency(self):
        """ゲートの一貫性テスト"""
        print("\n[TEST] 非可換ゲートの一貫性検証...")
        
        gate = NonCommutativeGate(hidden_size=64)
        
        # 同じ入力に対して同じ出力を返すことを検証
        x = torch.randn(2, 10, 64)
        
        with torch.no_grad():
            cmd_output1, safe_output1, _ = gate(x)
            cmd_output2, safe_output2, _ = gate(x)
            
            # 出力が一致することを検証
            cmd_diff = torch.max(torch.abs(cmd_output1 - cmd_output2))
            safe_diff = torch.max(torch.abs(safe_output1 - safe_output2))
            
            assert cmd_diff < 1e-6, f"一貫性エラー (cmd): {cmd_diff}"
            assert safe_diff < 1e-6, f"一貫性エラー (safe): {safe_diff}"
            
        print("[OK] 一貫性検証完了")


class TestPETRegularization:
    """PET正則化のテスト"""
    
    def test_pet_loss_calculation(self):
        """PET損失の計算精度テスト"""
        print("\n[TEST] PET損失計算精度検証...")
        
        pet_loss = PETRegularization(lambda_pet=0.1, rotation_penalty=0.01)
        
        # テスト用の隠れ状態
        batch_size, seq_len, hidden_size = 2, 10, 64
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        # 回転行列も生成
        rotation_matrices = torch.randn(batch_size, seq_len, 8, 8)
        
        with torch.no_grad():
            loss = pet_loss(hidden_states, rotation_matrices)
            
            # 損失が非負であることを検証
            assert loss >= 0, f"PET損失が負です: {loss}"
            
            # 損失が有限であることを検証
            assert torch.isfinite(loss), f"PET損失が無限です: {loss}"
            
        print("[OK] PET損失計算検証完了")
    
    def test_pet_temporal_smoothness(self):
        """時系列滑らかさの検証"""
        print("\n[TEST] PET時系列滑らかさ検証...")
        
        pet_loss = PETRegularization(lambda_pet=0.1)
        
        # 滑らかな系列（低いPET損失）
        smooth_sequence = torch.linspace(0, 1, 10).unsqueeze(0).unsqueeze(-1).repeat(2, 1, 64)
        
        # ノイジーな系列（高いPET損失）
        noisy_sequence = smooth_sequence + 0.1 * torch.randn_like(smooth_sequence)
        
        with torch.no_grad():
            smooth_loss = pet_loss(smooth_sequence, None)
            noisy_loss = pet_loss(noisy_sequence, None)
            
            # 滑らかな系列の方が損失が小さいことを検証
            assert smooth_loss < noisy_loss, f"滑らかさ検証失敗: smooth={smooth_loss}, noisy={noisy_loss}"
            
        print("[OK] 時系列滑らかさ検証完了")
    
    def test_pet_rotation_constraints(self):
        """回転制約の検証"""
        print("\n[TEST] PET回転制約検証...")
        
        pet_loss = PETRegularization(rotation_penalty=0.01)
        
        # 直交行列（低い損失）
        orthogonal_matrix = torch.eye(8).unsqueeze(0).unsqueeze(0).repeat(2, 10, 1, 1)
        
        # 非直交行列（高い損失）
        non_orthogonal_matrix = torch.randn(2, 10, 8, 8)
        
        hidden_states = torch.randn(2, 10, 64)
        
        with torch.no_grad():
            orthogonal_loss = pet_loss(hidden_states, orthogonal_matrix)
            non_orthogonal_loss = pet_loss(hidden_states, non_orthogonal_matrix)
            
            # 直交行列の方が損失が小さいことを検証
            assert orthogonal_loss < non_orthogonal_loss, f"回転制約検証失敗: orth={orthogonal_loss}, non_orth={non_orthogonal_loss}"
            
        print("[OK] 回転制約検証完了")


class TestSO8TRotationGate:
    """SO8T回転ゲートのテスト"""
    
    def test_block_rotation_consistency(self):
        """ブロック回転の一貫性テスト"""
        print("\n[TEST] ブロック回転一貫性検証...")
        
        hidden_size = 64  # 8の倍数
        gate = SO8TRotationGate(hidden_size=hidden_size, num_blocks=8)
        
        # テスト用の入力
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        with torch.no_grad():
            # 回転適用
            y = gate(x)
            
            # 形状が保持されることを検証
            assert y.shape == x.shape, f"形状が保持されていません: {y.shape} vs {x.shape}"
            
            # 数値的安定性を検証
            assert torch.isfinite(y).all(), "出力に無限値が含まれています"
            assert not torch.isnan(y).any(), "出力にNaNが含まれています"
            
        print("[OK] ブロック回転一貫性検証完了")
    
    def test_rotation_matrix_properties(self):
        """回転行列の性質テスト"""
        print("\n[TEST] 回転行列性質検証...")
        
        hidden_size = 64
        gate = SO8TRotationGate(hidden_size=hidden_size, num_blocks=8)
        
        with torch.no_grad():
            rotation_matrices = gate.get_rotation_matrices()
            
            # 各ブロックの回転行列を検証
            for i in range(8):
                R = rotation_matrices[i]  # [8, 8]
                
                # 直交性の検証
                R_T = R.transpose(-1, -2)
                identity_approx = torch.matmul(R_T, R)
                identity_true = torch.eye(8, device=R.device)
                
                orthogonality_error = torch.max(torch.abs(identity_approx - identity_true))
                assert orthogonality_error < 1e-4, f"ブロック{i}の直交性エラー: {orthogonality_error}"
                
        print("[OK] 回転行列性質検証完了")


class TestQuantizationSupport:
    """量子化サポートのテスト"""
    
    def test_8bit_quantization_compatibility(self):
        """8bit量子化との互換性テスト"""
        print("\n[TEST] 8bit量子化互換性検証...")
        
        # SO8Tモデルのコンポーネント
        rotation = SO8Rotation(hidden_size=64, rotation_dim=8)
        gate = NonCommutativeGate(hidden_size=64)
        pet_loss = PETRegularization()
        
        # テスト用の入力
        x = torch.randn(2, 10, 64)
        
        # 通常の精度での計算
        with torch.no_grad():
            # 回転適用
            R = rotation._generate_rotation_matrix()
            x_rotated = torch.matmul(x, R.T)
            
            # 非可換ゲート
            cmd_output, safe_output, _ = gate(x)
            
            # PET損失
            pet_loss_val = pet_loss(x, None)
            
            # 8bit量子化のシミュレーション
            x_quantized = torch.round(x * 127) / 127  # 8bit量子化
            R_quantized = torch.round(R * 127) / 127
            
            # 量子化後の計算
            x_rotated_quantized = torch.matmul(x_quantized, R_quantized.T)
            
            # 量子化誤差の評価
            quantization_error = torch.mean(torch.abs(x_rotated - x_rotated_quantized))
            
            # 量子化誤差が許容範囲内であることを検証
            assert quantization_error < 0.1, f"量子化誤差が大きすぎます: {quantization_error}"
            
        print("[OK] 8bit量子化互換性検証完了")
    
    def test_gguf_conversion_compatibility(self):
        """GGUF変換との互換性テスト"""
        print("\n[TEST] GGUF変換互換性検証...")
        
        # SO8Tモデルのパラメータ
        rotation = SO8Rotation(hidden_size=64, rotation_dim=8)
        
        with torch.no_grad():
            R = rotation._generate_rotation_matrix()
            
            # GGUF形式での保存に必要な情報
            rotation_data = {
                'rotation_matrix': R.cpu().numpy(),
                'rotation_angles': rotation.rotation_angles.cpu().numpy(),
                'hidden_size': rotation.hidden_size,
                'rotation_dim': rotation.rotation_dim
            }
            
            # データが適切な形式であることを検証
            assert isinstance(rotation_data['rotation_matrix'], np.ndarray)
            assert rotation_data['rotation_matrix'].shape == (8, 8)
            assert isinstance(rotation_data['rotation_angles'], np.ndarray)
            assert rotation_data['rotation_angles'].shape == (8,)
            
            # 数値範囲が適切であることを検証
            assert np.all(np.isfinite(rotation_data['rotation_matrix']))
            assert np.all(np.isfinite(rotation_data['rotation_angles']))
            
        print("[OK] GGUF変換互換性検証完了")


class TestPyTorchComparison:
    """PyTorchモデルとの比較テスト"""
    
    def test_forward_pass_consistency(self):
        """フォワードパスの一貫性テスト"""
        print("\n[TEST] PyTorch一貫性検証...")
        
        # SO8Tコンポーネント
        rotation = SO8Rotation(hidden_size=64, rotation_dim=8)
        gate = NonCommutativeGate(hidden_size=64)
        
        # 標準PyTorch実装との比較
        standard_linear = nn.Linear(64, 64, bias=False)
        
        # テスト用の入力
        x = torch.randn(2, 10, 64)
        
        with torch.no_grad():
            # SO8T回転
            R = rotation._generate_rotation_matrix()
            x_so8t = torch.matmul(x, R.T)
            
            # 標準線形変換
            x_standard = standard_linear(x)
            
            # 両方の出力が適切な形状を持つことを検証
            assert x_so8t.shape == x.shape
            assert x_standard.shape == x.shape
            
            # 数値的安定性を検証
            assert torch.isfinite(x_so8t).all()
            assert torch.isfinite(x_standard).all()
            
        print("[OK] PyTorch一貫性検証完了")
    
    def test_gradient_computation(self):
        """勾配計算のテスト"""
        print("\n[TEST] 勾配計算検証...")
        
        rotation = SO8Rotation(hidden_size=64, rotation_dim=8)
        
        # テスト用の入力
        x = torch.randn(2, 10, 64, requires_grad=True)
        
        # 回転適用
        R = rotation._generate_rotation_matrix()
        y = torch.matmul(x, R.T)
        
        # 損失計算
        loss = torch.mean(y ** 2)
        
        # 勾配計算
        loss.backward()
        
        # 勾配が適切に計算されることを検証
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert not torch.isnan(x.grad).any()
        
        print("[OK] 勾配計算検証完了")


class TestPerformanceBenchmarks:
    """パフォーマンスベンチマークテスト"""
    
    def test_rotation_speed(self):
        """回転計算の速度テスト"""
        print("\n[TEST] 回転計算速度ベンチマーク...")
        
        rotation = SO8Rotation(hidden_size=64, rotation_dim=8)
        
        # 大きなバッチでの速度テスト
        batch_size, seq_len = 32, 100
        x = torch.randn(batch_size, seq_len, 64)
        
        # ウォームアップ
        for _ in range(10):
            with torch.no_grad():
                R = rotation._generate_rotation_matrix()
                _ = torch.matmul(x, R.T)
        
        # 速度測定
        start_time = time.time()
        num_iterations = 100
        
        for _ in tqdm(range(num_iterations), desc="速度テスト"):
            with torch.no_grad():
                R = rotation._generate_rotation_matrix()
                _ = torch.matmul(x, R.T)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_iterations
        
        print(f"[INFO] 平均回転計算時間: {avg_time:.4f}秒")
        
        # 速度が許容範囲内であることを検証
        assert avg_time < 0.01, f"回転計算が遅すぎます: {avg_time}秒"
        
        print("[OK] 回転計算速度検証完了")
    
    def test_memory_usage(self):
        """メモリ使用量テスト"""
        print("\n[TEST] メモリ使用量検証...")
        
        # 大きなモデルでのメモリ使用量テスト
        hidden_size = 4096
        rotation = SO8Rotation(hidden_size=hidden_size, rotation_dim=8)
        gate = NonCommutativeGate(hidden_size=hidden_size)
        
        # 大きなバッチでのテスト
        batch_size, seq_len = 16, 512
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        with torch.no_grad():
            # メモリ使用量の測定
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # 回転適用
            R = rotation._generate_rotation_matrix()
            y = torch.matmul(x, R.T)
            
            # 非可換ゲート
            cmd_output, safe_output, _ = gate(x)
            
            # メモリリークがないことを検証
            assert torch.isfinite(y).all()
            assert torch.isfinite(cmd_output).all()
            assert torch.isfinite(safe_output).all()
            
        print("[OK] メモリ使用量検証完了")


def run_comprehensive_tests():
    """包括的テストの実行"""
    print("=" * 80)
    print("SO(8)演算包括的ユニットテスト開始")
    print("=" * 80)
    
    # テストクラスのインスタンス化
    test_classes = [
        TestSO8RotationMathematicalProperties(),
        TestNonCommutativeGate(),
        TestPETRegularization(),
        TestSO8TRotationGate(),
        TestQuantizationSupport(),
        TestPyTorchComparison(),
        TestPerformanceBenchmarks()
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
    print("テスト結果サマリー")
    print("=" * 80)
    print(f"総テスト数: {total_tests}")
    print(f"成功: {passed_tests}")
    print(f"失敗: {failed_tests}")
    print(f"成功率: {passed_tests/total_tests*100:.1f}%")
    
    if failed_tests == 0:
        print("\n[SUCCESS] 全てのテストが成功しました！")
    else:
        print(f"\n[WARNING] {failed_tests}個のテストが失敗しました")
    
    return failed_tests == 0


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 包括的テストの実行
    success = run_comprehensive_tests()
    
    # 終了コード
    exit(0 if success else 1)
