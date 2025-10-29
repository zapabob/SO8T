"""
SO8Tモデル量子化テスト

SO8Tモデルの量子化機能をテスト:
- 8bit/4bit量子化の精度検証
- GGUF変換のテスト
- llama.cpp統合のテスト
- 量子化後のSO(8)群性質保持
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
import tempfile
import os
from pathlib import Path
import time
from tqdm import tqdm

# SO8T関連のインポート
from models.so8t_group_structure import SO8Rotation, NonCommutativeGate, PETRegularization, SO8TGroupStructure
from so8t-mmllm.src.modules.rotation_gate import SO8TRotationGate
from models.so8t_mlp import SO8TMLP
from models.so8t_attention import SO8TAttention

# 量子化関連のインポート
from utils.so8t_quantization import (
    SO8TQuantizer, 
    SO8TGGUFConverter, 
    SO8TQuantizationValidator,
    quantize_so8t_model
)

logger = logging.getLogger(__name__)


class TestSO8TQuantization:
    """SO8T量子化テスト"""
    
    def test_8bit_quantization_accuracy(self):
        """8bit量子化の精度テスト"""
        print("\n[TEST] 8bit量子化精度検証...")
        
        # テスト用のSO8Tモデル
        model = SO8TGroupStructure(hidden_size=64, rotation_dim=8)
        
        # 量子化器の作成
        quantizer = SO8TQuantizer(model, quantization_type="8bit")
        
        # キャリブレーションデータの生成
        calibration_data = [torch.randn(4, 16, 64) for _ in range(50)]
        
        # キャリブレーション
        quantizer.calibrate(calibration_data)
        
        # 量子化実行
        quantized_model = quantizer.quantize_model()
        
        # 精度検証
        validator = SO8TQuantizationValidator(model, quantized_model)
        results = validator.validate_quantization(calibration_data)
        
        # 精度チェック
        assert results['mse_error'] < 0.1, f"MSE誤差が大きすぎます: {results['mse_error']}"
        assert results['mae_error'] < 0.05, f"MAE誤差が大きすぎます: {results['mae_error']}"
        assert results['cosine_similarity'] > 0.95, f"コサイン類似度が低すぎます: {results['cosine_similarity']}"
        
        print("[OK] 8bit量子化精度検証完了")
    
    def test_4bit_quantization_accuracy(self):
        """4bit量子化の精度テスト"""
        print("\n[TEST] 4bit量子化精度検証...")
        
        # テスト用のSO8Tモデル
        model = SO8TGroupStructure(hidden_size=64, rotation_dim=8)
        
        # 量子化器の作成
        quantizer = SO8TQuantizer(model, quantization_type="4bit")
        
        # キャリブレーションデータの生成
        calibration_data = [torch.randn(4, 16, 64) for _ in range(50)]
        
        # キャリブレーション
        quantizer.calibrate(calibration_data)
        
        # 量子化実行
        quantized_model = quantizer.quantize_model()
        
        # 精度検証
        validator = SO8TQuantizationValidator(model, quantized_model)
        results = validator.validate_quantization(calibration_data)
        
        # 4bit量子化は精度が低くなることを許容
        assert results['mse_error'] < 0.5, f"MSE誤差が大きすぎます: {results['mse_error']}"
        assert results['mae_error'] < 0.2, f"MAE誤差が大きすぎます: {results['mae_error']}"
        assert results['cosine_similarity'] > 0.8, f"コサイン類似度が低すぎます: {results['cosine_similarity']}"
        
        print("[OK] 4bit量子化精度検証完了")
    
    def test_fp16_quantization_accuracy(self):
        """FP16量子化の精度テスト"""
        print("\n[TEST] FP16量子化精度検証...")
        
        # テスト用のSO8Tモデル
        model = SO8TGroupStructure(hidden_size=64, rotation_dim=8)
        
        # 量子化器の作成
        quantizer = SO8TQuantizer(model, quantization_type="fp16")
        
        # キャリブレーションデータの生成
        calibration_data = [torch.randn(4, 16, 64) for _ in range(50)]
        
        # キャリブレーション
        quantizer.calibrate(calibration_data)
        
        # 量子化実行
        quantized_model = quantizer.quantize_model()
        
        # 精度検証
        validator = SO8TQuantizationValidator(model, quantized_model)
        results = validator.validate_quantization(calibration_data)
        
        # FP16量子化は高精度であることを期待
        assert results['mse_error'] < 1e-6, f"MSE誤差が大きすぎます: {results['mse_error']}"
        assert results['mae_error'] < 1e-6, f"MAE誤差が大きすぎます: {results['mae_error']}"
        assert results['cosine_similarity'] > 0.999, f"コサイン類似度が低すぎます: {results['cosine_similarity']}"
        
        print("[OK] FP16量子化精度検証完了")
    
    def test_quantization_so8_properties(self):
        """量子化後のSO(8)群性質保持テスト"""
        print("\n[TEST] 量子化後SO(8)群性質保持検証...")
        
        # テスト用のSO8Tモデル
        model = SO8TGroupStructure(hidden_size=64, rotation_dim=8)
        
        # 8bit量子化
        quantizer = SO8TQuantizer(model, quantization_type="8bit")
        calibration_data = [torch.randn(4, 16, 64) for _ in range(50)]
        quantizer.calibrate(calibration_data)
        quantized_model = quantizer.quantize_model()
        
        # SO(8)群性質の検証
        validator = SO8TQuantizationValidator(model, quantized_model)
        so8_results = validator.validate_so8_properties(calibration_data)
        
        # SO(8)群性質の保持をチェック
        assert so8_results['orthogonality_error'] < 1e-3, f"直交性誤差が大きすぎます: {so8_results['orthogonality_error']}"
        assert so8_results['determinant_error'] < 1e-3, f"行列式誤差が大きすぎます: {so8_results['determinant_error']}"
        assert so8_results['norm_preservation_error'] < 1e-3, f"ノルム保持誤差が大きすぎます: {so8_results['norm_preservation_error']}"
        
        print("[OK] 量子化後SO(8)群性質保持検証完了")
    
    def test_quantization_memory_efficiency(self):
        """量子化によるメモリ効率テスト"""
        print("\n[TEST] 量子化メモリ効率検証...")
        
        # テスト用のSO8Tモデル
        model = SO8TGroupStructure(hidden_size=256, rotation_dim=8)
        
        # 元のモデルのメモリ使用量
        original_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        
        # 8bit量子化
        quantizer = SO8TQuantizer(model, quantization_type="8bit")
        calibration_data = [torch.randn(4, 16, 256) for _ in range(50)]
        quantizer.calibrate(calibration_data)
        quantized_model = quantizer.quantize_model()
        
        # 量子化後のメモリ使用量（推定）
        quantized_memory = sum(p.numel() * 1 for p in quantized_model.parameters())  # 8bit = 1byte
        
        # メモリ削減率の計算
        memory_reduction = (original_memory - quantized_memory) / original_memory
        
        print(f"[INFO] 元のメモリ使用量: {original_memory / 1024:.2f} KB")
        print(f"[INFO] 量子化後メモリ使用量: {quantized_memory / 1024:.2f} KB")
        print(f"[INFO] メモリ削減率: {memory_reduction:.2%}")
        
        # メモリ削減が期待される範囲内であることをチェック
        assert memory_reduction > 0.5, f"メモリ削減が不十分です: {memory_reduction:.2%}"
        
        print("[OK] 量子化メモリ効率検証完了")


class TestGGUFConversion:
    """GGUF変換テスト"""
    
    def test_gguf_conversion_basic(self):
        """基本的なGGUF変換テスト"""
        print("\n[TEST] 基本的なGGUF変換検証...")
        
        # テスト用のSO8Tモデル
        model = SO8TGroupStructure(hidden_size=64, rotation_dim=8)
        
        # 一時ファイルでのGGUF変換テスト
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as tmp_file:
            output_path = tmp_file.name
        
        try:
            # GGUF変換
            converter = SO8TGGUFConverter(model, "so8t-test")
            converter.convert_to_gguf(output_path, quantization_type="fp16")
            
            # ファイルが作成されたことを確認
            assert os.path.exists(output_path), "GGUFファイルが作成されていません"
            
            # ファイルサイズの確認
            file_size = os.path.getsize(output_path)
            assert file_size > 0, "GGUFファイルが空です"
            
            print(f"[INFO] GGUFファイルサイズ: {file_size / 1024:.2f} KB")
            
        finally:
            # 一時ファイルの削除
            if os.path.exists(output_path):
                os.unlink(output_path)
        
        print("[OK] 基本的なGGUF変換検証完了")
    
    def test_gguf_conversion_with_quantization(self):
        """量子化付きGGUF変換テスト"""
        print("\n[TEST] 量子化付きGGUF変換検証...")
        
        # テスト用のSO8Tモデル
        model = SO8TGroupStructure(hidden_size=64, rotation_dim=8)
        
        # 8bit量子化
        quantizer = SO8TQuantizer(model, quantization_type="8bit")
        calibration_data = [torch.randn(4, 16, 64) for _ in range(50)]
        quantizer.calibrate(calibration_data)
        quantized_model = quantizer.quantize_model()
        
        # 一時ファイルでのGGUF変換テスト
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as tmp_file:
            output_path = tmp_file.name
        
        try:
            # 量子化付きGGUF変換
            converter = SO8TGGUFConverter(quantized_model, "so8t-quantized")
            converter.convert_to_gguf(output_path, quantization_type="8bit")
            
            # ファイルが作成されたことを確認
            assert os.path.exists(output_path), "GGUFファイルが作成されていません"
            
            # ファイルサイズの確認
            file_size = os.path.getsize(output_path)
            assert file_size > 0, "GGUFファイルが空です"
            
            print(f"[INFO] 量子化GGUFファイルサイズ: {file_size / 1024:.2f} KB")
            
        finally:
            # 一時ファイルの削除
            if os.path.exists(output_path):
                os.unlink(output_path)
        
        print("[OK] 量子化付きGGUF変換検証完了")
    
    def test_gguf_metadata(self):
        """GGUFメタデータテスト"""
        print("\n[TEST] GGUFメタデータ検証...")
        
        # テスト用のSO8Tモデル
        model = SO8TGroupStructure(hidden_size=64, rotation_dim=8)
        
        # カスタムメタデータ
        custom_metadata = {
            "general.name": "so8t-custom-test",
            "general.description": "Custom SO8T Test Model",
            "general.author": "Test Author",
            "general.license": "MIT",
            "so8t.rotation_dim": "8",
            "so8t.group_structure": "SO(8)",
            "so8t.safety_features": "enabled"
        }
        
        # 一時ファイルでのGGUF変換テスト
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as tmp_file:
            output_path = tmp_file.name
        
        try:
            # カスタムメタデータ付きGGUF変換
            converter = SO8TGGUFConverter(model, "so8t-custom")
            converter.convert_to_gguf(output_path, quantization_type="fp16", metadata=custom_metadata)
            
            # ファイルが作成されたことを確認
            assert os.path.exists(output_path), "GGUFファイルが作成されていません"
            
            # メタデータの検証（簡易版）
            with open(output_path, 'rb') as f:
                # GGUFファイルのヘッダーを読み取り
                header = f.read(1024)  # 最初の1KBを読み取り
                
                # メタデータが含まれていることを確認
                assert b"so8t-custom-test" in header, "カスタムメタデータが含まれていません"
                assert b"SO8T" in header, "SO8Tメタデータが含まれていません"
            
        finally:
            # 一時ファイルの削除
            if os.path.exists(output_path):
                os.unlink(output_path)
        
        print("[OK] GGUFメタデータ検証完了")


class TestLlamaCppIntegration:
    """llama.cpp統合テスト"""
    
    def test_quantized_model_compatibility(self):
        """量子化モデルのllama.cpp互換性テスト"""
        print("\n[TEST] llama.cpp互換性検証...")
        
        # テスト用のSO8Tモデル
        model = SO8TGroupStructure(hidden_size=64, rotation_dim=8)
        
        # 8bit量子化
        quantizer = SO8TQuantizer(model, quantization_type="8bit")
        calibration_data = [torch.randn(4, 16, 64) for _ in range(50)]
        quantizer.calibrate(calibration_data)
        quantized_model = quantizer.quantize_model()
        
        # GGUF変換
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as tmp_file:
            output_path = tmp_file.name
        
        try:
            converter = SO8TGGUFConverter(quantized_model, "so8t-llamacpp")
            converter.convert_to_gguf(output_path, quantization_type="8bit")
            
            # GGUFファイルの基本検証
            assert os.path.exists(output_path), "GGUFファイルが作成されていません"
            
            # ファイルサイズの確認
            file_size = os.path.getsize(output_path)
            assert file_size > 0, "GGUFファイルが空です"
            
            # llama.cppで読み込める形式であることを確認（簡易版）
            with open(output_path, 'rb') as f:
                # GGUFマジックナンバーの確認
                magic = f.read(4)
                assert magic == b'GGUF', f"GGUFマジックナンバーが正しくありません: {magic}"
            
            print(f"[INFO] llama.cpp互換GGUFファイルサイズ: {file_size / 1024:.2f} KB")
            
        finally:
            # 一時ファイルの削除
            if os.path.exists(output_path):
                os.unlink(output_path)
        
        print("[OK] llama.cpp互換性検証完了")
    
    def test_model_parameter_export(self):
        """モデルパラメータのエクスポートテスト"""
        print("\n[TEST] モデルパラメータエクスポート検証...")
        
        # テスト用のSO8Tモデル
        model = SO8TGroupStructure(hidden_size=64, rotation_dim=8)
        
        # パラメータの統計情報
        param_count = sum(p.numel() for p in model.parameters())
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        
        print(f"[INFO] パラメータ数: {param_count:,}")
        print(f"[INFO] パラメータサイズ: {param_size / 1024:.2f} KB")
        
        # 各層のパラメータ情報
        layer_info = {}
        for name, param in model.named_parameters():
            layer_info[name] = {
                'shape': list(param.shape),
                'numel': param.numel(),
                'dtype': str(param.dtype),
                'device': str(param.device)
            }
        
        # パラメータ情報の検証
        assert len(layer_info) > 0, "パラメータが見つかりません"
        
        # SO8T特有のパラメータの確認
        so8_params = [name for name in layer_info.keys() if 'rotation' in name.lower()]
        assert len(so8_params) > 0, "SO8T回転パラメータが見つかりません"
        
        print(f"[INFO] SO8T回転パラメータ: {len(so8_params)}個")
        for param_name in so8_params:
            print(f"  - {param_name}: {layer_info[param_name]['shape']}")
        
        print("[OK] モデルパラメータエクスポート検証完了")


class TestQuantizationPerformance:
    """量子化パフォーマンステスト"""
    
    def test_quantization_speed(self):
        """量子化速度テスト"""
        print("\n[TEST] 量子化速度検証...")
        
        # テスト用のSO8Tモデル
        model = SO8TGroupStructure(hidden_size=128, rotation_dim=8)
        
        # キャリブレーションデータの生成
        calibration_data = [torch.randn(8, 32, 128) for _ in range(100)]
        
        # 各量子化タイプでの速度テスト
        quantization_types = ["8bit", "4bit", "fp16"]
        
        for qtype in quantization_types:
            print(f"[INFO] {qtype}量子化速度テスト...")
            
            start_time = time.time()
            
            # 量子化器の作成
            quantizer = SO8TQuantizer(model, quantization_type=qtype)
            
            # キャリブレーション
            quantizer.calibrate(calibration_data)
            
            # 量子化実行
            quantized_model = quantizer.quantize_model()
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"[INFO] {qtype}量子化時間: {duration:.4f}秒")
            
            # 速度が許容範囲内であることを確認
            assert duration < 10.0, f"{qtype}量子化が遅すぎます: {duration:.4f}秒"
        
        print("[OK] 量子化速度検証完了")
    
    def test_quantization_memory_usage(self):
        """量子化メモリ使用量テスト"""
        print("\n[TEST] 量子化メモリ使用量検証...")
        
        # 異なるサイズのモデルでのテスト
        model_sizes = [64, 128, 256]
        
        for hidden_size in model_sizes:
            print(f"[INFO] モデルサイズ {hidden_size} でのメモリ使用量テスト...")
            
            # テスト用のSO8Tモデル
            model = SO8TGroupStructure(hidden_size=hidden_size, rotation_dim=8)
            
            # 元のメモリ使用量
            original_memory = sum(p.numel() * p.element_size() for p in model.parameters())
            
            # 8bit量子化
            quantizer = SO8TQuantizer(model, quantization_type="8bit")
            calibration_data = [torch.randn(4, 16, hidden_size) for _ in range(50)]
            quantizer.calibrate(calibration_data)
            quantized_model = quantizer.quantize_model()
            
            # 量子化後のメモリ使用量（推定）
            quantized_memory = sum(p.numel() * 1 for p in quantized_model.parameters())
            
            # メモリ削減率
            memory_reduction = (original_memory - quantized_memory) / original_memory
            
            print(f"[INFO] 元のメモリ: {original_memory / 1024:.2f} KB")
            print(f"[INFO] 量子化後メモリ: {quantized_memory / 1024:.2f} KB")
            print(f"[INFO] 削減率: {memory_reduction:.2%}")
            
            # メモリ削減が期待される範囲内であることを確認
            assert memory_reduction > 0.4, f"メモリ削減が不十分です: {memory_reduction:.2%}"
        
        print("[OK] 量子化メモリ使用量検証完了")


def run_quantization_tests():
    """量子化テストの実行"""
    print("=" * 80)
    print("SO8Tモデル量子化テスト開始")
    print("=" * 80)
    
    # テストクラスのインスタンス化
    test_classes = [
        TestSO8TQuantization(),
        TestGGUFConversion(),
        TestLlamaCppIntegration(),
        TestQuantizationPerformance()
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
    print("量子化テスト結果サマリー")
    print("=" * 80)
    print(f"総テスト数: {total_tests}")
    print(f"成功: {passed_tests}")
    print(f"失敗: {failed_tests}")
    print(f"成功率: {passed_tests/total_tests*100:.1f}%")
    
    if failed_tests == 0:
        print("\n[SUCCESS] 全ての量子化テストが成功しました！")
    else:
        print(f"\n[WARNING] {failed_tests}個の量子化テストが失敗しました")
    
    return failed_tests == 0


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 量子化テストの実行
    success = run_quantization_tests()
    
    # 終了コード
    exit(0 if success else 1)
