#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple RLPO Test for MOONSHOT Quick Verification
SO(8) NKAT Theory RLPO機能の基本テスト
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_so8t_imports():
    """SO8Tモジュールのインポートテスト"""
    print("🔍 SO8Tモジュールインポートテスト...")

    try:
        from so8t_core import (
            SO8TRotationGate,
            PETRegularizer,
            PETSchedule,
            TrialityHead,
            SelfVerifier,
            SO8TModelConfig,
            BurnInManager
        )
        print("✅ SO8Tコアモジュールインポート成功")
        return True
    except ImportError as e:
        print(f"❌ SO8Tインポートエラー: {e}")
        return False

def test_so8t_rotation_gate():
    """SO8TRotationGateの基本テスト"""
    print("🔍 SO8T Rotation Gateテスト...")

    try:
        from so8t_core import SO8TRotationGate

        # 基本的な回転ゲートテスト (hidden_sizeは8の倍数である必要がある)
        gate = SO8TRotationGate(hidden_size=256)
        x = torch.randn(2, 10, 256)  # (batch, seq, hidden_size)

        output = gate(x)
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"

        print("✅ SO8T Rotation Gateテスト成功")
        return True
    except Exception as e:
        print(f"❌ Rotation Gateテスト失敗: {e}")
        return False

def test_pet_regularizer():
    """PET Regularizerの基本テスト"""
    print("🔍 PET Regularizerテスト...")

    try:
        from so8t_core import PETRegularizer, PETSchedule

        # PET Regularizerテスト
        schedule = PETSchedule()
        regularizer = PETRegularizer(schedule=schedule)

        # hidden_statesでテスト (B, T, D)
        hidden_states = torch.randn(2, 10, 256)
        progress = 0.5  # トレーニング進捗 (0-1)

        reg_loss = regularizer(hidden_states, progress)
        assert isinstance(reg_loss, torch.Tensor), "Regularizer should return tensor"

        print("✅ PET Regularizerテスト成功")
        return True
    except Exception as e:
        print(f"❌ PET Regularizerテスト失敗: {e}")
        return False

def test_triality_head():
    """Triality Headの基本テスト"""
    print("🔍 Triality Headテスト...")

    try:
        from so8t_core import TrialityHead

        # Triality Headテスト
        head = TrialityHead(hidden_size=256)
        x = torch.randn(2, 10, 256)  # (batch, seq, hidden_size)

        output = head(x)
        assert hasattr(output, 'logits'), "TrialityOutput should have logits"
        assert hasattr(output, 'probabilities'), "TrialityOutput should have probabilities"
        assert hasattr(output, 'predicted'), "TrialityOutput should have predicted"

        print("✅ Triality Headテスト成功")
        return True
    except Exception as e:
        print(f"❌ Triality Headテスト失敗: {e}")
        return False

def test_self_verifier():
    """Self-Verifierの基本テスト"""
    print("🔍 Self-Verifierテスト...")

    try:
        from so8t_core import SelfVerifier

        # Self-Verifierテスト
        verifier = SelfVerifier()
        reasoning = "This is a test reasoning."
        logits = torch.randn(1, 10)  # (batch, vocab_size)
        compliance_score = 0.8  # コンプライアンススコア

        result = verifier.score_pass(reasoning, logits, compliance_score)
        assert isinstance(result, float), "score_pass should return float"

        print("✅ Self-Verifierテスト成功")
        return True
    except Exception as e:
        print(f"❌ Self-Verifierテスト失敗: {e}")
        return False

def test_torch_cuda():
    """PyTorch CUDA利用可能性テスト"""
    print("🔍 PyTorch CUDAテスト...")

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)

        print(f"✅ CUDA利用可能: {device_count}デバイス")
        print(f"   現在のデバイス: {current_device} ({device_name})")

        # 簡単なCUDAテンソルテスト
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = torch.matmul(x, y)
        assert z.shape == (100, 100), "CUDA matrix multiplication failed"

        print("✅ CUDAテンソル演算成功")
        return True
    else:
        print("⚠️ CUDA利用不可 - CPUモードで実行")
        return True

def main():
    """メイン実行関数"""
    print("🚀 MOONSHOT Simple RLPO Test開始")
    print("=" * 50)

    # システム情報
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
    print()

    # テスト実行
    tests = [
        ("SO8T Imports", test_so8t_imports),
        ("CUDA Support", test_torch_cuda),
        ("SO8T Rotation Gate", test_so8t_rotation_gate),
        ("PET Regularizer", test_pet_regularizer),
        ("Triality Head", test_triality_head),
        ("Self-Verifier", test_self_verifier),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
        print()

    # 結果サマリー
    print("=" * 50)
    print(f"テスト結果: {passed}/{total} 通過")

    if passed == total:
        print("🎉 すべてのテストが成功しました！MOONSHOT Ready!")
        return 0
    else:
        print("⚠️ 一部のテストが失敗しました。システムを確認してください。")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
