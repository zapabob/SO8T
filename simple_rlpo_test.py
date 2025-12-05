#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOONSHOT System Test - AEGIS Autonomous A/B Testing Platform

簡易システムテスト：Python環境、PyTorch、Transformersの基本機能を検証
"""

import sys
import torch
import platform
from pathlib import Path

def test_python_environment():
    """Python環境テスト"""
    print(f"[TEST] Python version: {sys.version}")
    print(f"[TEST] Platform: {platform.platform()}")

    # Python 3.8+ チェック
    if sys.version_info < (3, 8):
        print("[ERROR] Python 3.8+ required")
        return False

    return True

def test_pytorch():
    """PyTorchテスト"""
    try:
        print(f"[TEST] PyTorch version: {torch.__version__}")

        # CUDA利用可能かチェック
        if torch.cuda.is_available():
            print(f"[TEST] CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"[TEST] CUDA version: {torch.version.cuda}")
        else:
            print("[WARNING] CUDA not available, using CPU")

        # 基本的なテンソル操作テスト
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = x + y
        print("[TEST] Basic tensor operations: OK")

        return True

    except Exception as e:
        print(f"[ERROR] PyTorch test failed: {e}")
        return False

def test_transformers():
    """Transformersライブラリテスト"""
    try:
        import transformers
        print(f"[TEST] Transformers version: {transformers.__version__}")

        # AutoModel, AutoTokenizerのインポートテスト
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("[TEST] Auto imports: OK")

        return True

    except Exception as e:
        print(f"[ERROR] Transformers test failed: {e}")
        return False

def test_peft():
    """PEFTライブラリテスト"""
    try:
        import peft
        print(f"[TEST] PEFT version: {peft.__version__}")

        # LoRA設定テスト
        from peft import LoraConfig
        config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
        print("[TEST] LoRA config: OK")

        return True

    except Exception as e:
        print(f"[ERROR] PEFT test failed: {e}")
        return False

def test_project_structure():
    """プロジェクト構造テスト"""
    required_files = [
        "scripts/data/create_aegis_high_quality_dataset.py",
        "scripts/evaluation/setup_lm_eval_elyza.py",
        "scripts/evaluation/run_llama_cpp_ab_test.py",
        "scripts/evaluation/analyze_ab_test_stats.py",
        "scripts/evaluation/prepare_hf_upload.py",
        "scripts/utils/system_monitor.py",
        "auto_ab_test_pipeline.bat"
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print("[WARNING] Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        return False

    print("[TEST] All required files present")
    return True

def main():
    """メインシステムテスト"""
    print("🚀 MOONSHOT System Test Starting...")
    print("=" * 50)

    tests = [
        ("Python Environment", test_python_environment),
        ("PyTorch", test_pytorch),
        ("Transformers", test_transformers),
        ("PEFT", test_peft),
        ("Project Structure", test_project_structure)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🧪 Testing {test_name}...")
        try:
            if test_func():
                print(f"[PASS] {test_name}")
                passed += 1
            else:
                print(f"[FAIL] {test_name}")
        except Exception as e:
            print(f"[ERROR] {test_name}: {e}")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All system tests passed! MOONSHOT ready to launch.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
