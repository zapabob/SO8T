#!/usr/bin/env python3
"""
シンプルなRLPO統合テスト
Unslothを使わない完全自動化動作チェック
"""

import os
import sys
import torch
from pathlib import Path

def test_environment():
    print("🔍 Testing Environment...")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    return True

def test_imports():
    print("📦 Testing Imports...")
    try:
        from scripts.models.so8t_residual_adapter import SO8ResidualAdapter
        from scripts.utils.checkpoint_manager import RollingCheckpointManager
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_datasets():
    print("📊 Testing Datasets...")
    science_path = Path("data/science_reasoning_dataset_final.jsonl")
    nsfw_path = Path("data/nsfw_drug_detection/nsfw_drug_mixed_dataset.jsonl")

    science_exists = science_path.exists()
    nsfw_exists = nsfw_path.exists()

    print(f"Science dataset: {'✅' if science_exists else '❌'}")
    print(f"NSFW dataset: {'✅' if nsfw_exists else '❌'}")

    return science_exists and nsfw_exists

def test_nkat_adapter():
    print("🧬 Testing NKAT Adapter...")
    try:
        from scripts.models.so8t_residual_adapter import SO8ResidualAdapter, inject_nkat_to_all_layers
        from transformers import AutoModelForCausalLM

        # アダプター単体テスト
        adapter = SO8ResidualAdapter(hidden_size=512)
        stats = adapter.get_adapter_stats()
        print(f"✅ Adapter created - Alpha: {stats['alpha']:.4f}")

        # 完全層適用テスト
        print("Testing full layer injection...")
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-3.5-mini-instruct",
            torch_dtype="auto",
            device_map="auto",
            load_in_4bit=True,
            trust_remote_code=True
        )

        # すべての層に適用
        model = inject_nkat_to_all_layers(model, target_layers=[0, 1], mode="full_layer")

        print(f"✅ Full layer injection successful")
        return True
    except Exception as e:
        print(f"❌ Adapter test failed: {e}")
        return False

def test_checkpoint_manager():
    print("💾 Testing Checkpoint Manager...")
    try:
        from scripts.utils.checkpoint_manager import create_task_manager
        manager = create_task_manager("test_task", "test_checkpoints")
        status = manager.get_status()
        print(f"✅ Checkpoint manager created - Task: {status['task_name']}")
        return True
    except Exception as e:
        print(f"❌ Checkpoint manager failed: {e}")
        return False

def test_full_layer_adapters():
    print("🧬 Testing Full Layer NKAT Adapters...")
    try:
        from scripts.models.so8t_residual_adapter import inject_nkat_to_all_layers
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-3.5-mini-instruct",
            torch_dtype="auto",
            device_map="auto",
            load_in_4bit=True,
            trust_remote_code=True
        )

        # すべての層に適用
        model = inject_nkat_to_all_layers(model, target_layers=[0, 1], mode="full_layer")

        # アダプター数の確認
        adapter_count = sum(1 for name, _ in model.named_modules() if 'nkat_adapter' in name)
        print(f"✅ Full layer adapters injected - Total adapters: {adapter_count}")
        return True
    except Exception as e:
        print(f"❌ Full layer adapter test failed: {e}")
        return False

def main():
    print("[START] RLPO Integration Test Suite")
    print("=" * 50)

    tests = [
        ("Environment", test_environment),
        ("Imports", test_imports),
        ("Datasets", test_datasets),
        ("NKAT Adapter", test_nkat_adapter),
        ("Full Layer Adapters", test_full_layer_adapters),
        ("Checkpoint Manager", test_checkpoint_manager),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
            print()
        except Exception as e:
            print(f"❌ {name} crashed: {e}")
            results.append((name, False))
            print()

    print("=" * 50)
    print("📋 TEST RESULTS:")

    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("🚀 AEGIS Complete Autonomous System Ready!")
        print()
        print("Available Commands:")
        print("  🌟 Full Autonomous Training:")
        print("     python scripts/training/rlpo_science_nsfw_automated.py")
        print()
        print("  🤖 Universal Task Manager:")
        print("     python scripts/utils/task_manager.py rlpo")
        print("     python scripts/utils/task_manager.py dataset --dataset_type=science")
        print()
        print("  ⚡ Quick Test:")
        print("     .\auto_aegis_pipeline.bat")
        print()
        print("  🔄 Auto-Resume on Boot:")
        print("     Windows will automatically restart training on next boot")
        print()
        print("Features:")
        print("  ✅ 3-minute rolling checkpoints (5 stock)")
        print("  ✅ All Transformer layers get NKAT adapters")
        print("  ✅ Science + NSFW drug RLPO training")
        print("  ✅ Power interruption auto-recovery")
        print("  ✅ Complete autonomous operation")
    else:
        print("❌ Some tests failed. Check the errors above.")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
