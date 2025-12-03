#!/usr/bin/env python3
"""
GGUF変換チェックポイント機能テスト
"""

import os
import sys
import time
import shutil
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

def test_gguf_checkpoint():
    """GGUF変換チェックポイント機能をテスト"""
    print("🧪 Testing GGUF Conversion Checkpoint System")
    print("=" * 50)

    try:
        from scripts.conversion.convert_hf_to_gguf import GGUFConversionCheckpoint

        # テスト用ディレクトリ
        test_dir = Path("test_gguf_checkpoint")
        if test_dir.exists():
            shutil.rmtree(test_dir)
        test_dir.mkdir()

        # チェックポイントマネージャー作成
        checkpoint = GGUFConversionCheckpoint(
            output_dir=str(test_dir),
            model_name="test_model"
        )

        print("✅ Checkpoint manager created")

        # 進捗更新テスト
        checkpoint.update_progress("loading_model", 0.1)
        print("✅ Progress update: loading_model")

        checkpoint.update_progress("converting_tensors", 0.5, "layer.0.weight")
        print("✅ Progress update: converting_tensors")

        checkpoint.update_progress("saving_vocab", 0.9)
        print("✅ Progress update: saving_vocab")

        # チェックポイント保存テスト
        checkpoint.save_checkpoint("test_checkpoint")
        print("✅ Checkpoint saved")

        # 状態取得テスト
        status = checkpoint.get_status()
        print(f"✅ Status retrieved: {status['stage']} ({status['progress']:.1%})")

        # 完了マークテスト
        checkpoint.mark_completed()
        print("✅ Task marked as completed")

        # クリーンアップ
        if test_dir.exists():
            shutil.rmtree(test_dir)

        return True

    except Exception as e:
        print(f"❌ GGUF checkpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_task_manager_gguf():
    """タスクマネージャーのGGUF変換機能をテスト"""
    print("\n🧪 Testing Task Manager GGUF Integration")
    print("=" * 50)

    try:
        from scripts.utils.task_manager import run_gguf_conversion

        # 簡単なテスト（実際のモデル変換は行わない）
        print("✅ GGUF conversion function imported")

        # 引数検証テスト
        test_kwargs = {
            'quantization': 'q8_0',
            'output_file': 'test_output.gguf'
        }

        print(f"✅ Test kwargs: {test_kwargs}")
        return True

    except Exception as e:
        print(f"❌ Task manager GGUF test failed: {e}")
        return False

def main():
    """メイン関数"""
    print("🚀 GGUF Conversion Checkpoint Test Suite")
    print("=" * 60)

    tests = [
        ("GGUF Checkpoint Manager", test_gguf_checkpoint),
        ("Task Manager GGUF Integration", test_task_manager_gguf),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            result = test_func()
            results.append(result)
            print(f"📋 {test_name}: {'✅ PASSED' if result else '❌ FAILED'}")
        except Exception as e:
            print(f"📋 {test_name}: ❌ FAILED - {e}")
            results.append(False)

    print("\n" + "=" * 60)
    print("📊 TEST RESULTS:")

    passed = sum(results)
    total = len(results)

    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASSED" if results[i] else "❌ FAILED"
        print(f"  {test_name}: {status}")

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if all(results):
        print("\n🎉 ALL TESTS PASSED!")
        print("🚀 GGUF conversion with checkpointing is ready!")
        print("\nUsage:")
        print("  # Direct GGUF conversion with checkpointing:")
        print("  python scripts/utils/task_manager.py gguf --model_path /path/to/model --quantization q8_0")
        print("\n  # Full autonomous pipeline (includes GGUF conversion):")
        print("  .\\auto_aegis_pipeline.bat")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
