#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改良型ムーンショットパイプラインの簡易テスト
"""

import sys
import os
import time

# SO8Tディレクトリをパスに追加
sys.path.insert(0, os.getcwd())

def test_pipeline():
    print("Testing Enhanced Moonshot Pipeline (Simplified)")
    print("=" * 60)

    try:
        # パイプラインインポート
        print("1. Importing pipeline...")
        import enhanced_moonshot_pipeline
        print("   [OK] Import successful")

        # パイプライン初期化
        print("2. Initializing pipeline...")
        pipeline = enhanced_moonshot_pipeline.EnhancedMoonshotPipeline()
        print("   [OK] Initialization successful")

        # 基本属性確認
        print("3. Checking attributes...")
        print(f"   Model path: {pipeline.boreas_model_path}")
        print(f"   Device: {pipeline.device}")
        print("   [OK] Attributes OK")

        # 再開機能テスト
        print("4. Testing resume functionality...")
        can_resume = pipeline.attempt_resume()
        print(f"   Resume available: {can_resume}")
        print("   [OK] Resume test completed")

        # 2024-2026最先端手法設定テスト
        print("5. Testing advanced techniques integration...")
        pipeline.execute_advanced_techniques_integration()
        print("   [OK] Advanced techniques configured")

        # GRPO統合テスト
        print("6. Testing DeepSeek GRPO integration...")
        pipeline.execute_deepseek_grpo_integration()
        print("   [OK] GRPO integration completed")

        # mHC多様体統合テスト
        print("7. Testing mHC manifold integration...")
        pipeline.execute_mhc_manifold_integration()
        print("   [OK] mHC manifold integration completed")

        # 幾何学的スケーリング統合テスト
        print("8. Testing geometric scaling integration...")
        pipeline.execute_geometric_scaling_integration()
        print("   [OK] Geometric scaling integration completed")

        print("=" * 60)
        print("SUCCESS: All tests passed!")
        print("Pipeline is ready for full execution")
        print("")
        print("To run the complete pipeline:")
        print("   python enhanced_moonshot_pipeline.py")
        print("")
        print("To run with subagents and progress monitoring:")
        print("   python skills/quantization-evaluation-pipeline/scripts/run_with_subagents.py")
        print("")
        print("To run quantization evaluation pipeline:")
        print("   python skills/quantization-evaluation-pipeline/scripts/quantization_evaluation_pipeline.py --model models/aegis_v25_final")

        return True

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)