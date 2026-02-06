#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改良型ムーンショットパイプラインのテストスクリプト
"""

import sys
import os
import pytest
sys.path.insert(0, os.getcwd())

if not os.getenv("SO8T_RUN_PIPELINE_TEST"):
    pytest.skip("Pipeline integration test skipped in CI", allow_module_level=True)


try:
    print("Testing enhanced moonshot pipeline import...")
    import enhanced_moonshot_pipeline
    print("[OK] Import successful")

    print("Testing pipeline initialization...")
    pipeline = enhanced_moonshot_pipeline.EnhancedMoonshotPipeline()
    print("[OK] Initialization successful")

    print("Testing basic attributes...")
    print(f"Model path: {pipeline.boreas_model_path}")
    print(f"Device: {pipeline.device}")
    print("[OK] Basic attributes OK")

    print("Testing resume functionality...")
    can_resume = pipeline.attempt_resume()
    print(f"Resume available: {can_resume}")
    print("[OK] Resume test completed")

    print("\n[DONE] All tests passed! Pipeline is ready to run.")

except Exception as e:
    print(f"[NG] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)