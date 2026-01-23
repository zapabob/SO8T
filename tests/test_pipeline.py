#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改良型ムーンショットパイプラインのテストスクリプト
"""

import sys
import os
sys.path.insert(0, os.getcwd())

try:
    print("Testing enhanced moonshot pipeline import...")
    import enhanced_moonshot_pipeline
    print("✅ Import successful")

    print("Testing pipeline initialization...")
    pipeline = enhanced_moonshot_pipeline.EnhancedMoonshotPipeline()
    print("✅ Initialization successful")

    print("Testing basic attributes...")
    print(f"Model path: {pipeline.boreas_model_path}")
    print(f"Device: {pipeline.device}")
    print("✅ Basic attributes OK")

    print("Testing resume functionality...")
    can_resume = pipeline.attempt_resume()
    print(f"Resume available: {can_resume}")
    print("✅ Resume test completed")

    print("\n🎉 All tests passed! Pipeline is ready to run.")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)