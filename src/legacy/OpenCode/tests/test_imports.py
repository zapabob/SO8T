#!/usr/bin/env python3
"""
インポートテストスクリプト
"""

import sys
from pathlib import Path

# プロジェクトディレクトリを追加
project_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_dir))

print(f"Project directory: {project_dir}")
print(f"Python path: {sys.path[:3]}")

try:
    print("Testing imports...")
    
    # 基本モジュール
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    
    import numpy as np
    print(f"✓ NumPy: {np.__version__}")
    
    # プロジェクトモジュール
    from agents.so8t.model_safety import SafetyModelConfig, build_safety_model
    print("✓ Safety model imports OK")
    
    from safety_losses import SafetyAwareLoss, SafetyMetrics
    print("✓ Safety losses imports OK")
    
    from shared.data import DialogueDataset
    print("✓ Data module imports OK")
    
    print("\nAll imports successful! Training should work.")
    
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
