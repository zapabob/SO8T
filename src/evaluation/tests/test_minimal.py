#!/usr/bin/env python3
"""Minimal test script to identify initialization issues"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("=== Minimal SO8T Test ===")

# Test 1: Basic imports
print("1. Testing basic imports...")
try:
    import torch
    print(f"   torch: OK (CUDA: {torch.cuda.is_available()})")
except Exception as e:
    print(f"   torch: FAILED - {e}")

try:
    import transformers
    print("   transformers: OK")
except Exception as e:
    print(f"   transformers: FAILED - {e}")

# Test 2: Unsloth import
print("2. Testing Unsloth import...")
try:
    from unsloth import FastLanguageModel
    print("   Unsloth: OK")
    UNSLOTH_AVAILABLE = True
except Exception as e:
    print(f"   Unsloth: FAILED - {e}")
    UNSLOTH_AVAILABLE = False

# Test 3: Config loading
print("3. Testing config loading...")
config_candidates = [
    project_root / "src" / "infrastructure" / "config" / "borea_training.json",
    project_root / "aegis_v2_test_config.json",
]
config_loaded = False
for config_path in config_candidates:
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"   Config: OK (loaded {config_path.name})")
            config_loaded = True
            break
        except Exception as e:
            print(f"   Config: FAILED - {e}")
if not config_loaded:
    print("   Config: SKIP - no config file found")

# Test 4: SO(8) components
print("4. Testing SO(8) components...")
try:
    from src.core.models.so8t_residual_adapter import SO8ResidualAdapter
    print("   SO(8) adapter: OK")
except Exception as e:
    print(f"   SO(8) adapter: FAILED - {e}")

# Test 5: Project utilities
print("5. Testing project utilities...")
try:
    from src.utils.path_resolver import PathResolver
    from src.utils.config_loader import ConfigLoader
    from src.utils.checkpoint_manager import RollingCheckpointManager
    print("   Project utilities: OK")
except Exception as e:
    print(f"   Project utilities: FAILED - {e}")

print("=== Test Complete ===")
