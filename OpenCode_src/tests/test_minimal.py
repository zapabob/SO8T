#!/usr/bin/env python3
"""Minimal test script to identify initialization issues"""

import sys
import os
import json

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
try:
    with open('aegis_v2_test_config.json', 'r') as f:
        config = json.load(f)
    print(f"   Config: OK (SO8T enabled: {config.get('so8t', {}).get('enable_so8t', False)})")
except Exception as e:
    print(f"   Config: FAILED - {e}")

# Test 4: SO(8) components
print("4. Testing SO(8) components...")
models_dir = os.path.join('models')
model_dir = os.path.join(models_dir, 'Borea-Phi-3.5-mini-Instruct-Jp')
sys.path.insert(0, models_dir)
sys.path.insert(0, model_dir)

try:
    from so8_rotation_adapter import SO8PhaseTransitionAnnealer
    print("   SO(8) components: OK")
except Exception as e:
    print(f"   SO(8) components: FAILED - {e}")

print("=== Test Complete ===")


