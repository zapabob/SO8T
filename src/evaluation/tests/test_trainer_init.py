#!/usr/bin/env python3
"""Test PPOTrainer initialization step by step"""

import sys
import os
import json

print("=== PPOTrainer Init Test ===")

# Setup paths
models_dir = os.path.join('models')
model_dir = os.path.join(models_dir, 'Borea-Phi-3.5-mini-Instruct-Jp')
sys.path.insert(0, models_dir)
sys.path.insert(0, model_dir)

# Test 1: SO(8) components import
print("1. Importing SO(8) components...")
try:
    from so8_rotation_adapter import SO8PhaseTransitionAnnealer, ChaosInducedDiversityEnhancer
    print("   SO(8) components: OK")
except Exception as e:
    print(f"   SO(8) components: FAILED - {e}")
    sys.exit(1)

# Test 2: Config loading
print("2. Loading config...")
try:
    config_path = 'aegis_v2_test_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"   Config loaded: SO8T={config.get('so8t', {}).get('enable_so8t', False)}")
except Exception as e:
    print(f"   Config loading: FAILED - {e}")
    sys.exit(1)

# Test 3: PPOTrainer initialization
print("3. Initializing PPOTrainer...")
try:
    from src.training.train_aegis_v2_ppo_so8t import PPOTrainer
    print("   PPOTrainer class imported")

    trainer = PPOTrainer(config_path, 'models/Borea-Phi-3.5-mini-Instruct-Jp')
    print("   PPOTrainer initialized successfully")
except Exception as e:
    print(f"   PPOTrainer initialization: FAILED - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=== PPOTrainer Init Test Complete ===")


