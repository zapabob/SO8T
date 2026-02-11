#!/usr/bin/env python3
"""Test minimal PPOTrainer initialization"""

import sys
import os
import json

print("=== Minimal PPOTrainer Test ===")

# Setup paths
models_dir = os.path.join('models')
model_dir = os.path.join(models_dir, 'Borea-Phi-3.5-mini-Instruct-Jp')
sys.path.insert(0, models_dir)
sys.path.insert(0, model_dir)

# Test config loading
print("1. Loading config...")
try:
    config_path = 'aegis_v2_test_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    print("   Config loaded successfully")
except Exception as e:
    print(f"   Config loading failed: {e}")
    sys.exit(1)

# Test PPOTrainer minimal init
print("2. Testing PPOTrainer minimal init...")
try:
    # Import minimal components
    from scripts.training.train_aegis_v2_ppo_so8t import PPOConfig
    print("   PPOConfig imported")

    # Create minimal trainer instance with just basic setup
    class MinimalPPOTrainer:
        def __init__(self, config_path, model_path):
            print("   MinimalPPOTrainer __init__ start")
            self.config_path = config_path
            self.model_path = model_path

            # Load config
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            print("   Config loaded in trainer")

            # PPO config only
            self.ppo_config = PPOConfig()
            print("   PPOConfig created")

            print("   MinimalPPOTrainer __init__ complete")

    trainer = MinimalPPOTrainer(config_path, 'models/Borea-Phi-3.5-mini-Instruct-Jp')
    print("   Minimal trainer created successfully")

except Exception as e:
    print(f"   Minimal trainer creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=== Minimal PPOTrainer Test Complete ===")
