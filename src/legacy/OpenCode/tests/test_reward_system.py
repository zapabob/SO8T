#!/usr/bin/env python3
"""Test PPOAlignmentRewardSystem initialization"""

import sys
import os

print("=== PPOAlignmentRewardSystem Test ===")

# Setup paths
models_dir = os.path.join('models')
model_dir = os.path.join(models_dir, 'Borea-Phi-3.5-mini-Instruct-Jp')
sys.path.insert(0, models_dir)
sys.path.insert(0, model_dir)

# Test PPOAlignmentRewardSystem
print("1. Testing PPOAlignmentRewardSystem...")
try:
    from so8_rotation_adapter import PPOAlignmentRewardSystem
    print("   PPOAlignmentRewardSystem imported")

    reward_system = PPOAlignmentRewardSystem(hidden_size=3072)
    print("   PPOAlignmentRewardSystem created successfully")
except Exception as e:
    print(f"   PPOAlignmentRewardSystem creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=== PPOAlignmentRewardSystem Test Complete ===")
