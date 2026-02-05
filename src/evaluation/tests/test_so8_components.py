#!/usr/bin/env python3
"""Test SO(8) components initialization"""

import sys
import os
import json

print("=== SO(8) Components Test ===")

# Setup paths
models_dir = os.path.join('models')
model_dir = os.path.join(models_dir, 'Borea-Phi-3.5-mini-Instruct-Jp')
sys.path.insert(0, models_dir)
sys.path.insert(0, model_dir)

# Test SO(8) components import
print("1. Importing SO(8) components...")
try:
    from so8_rotation_adapter import (
        SO8PhaseTransitionAnnealer,
        ChaosInducedDiversityEnhancer,
        PPOAlignmentRewardSystem
    )
    print("   SO(8) components imported successfully")
except Exception as e:
    print(f"   SO(8) components import failed: {e}")
    sys.exit(1)

# Test PPOConfig
print("2. Testing PPOConfig...")
try:
    from src.training.train_aegis_v2_ppo_so8t import PPOConfig
    ppo_config = PPOConfig()
    print(f"   PPOConfig created: alpha_initial={ppo_config.alpha_initial}")
except Exception as e:
    print(f"   PPOConfig creation failed: {e}")
    sys.exit(1)

# Test SO(8) components initialization
print("3. Testing SO(8) components initialization...")
try:
    phase_annealer = SO8PhaseTransitionAnnealer(
        initial_alpha=ppo_config.alpha_initial,
        target_alpha=ppo_config.alpha_target,
        annealing_steps=ppo_config.annealing_steps
    )
    print("   SO8PhaseTransitionAnnealer created successfully")
except Exception as e:
    print(f"   SO8PhaseTransitionAnnealer creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    chaos_enhancer = ChaosInducedDiversityEnhancer(
        hidden_size=3072,
        chaos_intensity=ppo_config.chaos_intensity
    )
    print("   ChaosInducedDiversityEnhancer created successfully")
except Exception as e:
    print(f"   ChaosInducedDiversityEnhancer creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    reward_system = PPOAlignmentRewardSystem(hidden_size=3072)
    print("   PPOAlignmentRewardSystem created successfully")
except Exception as e:
    print(f"   PPOAlignmentRewardSystem creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=== SO(8) Components Test Complete ===")
