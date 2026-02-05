#!/usr/bin/env python3
"""Test PPOTrainer initialization step by step"""

import sys
import os
import json

print("=== PPOTrainer Partial Init Test ===")

# Setup paths
models_dir = os.path.join('models')
model_dir = os.path.join(models_dir, 'Borea-Phi-3.5-mini-Instruct-Jp')
sys.path.insert(0, models_dir)
sys.path.insert(0, model_dir)

# Test step by step initialization
print("1. Loading config...")
try:
    config_path = 'aegis_v2_test_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    print("   Config loaded successfully")
except Exception as e:
    print(f"   Config loading failed: {e}")
    sys.exit(1)

print("2. Testing step-by-step PPOTrainer init...")
try:
    from src.training.train_aegis_v2_ppo_so8t import PPOConfig
    from so8_rotation_adapter import (
        SO8PhaseTransitionAnnealer,
        ChaosInducedDiversityEnhancer,
        PPOAlignmentRewardSystem
    )

    print("   Components imported")

    # Step 1: Basic setup
    print("   Step 1: Basic setup")
    config_path = 'aegis_v2_test_config.json'
    model_path = 'models/Borea-Phi-3.5-mini-Instruct-Jp'

    with open(config_path, 'r') as f:
        config = json.load(f)
    print("   Config loaded")

    # Step 2: PPO config
    print("   Step 2: PPO config")
    ppo_config = PPOConfig()
    print("   PPOConfig created")

    # Step 3: SO(8) components
    print("   Step 3: SO(8) components")
    phase_annealer = SO8PhaseTransitionAnnealer(
        initial_alpha=ppo_config.alpha_initial,
        target_alpha=ppo_config.alpha_target,
        annealing_steps=ppo_config.annealing_steps
    )
    print("   SO8PhaseTransitionAnnealer created")

    chaos_enhancer = ChaosInducedDiversityEnhancer(
        hidden_size=3072,
        chaos_intensity=ppo_config.chaos_intensity
    )
    print("   ChaosInducedDiversityEnhancer created")

    reward_system = PPOAlignmentRewardSystem(hidden_size=3072)
    print("   PPOAlignmentRewardSystem created")

    print("   All components initialized successfully")

except Exception as e:
    print(f"   Step-by-step init failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=== PPOTrainer Partial Init Test Complete ===")
