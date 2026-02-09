@echo off
REM SO8T Pipeline Quick Test

set PYTHONPATH=%CD%\src;%PYTHONPATH%

echo ============================================
echo SO8T Pipeline Quick Test
echo ============================================

echo.
echo Testing imports...
py -3 -c "
import sys
sys.path.insert(0, 'src')

# Test core models
from core.models.so8t_moe_router import SO8TrialityRouter, SO8MoELayer
import torch
print('[OK] SO8T MoE Router imported')

# Test evolution
from training.evolution.ebbinghaus_forgetting import EbbinghausForgettingCurve
curve = EbbinghausForgettingCurve()
curve.update([1,2,3], is_reinforced=[True, False, False])
print('[OK] EbbinghausForgettingCurve imported')

# Test checkpoint manager
from utils.checkpoint_manager import RollingCheckpointManager, CheckpointConfig
print('[OK] RollingCheckpointManager imported')

# Test progress tracker
from utils.progress_tracker import TrainingProgressTracker
print('[OK] TrainingProgressTracker imported')

print()
print('All imports successful!')
"

echo ============================================
echo Test Complete
echo ============================================
pause
