"""
SO8T Training Module

This module provides training utilities and pipelines for SO8T models including:
- QLoRA training
- PET regularization training
- Loss functions
- Training pipelines
"""

from .loss_functions import PETLoss, SO8TCompositeLoss
from .qlora import QLoRATrainer
from .trainer_with_pet import TrainerWithPET

# Alias for backward compatibility
SO8TLoss = SO8TCompositeLoss

__all__ = [
    'PETLoss',
    'SO8TLoss',
    'SO8TCompositeLoss',
    'QLoRATrainer',
    'TrainerWithPET',
]