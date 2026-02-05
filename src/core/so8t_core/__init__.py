"""
Core SO8T modules exposed as a convenience import surface.
"""

from .attention_so8 import SO8SelfAttention
from .mlp_so8 import SO8FeedForward
from .pet_regularizer import PETRegularizer, PETSchedule
from .triality_heads import TrialityHead, TrialityOutput
from .self_verification import SelfVerifier, VerificationResult
from .transformer import SO8TTransformerBlock, SO8TModelConfig, SO8TModel
from .so8t_layer import SO8TRotationGate, SO8TAttentionWrapper
from .burn_in import BurnInManager, burn_in_phi4_model

__all__ = [
    "SO8SelfAttention",
    "SO8FeedForward",
    "PETRegularizer",
    "PETSchedule",
    "TrialityHead",
    "TrialityOutput",
    "SelfVerifier",
    "VerificationResult",
    "SO8TTransformerBlock",
    "SO8TModelConfig",
    "SO8TModel",
    "SO8TRotationGate",
    "SO8TAttentionWrapper",
    "BurnInManager",
    "burn_in_phi4_model",
]
