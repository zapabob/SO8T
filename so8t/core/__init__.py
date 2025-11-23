"""
SO8T Core Components

This module contains the core SO(8) transformation components including:
- SO(8) rotation gates and layers
- PET regularization
- Self-verification mechanisms
- Triality heads
- Transformer implementations
"""

from .attention_so8 import SO8Attention
from .mlp_so8 import SO8MLP
from .pet_regularizer import PETRegularizer
from .so8t_layer import SO8TLayer
from .transformer import SO8TTransformer
from .triality_heads import TrialityHeads
from .self_verification import SelfVerification
from .triple_reasoning_agent import TripleReasoningAgent
from .burn_in import BurnInProcessor

# Import from integrated models
try:
    from .safety_aware_so8t import SafetyAwareSO8T
    from .so8t_thinking_model import SO8TThinkingModel
    from .strict_so8_rotation_gate import StrictSO8RotationGate
    from .thinking_tokens import ThinkingTokens
except ImportError:
    pass  # Optional components

__all__ = [
    'SO8Attention',
    'SO8MLP',
    'PETRegularizer',
    'SO8TLayer',
    'SO8TTransformer',
    'TrialityHeads',
    'SelfVerification',
    'TripleReasoningAgent',
    'BurnInProcessor',
]