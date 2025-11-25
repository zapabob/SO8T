"""
SO8T Core Components

This module contains the core SO(8) transformation components including:
- SO(8) rotation gates and layers
- PET regularization
- Self-verification mechanisms
- Triality heads
- Transformer implementations
"""

from .attention_so8 import SO8SelfAttention
from .mlp_so8 import SO8FeedForward

# Alias for backward compatibility
SO8MLP = SO8FeedForward

from .pet_regularizer import PETRegularizer
from .so8t_layer import SO8TRotationGate, SO8TAttentionWrapper

# Alias for backward compatibility
SO8TLayer = SO8TRotationGate

from .transformer import SO8TTransformerBlock, SO8TModel

# Alias for backward compatibility
SO8TTransformer = SO8TTransformerBlock
from .triality_heads import TrialityHead

# Alias for backward compatibility
TrialityHeads = TrialityHead
from .self_verification import SelfVerifier
from .triple_reasoning_agent import TripleReasoningAgent
from .burn_in import BurnInManager

# Alias for backward compatibility
SelfVerification = SelfVerifier
BurnInProcessor = BurnInManager

# Import from integrated models
try:
    from .safety_aware_so8t import SafetyAwareSO8T
    from .so8t_thinking_model import SO8TThinkingModel
    from .strict_so8_rotation_gate import StrictSO8RotationGate
    from .thinking_tokens import ThinkingTokens
except ImportError:
    pass  # Optional components

__all__ = [
    'SO8SelfAttention',
    'SO8MLP',
    'PETRegularizer',
    'SO8TLayer',
    'SO8TTransformer',
    'TrialityHeads',
    'SelfVerification',
    'TripleReasoningAgent',
    'BurnInProcessor',
]

# Alias for backward compatibility
SO8Attention = SO8SelfAttention