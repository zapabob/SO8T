"""
SO8T Models Package
"""

from .strict_so8_rotation_gate import StrictSO8RotationGate
from .safety_aware_so8t import SafetyAwareSO8TConfig, SafetyAwareSO8TModel
from .thinking_tokens import (
    get_thinking_tokens,
    add_thinking_tokens_to_tokenizer,
    get_token_ids,
    extract_thinking_and_final,
    format_thinking_output,
    build_thinking_prompt,
)
from .so8t_thinking_model import SO8TThinkingModel

__all__ = [
    "StrictSO8RotationGate",
    "SafetyAwareSO8TConfig",
    "SafetyAwareSO8TModel",
    "get_thinking_tokens",
    "add_thinking_tokens_to_tokenizer",
    "get_token_ids",
    "extract_thinking_and_final",
    "format_thinking_output",
    "build_thinking_prompt",
    "SO8TThinkingModel",
]


