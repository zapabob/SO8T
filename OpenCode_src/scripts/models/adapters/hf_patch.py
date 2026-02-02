"""
Utilities for patching Hugging Face transformer blocks with SO8T components.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator, List

from torch import nn

from so8t_core.attention_so8 import SO8SelfAttention
from so8t_core.transformer import SO8TModelConfig, SO8TTransformerBlock


def _replace_module(parent: nn.Module, name: str, new_module: nn.Module) -> None:
    parts: List[str] = name.split(".")
    current = parent
    for part in parts[:-1]:
        current = getattr(current, part)
    setattr(current, parts[-1], new_module)


def patch_attention_blocks(model: nn.Module, hidden_size: int, num_heads: int, dropout: float) -> None:
    """
    Replace all child modules named `self_attn` or `attention` with SO8 attention layers.
    """
    for name, module in model.named_modules():
        if name.endswith("self_attn") or name.endswith("attention"):
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            parent = model.get_submodule(parent_name) if parent_name else model
            replacement = SO8SelfAttention(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)
            _replace_module(parent, name.split(".")[-1], replacement)


def patch_with_so8_block(model: nn.Module, config: SO8TModelConfig) -> None:
    """
    Replace full transformer block modules with SO8TTransformerBlock.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.ModuleList):
            for idx, layer in enumerate(module):
                if hasattr(layer, "attn") and hasattr(layer, "mlp"):
                    module[idx] = SO8TTransformerBlock(config)


@contextmanager
def temporary_patch(model: nn.Module, patch_fn: Callable[[nn.Module], None]) -> Iterator[nn.Module]:
    """
    Context manager that applies a patch and restores the original parameters after the block.
    """
    original_state = model.state_dict()
    patch_fn(model)
    try:
        yield model
    finally:
        model.load_state_dict(original_state)
