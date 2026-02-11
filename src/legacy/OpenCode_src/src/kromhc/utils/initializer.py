"""
Initialization utilities for KromHC modules.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def init_kronecker_factors(
    factor_a: nn.Parameter,
    factor_b: nn.Parameter,
    *,
    gain: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Initialize Kronecker factors with orthogonal matrices."""
    nn.init.orthogonal_(factor_a, gain=gain)
    nn.init.orthogonal_(factor_b, gain=gain)
    return factor_a.data, factor_b.data


def init_kromhc_module(module: nn.Module, gain: float = 1.0) -> None:
    """Initialize any KromHC module with default settings."""
    for name, param in module.named_parameters():
        if "factor_A" in name or "factor_B" in name:
            nn.init.orthogonal_(param, gain=gain)
        elif param.dim() > 1:
            nn.init.xavier_uniform_(param, gain=gain)
        else:
            nn.init.zeros_(param)
