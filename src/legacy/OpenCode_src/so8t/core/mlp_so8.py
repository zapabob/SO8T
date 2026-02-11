"""
Feed-forward network tailored for SO8T transformer blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


_Activation = Literal["silu", "gelu"]


def _activation_fn(x: Tensor, kind: _Activation) -> Tensor:
    if kind == "silu":
        return F.silu(x)
    if kind == "gelu":
        return F.gelu(x)
    raise ValueError(f"Unsupported activation {kind}")


@dataclass
class FeedForwardConfig:
    hidden_size: int
    intermediate_size: int
    activation: _Activation = "silu"
    dropout: float = 0.0


class SO8FeedForward(nn.Module):
    """
    Group-aware feed-forward layer with optional gated linear unit.
    """

    def __init__(self, config: FeedForwardConfig) -> None:
        super().__init__()
        self.config = config

        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: Tensor) -> Tensor:
        up = self.fc1(x)
        gate, value = up.chunk(2, dim=-1)
        activated = _activation_fn(gate, self.config.activation)
        fused = activated * value
        fused = self.fc2(fused)
        return self.dropout(fused)
