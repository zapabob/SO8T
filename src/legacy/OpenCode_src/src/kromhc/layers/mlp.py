"""
KromHC MLP layer with optional Kronecker residual stream.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ..core.kronecker_residual import KroneckerResidualConfig, KroneckerResidualStream
from ...utils.errors import ModelDimensionError


@dataclass
class KromHCMLPConfig:
    """Configuration for KromHC MLP."""

    embed_dim: int
    hidden_dim: int
    n_streams: int = 1
    dropout: float = 0.0
    use_kromhc: bool = True


class KromHCMLPLayer(nn.Module):
    """Feed-forward layer with optional KromHC residual stream."""

    def __init__(self, config: KromHCMLPConfig) -> None:
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.embed_dim, config.hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.hidden_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.use_kromhc = config.use_kromhc
        self.residual_stream: Optional[KroneckerResidualStream] = None
        if self.use_kromhc:
            if config.embed_dim % config.n_streams != 0:
                raise ModelDimensionError(
                    expected=config.embed_dim,
                    actual=config.n_streams,
                    param_name="n_streams",
                )
            hidden = config.embed_dim // config.n_streams
            kron_cfg = KroneckerResidualConfig(
                n_streams=config.n_streams,
                hidden_dim=hidden,
                learnable_projection=False,
            )
            self.residual_stream = KroneckerResidualStream(kron_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MLP to input (batch, seq, dim)."""
        y = self.fc2(self.dropout(self.act(self.fc1(x))))
        if not self.residual_stream:
            return y
        batch, seq, dim = y.shape
        reshaped = y.reshape(batch * seq, dim)
        projected = self.residual_stream(reshaped)
        return projected.reshape(batch, seq, dim)
