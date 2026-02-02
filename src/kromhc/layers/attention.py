"""
KromHC attention layer with optional Kronecker residual stream.
References:
    Zhou et al. (2026) arXiv:2601.21579
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ..core.kronecker_residual import KroneckerResidualConfig, KroneckerResidualStream
from ...utils.errors import ModelDimensionError
from ...utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class KromHCAttentionConfig:
    """Configuration for KromHC attention."""

    embed_dim: int
    num_heads: int
    n_streams: int = 1
    dropout: float = 0.0
    use_kromhc: bool = True


class KromHCAttentionLayer(nn.Module):
    """Multi-head attention with optional KromHC residual projection."""

    def __init__(self, config: KromHCAttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.attn = nn.MultiheadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.use_kromhc = config.use_kromhc
        self.residual_stream: Optional[KroneckerResidualStream] = None
        if self.use_kromhc:
            if config.embed_dim % config.n_streams != 0:
                raise ModelDimensionError(
                    expected=config.embed_dim,
                    actual=config.n_streams,
                    param_name="n_streams",
                )
            hidden_dim = config.embed_dim // config.n_streams
            kron_cfg = KroneckerResidualConfig(
                n_streams=config.n_streams,
                hidden_dim=hidden_dim,
                learnable_projection=False,
            )
            self.residual_stream = KroneckerResidualStream(kron_cfg)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply attention (batch, seq, dim)."""
        attn_output, _ = self.attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        if not self.residual_stream:
            return attn_output
        batch, seq, dim = attn_output.shape
        reshaped = attn_output.reshape(batch * seq, dim)
        projected = self.residual_stream(reshaped)
        return projected.reshape(batch, seq, dim)
