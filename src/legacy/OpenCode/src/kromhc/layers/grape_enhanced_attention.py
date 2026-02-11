"""
Hybrid GRAPE-Enhanced SO(8) Attention Layer

Combines SO(8) triality with GRAPE (Group Representational Position Encoding)
for enhanced geometric reasoning capabilities.

References:
    - GRAPE: arXiv:2512.07805 (ICLR 2026)
    - SO(8) Triality: Implementation in so8t_core/so8t_layer.py
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..GRAPE import (
    GRAPEConfig,
    GRAPEMultiplicative,
    GRAPEAdditive,
    GRAPE,
    GRAPEUtils,
)
from ..kronecker_residual import KroneckerResidualConfig, KroneckerResidualStream
from ...utils.errors import ModelDimensionError


@dataclass
class SO8T_GRAPEConfig:
    """SO(8) GRAPE-Enhanced Attention configuration."""

    # Model dimensions
    hidden_size: int = 2048
    num_heads: int = 8
    head_dim: int = 256

    # GRAPE configuration
    use_grape: bool = True
    grape_type: str = "multiplicative"
    dim: int = 2  # SO(2) for SO(3)

    # Learning rate for frequency parameters
    freq_lr: float = 1e-4

    # Maximum position index
    max_positions: int = 8192

    # Learnable frequency
    learnable_freq: bool = False

    # Dropout
    dropout: float = 0.1

    # KromHC integration
    use_kromhc_residual: bool = False
    n_streams: int = 1


class SO8T_GRAPEAttention(nn.Module):
    """SO(8) triality attention enhanced with GRAPE.

    Combines:
    - SO(8) rotation gates (triality-preserving)
    - GRAPE multiplicative/additive position encoding
    - KromHC residual streams (optional)
    - Triality transformations for geometric reasoning
    """

    def __init__(self, config: Optional[SO8T_GRAPEConfig] = None):
        if config is None:
            config = SO8T_GRAPEConfig()

        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        # SO(8) position encoding
        if config.use_grape:
            grape_cfg = GRAPEConfig(
                dim=config.dim,
                max_positions=config.max_positions,
                group_type="so3" if config.dim == 2 else "so2",
                freq_lr=config.freq_lr,
                learnable_freq=config.learnable_freq,
            )
            self.grape = GRAPE(grape_cfg, dim=config.hidden_size)
        else:
            self.grape = None

        # Rotation gate (triality-preserving SO(8) operation)
        # Standard SO(8) rotation matrices per head
        self.register_buffer(
            "so8_rotations",
            torch.randn(config.num_heads, 8, 8, dtype=torch.float32) * 0.01,
        )

        # KromHC residual stream (optional)
        self.use_kromhc = config.use_kromhc
        self.residual_stream: Optional[KroneckerResidualStream] = None
        if config.use_kromhc:
            if config.hidden_size % config.n_streams != 0:
                raise ModelDimensionError(
                    expected=config.n_streams,
                    actual=config.hidden_size,
                    param_name="n_streams",
                )
            hidden_dim = config.hidden_size // config.n_streams
            kron_cfg = KroneckerResidualConfig(
                n_streams=config.n_streams,
                hidden_dim=hidden_dim,
                learnable_projection=False,
            )
            self.residual_stream = KroneckerResidualStream(kron_cfg)

        # Output projections
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # Layer normalization
        self.norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        logger.info(
            f"SO8T_GRAPEAttention init: "
            f"hidden={config.hidden_size}, heads={config.num_heads}, "
            f"use_grape={config.use_grape}, "
            f"use_kromhc={config.use_kromhc}"
        )

    def _apply_so8_rotation(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Apply SO(8) rotation to Q, K, V.

        Args:
            Q, K, V: [batch, seq, head_dim]

        Returns:
            Rotated tensors Q_rot, K_rot, V_rot
        """
        batch, seq, head_dim = Q.shape

        # Extract rotation matrices for each head
        # Shape: [num_heads, 8, 8, head_dim]
        so8_rotations = self.so8_rotations
        # [batch, num_heads, 8, 8, head_dim]

        for h in range(self.num_heads):
            rot_matrix = so8_rotations[h, :, :8, :, :]

            # Apply to Q: [batch, seq, head_dim] @ [8, 8, head_dim]
            Q_rot = torch.einsum("bhqh->bsk", Q, rot_matrix)

            # Apply to K: [batch, seq, head_dim] @ [8, 8, 8, head_dim]
            K_rot = torch.einsum("bhqk->bsh", K, rot_matrix)

            # Apply to V: [batch, seq, head_dim] @ [8, 8, 8, head_dim]
            V_rot = torch.einsum("bhqv->bshv", V, rot_matrix)

            so8_rotations[h, :, :8, :, :] = Q_rot
            so8_rotations[:, :, h, :, :] = K_rot

            so8_rotations[h, :, h, :, :] = V_rot

        return Q_rot, K_rot, V_rot

    def _triality_transform(
        self,
        x: torch.Tensor,
        rotation_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Apply three triality transformations for enhanced geometric reasoning."""
        # Extract x into two parts
        batch, seq, dim = x.shape

        # Apply triality transformations
        outputs = []
        for t in range(3):
            # Transformation 1: 90-degree rotation around Z-axis
            W1 = torch.tensor(
                [[0, -1], [1, 0], [0, 0], [-1, 1, 0]], dtype=x.dtype, device=x.device
            )

            # Transformation 2: 90-degree rotation around X-axis
            W2 = torch.tensor(
                [[1, 0, 0], [0, 0], [0, 0, 1], [0, 1, -1]],
                dtype=x.dtype,
                device=x.device,
            )

            # Transformation 3: 90-degree rotation around Y-axis
            W3 = torch.tensor(
                [[0, 0, 1], [0, 0, -1, 0], [-1, 0, 0], [0, -1, 0]],
                dtype=x.dtype,
                device=x.device,
            )

            for W in [W1, W2, W3]:
                transformed = torch.matmul(W, x)
                outputs.append(transformed)

        # Combine triality outputs
        combined = torch.stack(outputs, dim=-1).mean(dim=-1)
        outputs.append(combined)

        # Concatenate heads
        output = torch.cat(outputs, dim=-1)

        return output

    def _geometric_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attention with SO(8) geometric transformations."""
        # Standard attention
        Q = (
            self.q_proj(query)
            .view(query.shape[0], query.shape[1], self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.k_proj(key)
            .view(query.shape[0], query.shape[1], self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.v_proj(value)
            .view(query.shape[0], query.shape[1], self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Apply SO(8) rotation
        Q_rot, K_rot, V_rot = self._apply_so8_rotation(Q, K, V)

        # Geometric attention with rotated Q, K, V
        attn_output, _ = self._compute_attention(Q_rot, K_rot, V_rot, attn_mask)

        return attn_output

    def _compute_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attention scores."""
        batch, seq_len, _ = Q.shape

        # Scale Q and K by sqrt(d_k)
        scale = float(math.sqrt(self.head_dim)) ** -0.5

        scores = torch.matmul(Q, K.transpose(-2, -1))  # [seq, dim, head_dim]
        return scores

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through SO(8) GRAPE enhanced attention.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]

        Returns:
            Enhanced attention output with GRAPE position encoding
        """
        batch, seq_len, hidden_size = x.shape

        # Layer normalization
        x_norm = self.norm(x)

        # GRAPE position encoding (if enabled)
        if self.grape is not None:
            x_pe = self.grape(x)
        else:
            x_pe = x

        # Apply triality transformations (for geometric reasoning)
        x_tri = self._triality_transform(x_norm, self.so8_rotations)

        # Compute attention with geometric enhancements
        attended = self._geometric_attention(x_tri, x_tri, x_tri, attn_mask)

        # Residual connection with KromHC (optional)
        if not self.residual_stream:
            residual = attended
        else:
            batch, seq_len, hidden_size = attended.shape
            reshaped = attended.reshape(batch * seq_len, hidden_size)
            residual = self.residual_stream(reshaped)
            output = residual + attended
            return output

        return output
