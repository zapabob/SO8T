"""
SO(8) rotation enhanced self-attention module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _cayley_rotation(theta: Tensor) -> Tensor:
    """
    Convert skew-symmetric parameter blocks into orthogonal rotations via the Cayley transform.
    """
    # theta expected shape [..., 8, 8]
    skew = theta - theta.transpose(-1, -2)
    eye = torch.eye(8, device=theta.device, dtype=theta.dtype).unsqueeze(0).expand(skew.shape)
    a = eye - 0.5 * skew
    b = eye + 0.5 * skew
    # solve (I - 0.5A)^{-1} (I + 0.5A)
    rot = torch.linalg.solve(a, b)
    return rot


def apply_so8_rotation(x: Tensor, theta: Tensor) -> Tensor:
    """
    Apply block-wise SO(8) rotation to the last dimension of the tensor.
    """
    b, t, d = x.shape
    assert d % 8 == 0, "hidden dimension must be divisible by 8"
    blocks = d // 8
    rot = _cayley_rotation(theta.view(blocks, 8, 8))
    x_blocks = x.view(b, t, blocks, 8)
    y = torch.einsum("btno,noa->btna", x_blocks, rot)
    return y.reshape(b, t, d)


@dataclass
class AttentionOutput:
    context: Tensor
    weights: Optional[Tensor]
    new_cache: Optional[Tuple[Tensor, Tensor]]


class SO8SelfAttention(nn.Module):
    """
    Multi-head self-attention with optional SO(8) post-rotation gate.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        rotation_enabled: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if hidden_size % 8 != 0:
            raise ValueError("hidden_size must be divisible by 8")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = nn.Dropout(dropout)
        self.rotation_enabled = rotation_enabled

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        if rotation_enabled:
            theta = torch.zeros(hidden_size // 8, 8, 8)
            nn.init.trunc_normal_(theta, std=0.02)
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer("theta", torch.zeros(hidden_size // 8, 8, 8))

    def _shape(self, proj: Tensor, batch: int, seq: int) -> Tensor:
        return proj.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> AttentionOutput:
        batch, seq, _ = hidden_states.shape

        q = self._shape(self.q_proj(hidden_states), batch, seq)
        k = self._shape(self.k_proj(hidden_states), batch, seq)
        v = self._shape(self.v_proj(hidden_states), batch, seq)

        if past_key_value is not None:
            pk, pv = past_key_value
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores += attention_mask
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch, seq, self.hidden_size)
        context = self.o_proj(context)

        if self.rotation_enabled:
            context = apply_so8_rotation(context, self.theta)

        cache = (k, v) if use_cache else None
        weights = attn if output_attentions else None
        return AttentionOutput(context=context, weights=weights, new_cache=cache)
