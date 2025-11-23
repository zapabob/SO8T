"""
Minimal transformer stack composed of SO8T blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

from .attention_so8 import SO8SelfAttention
from .mlp_so8 import FeedForwardConfig, SO8FeedForward
from .pet_regularizer import PETRegularizer, PETSchedule


@dataclass
class SO8TModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_hidden_layers: int = 12
    intermediate_size: int = 2816
    max_position_embeddings: int = 4096
    dropout: float = 0.0
    attn_dropout: float = 0.0
    pet_schedule: PETSchedule = field(default_factory=PETSchedule)


class SO8TTransformerBlock(nn.Module):
    """
    Single SO8T block = LayerNorm -> Attention -> LayerNorm -> FeedForward + PET regularizer.
    """

    def __init__(self, config: SO8TModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size)
        self.attn = SO8SelfAttention(
            config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attn_dropout,
        )
        ff_config = FeedForwardConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dropout=config.dropout,
        )
        self.ff = SO8FeedForward(ff_config)
        self.ln_2 = nn.LayerNorm(config.hidden_size)
        self.pet = PETRegularizer(config.pet_schedule)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        pet_progress: float,
    ) -> Tuple[Tensor, Tensor]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_out = self.attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + attn_out.context

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        ff_out = self.ff(hidden_states)
        hidden_states = residual + ff_out

        pet_loss = self.pet(hidden_states, pet_progress)
        return hidden_states, pet_loss


class SO8TModel(nn.Module):
    """
    Transformer encoder model with PET auxiliary loss summation.
    """

    def __init__(self, config: SO8TModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.layers = nn.ModuleList([SO8TTransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        pet_progress: float = 0.0,
    ) -> Tuple[Tensor, Tensor]:
        batch, seq = input_ids.shape
        device = input_ids.device
        positions = torch.arange(seq, device=device).unsqueeze(0).expand(batch, seq)

        hidden_states = self.embeddings(input_ids) + self.position_embeddings(positions)

        pet_losses: List[Tensor] = []
        for block in self.layers:
            hidden_states, pet_loss = block(hidden_states, attention_mask, pet_progress)
            pet_losses.append(pet_loss)

        hidden_states = self.norm(hidden_states)
        pet_aux = torch.stack(pet_losses).sum()
        return hidden_states, pet_aux
