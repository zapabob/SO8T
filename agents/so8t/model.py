from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    dropout: float
    num_labels: int
    max_seq_len: int
    gate_order: Sequence[str]


def _rotation_from_skew(matrix: torch.Tensor) -> torch.Tensor:
    skew = matrix - matrix.transpose(-1, -2)
    return torch.matrix_exp(skew)


class SO8Gate(nn.Module):
    """Applies learnable SO(8) rotations per channel block."""

    def __init__(self, d_model: int, order: Sequence[str]):
        super().__init__()
        if d_model % 8 != 0:
            raise ValueError("d_model must be divisible by 8 for SO(8) gating.")
        self.blocks = d_model // 8
        self.order = list(order)
        for name in ("R_env", "R_safe", "R_cmd"):
            if name not in self.order:
                self.order.append(name)
        self.params = nn.ParameterDict(
            {
                name: nn.Parameter(torch.zeros(self.blocks, 8, 8))
                for name in {"R_env", "R_safe", "R_cmd"}
            }
        )

    def _matrix(self, label: str) -> torch.Tensor:
        return _rotation_from_skew(self.params[label])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: [batch, seq, dim]
        bsz, seq_len, dim = inputs.shape
        x = inputs.reshape(bsz, seq_len, self.blocks, 8)
        for label in self.order:
            matrix = self._matrix(label)
            x = torch.einsum("btcn,cnm->btcm", x, matrix)
        return x.reshape(bsz, seq_len, dim)


class SO8TBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, gate_order: Sequence[str]):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.gate = SO8Gate(d_model, gate_order)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # attention_mask: [batch, seq] where 1 = keep
        key_padding_mask = attention_mask == 0
        attn_output, attn_weights = self.attn(
            hidden_states,
            hidden_states,
            hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        attn_output = self.gate(attn_output)
        hidden_states = hidden_states + self.dropout(attn_output)
        hidden_states = self.norm1(hidden_states)

        ff = self.linear2(self.dropout(F.gelu(self.linear1(hidden_states))))
        hidden_states = hidden_states + self.dropout(ff)
        hidden_states = self.norm2(hidden_states)

        pet_loss = self._pet_curvature(hidden_states, attention_mask)
        return hidden_states, pet_loss

    @staticmethod
    def _pet_curvature(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Compute second difference along time dimension
        if hidden_states.size(1) < 3:
            return torch.zeros((), device=hidden_states.device)
        x0 = hidden_states[:, :-2, :]
        x1 = hidden_states[:, 1:-1, :]
        x2 = hidden_states[:, 2:, :]
        diff2 = x0 - 2 * x1 + x2
        curvature = torch.sum(diff2 ** 2, dim=-1)
        valid = attention_mask[:, :-2] * attention_mask[:, 1:-1] * attention_mask[:, 2:]
        valid = valid.to(curvature.dtype)
        denom = valid.sum().clamp(min=1.0)
        return (curvature * valid).sum() / denom


class TinyNCGTransformerSO8T(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.max_seq_len, config.d_model)
        self.layers = nn.ModuleList(
            [
                SO8TBlock(config.d_model, config.n_heads, config.d_ff, config.dropout, config.gate_order)
                for _ in range(config.n_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.d_model, config.num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        positions = positions.clamp(max=self.position_embeddings.num_embeddings - 1)

        hidden_states = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        hidden_states = self.dropout(hidden_states)

        pet_losses: List[torch.Tensor] = []
        for layer in self.layers:
            hidden_states, pet_loss = layer(hidden_states, attention_mask)
            pet_losses.append(pet_loss)

        hidden_states = self.layer_norm(hidden_states)

        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        masked_hidden = hidden_states * mask
        summed = masked_hidden.sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / denom
        logits = self.classifier(pooled)

        pet_total = torch.stack(pet_losses).mean()
        return {"logits": logits, "pet_loss": pet_total}

    def pet_loss(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.forward(input_ids, attention_mask)["pet_loss"]


def build_model(config: ModelConfig) -> TinyNCGTransformerSO8T:
    return TinyNCGTransformerSO8T(config)
