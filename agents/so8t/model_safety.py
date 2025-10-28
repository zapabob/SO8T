from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SafetyModelConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    dropout: float
    num_labels: int
    num_safety_labels: int  # REFUSE, ESCALATE, ALLOW
    max_seq_len: int
    gate_order: Sequence[str]
    safety_first: bool = True  # R_safe -> R_cmd の順序を強制


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

    def forward(self, inputs: torch.Tensor, safety_first: bool = False) -> torch.Tensor:
        # inputs: [batch, seq, dim]
        bsz, seq_len, dim = inputs.shape
        x = inputs.reshape(bsz, seq_len, self.blocks, 8)
        
        # 安全優先の場合は順序を変更
        if safety_first:
            # R_safe -> R_env -> R_cmd の順序で安全を優先
            safety_order = ["R_safe", "R_env", "R_cmd"]
            for label in safety_order:
                if label in self.order:
                    matrix = self._matrix(label)
                    x = torch.einsum("btcn,cnm->btcm", x, matrix)
        else:
            # 通常の順序
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

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, safety_first: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        # attention_mask: [batch, seq] where 1 = keep
        key_padding_mask = attention_mask == 0
        attn_output, attn_weights = self.attn(
            hidden_states,
            hidden_states,
            hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        attn_output = self.gate(attn_output, safety_first=safety_first)
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


class SafetyAwareSO8T(nn.Module):
    """安全判断を独立したヘッドで行うSO8Tモデル"""
    
    def __init__(self, config: SafetyModelConfig):
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
        
        # タスク遂行系ヘッド（COMPLY/REFUSE/ESCALATE）
        self.task_classifier = nn.Linear(config.d_model, config.num_labels)
        
        # 安全判断系ヘッド（REFUSE/ESCALATE/ALLOW）
        self.safety_classifier = nn.Linear(config.d_model, config.num_safety_labels)
        
        # 安全判断用の独立したレイヤー（R_safe優先）
        self.safety_layers = nn.ModuleList(
            [
                SO8TBlock(config.d_model, config.n_heads, config.d_ff, config.dropout, config.gate_order)
                for _ in range(2)  # 安全判断専用の2層
            ]
        )
        self.safety_layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        positions = positions.clamp(max=self.position_embeddings.num_embeddings - 1)

        hidden_states = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        hidden_states = self.dropout(hidden_states)

        # 通常のレイヤー（タスク遂行用）
        pet_losses: List[torch.Tensor] = []
        for layer in self.layers:
            hidden_states, pet_loss = layer(hidden_states, attention_mask, safety_first=False)
            pet_losses.append(pet_loss)

        hidden_states = self.layer_norm(hidden_states)

        # タスク遂行系の予測
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        masked_hidden = hidden_states * mask
        summed = masked_hidden.sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / denom
        
        task_logits = self.task_classifier(pooled)

        # 安全判断系の予測（R_safe優先）
        safety_hidden = hidden_states.clone()
        safety_pet_losses: List[torch.Tensor] = []
        
        for layer in self.safety_layers:
            safety_hidden, safety_pet_loss = layer(safety_hidden, attention_mask, safety_first=True)
            safety_pet_losses.append(safety_pet_loss)
        
        safety_hidden = self.safety_layer_norm(safety_hidden)
        
        # 安全判断用のプーリング
        safety_masked = safety_hidden * mask
        safety_summed = safety_masked.sum(dim=1)
        safety_pooled = safety_summed / denom
        
        safety_logits = self.safety_classifier(safety_pooled)

        # PET損失（通常 + 安全判断）
        pet_total = torch.stack(pet_losses).mean()
        safety_pet_total = torch.stack(safety_pet_losses).mean()
        combined_pet_loss = pet_total + safety_pet_total

        return {
            "task_logits": task_logits,
            "safety_logits": safety_logits,
            "pet_loss": combined_pet_loss,
            "task_pet_loss": pet_total,
            "safety_pet_loss": safety_pet_total
        }

    def pet_loss(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.forward(input_ids, attention_mask)["pet_loss"]


def build_safety_model(config: SafetyModelConfig) -> SafetyAwareSO8T:
    return SafetyAwareSO8T(config)
