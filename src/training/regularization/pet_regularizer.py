from __future__ import annotations

from typing import Dict, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math


@dataclass
class PETConfig:
    lambda_reg: float = 0.01
    position_decay: float = 0.1
    attention_scale: float = 1.0
    safety_gate_threshold: float = 0.7


class PETRegularizer(nn.Module):
    def __init__(self, model: nn.Module, config: Optional[PETConfig] = None):
        super().__init__()
        self.model = model
        self.config = config or PETConfig()
        self.original_embeddings: Dict[str, torch.Tensor] = {}
        self.original_attentions: Dict[str, torch.Tensor] = {}
        self.safety_scores: Dict[str, torch.Tensor] = {}

    def save_original_weights(self) -> None:
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Embedding):
                self.original_embeddings[name] = module.weight.data.clone()
            elif isinstance(module, nn.MultiheadAttention):
                self.original_attentions[name] = {
                    "in_proj_weight": module.in_proj_weight.data.clone(),
                    "out_proj_weight": module.out_proj.weight.data.clone(),
                }

    def compute_position_decay_loss(self, position_ids: torch.Tensor) -> torch.Tensor:
        decay_mask = torch.exp(-self.config.position_decay * position_ids.float())
        loss = 0.0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Embedding) and name in self.original_embeddings:
                diff = module.weight - self.original_embeddings[name]
                weighted_diff = diff * decay_mask[: diff.size(0), : diff.size(1)]
                loss = loss + 0.5 * (weighted_diff**2).sum()
        return self.config.lambda_reg * loss

    def compute_safety_preservation_loss(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = 0.0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                if name in self.original_attentions:
                    orig = self.original_attentions[name]
                    current_in = module.in_proj_weight
                    current_out = module.out_proj.weight
                    in_diff = (current_in - orig["in_proj_weight"]).pow(2).sum()
                    out_diff = (current_out - orig["out_proj_weight"]).pow(2).sum()
                    loss = loss + self.config.lambda_reg * (in_diff + out_diff)
        return self.config.lambda_reg * loss

    def compute_safety_score(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones_like(hidden_states[:, :, 0])
        extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        safety_scores = hidden_states * extended_mask
        safety_scores = safety_scores.mean(dim=(1, 2))
        safety_scores = torch.sigmoid(safety_scores * self.config.attention_scale)
        return safety_scores

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        position_loss = self.compute_position_decay_loss(position_ids)
        safety_loss = self.compute_safety_preservation_loss(
            hidden_states, attention_mask
        )
        safety_score = self.compute_safety_score(hidden_states, attention_mask)
        total_loss = position_loss + safety_loss
        return total_loss, {
            "position_loss": position_loss.item(),
            "safety_loss": safety_loss.item(),
            "safety_score": safety_score.mean().item(),
        }


class PETScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_lambda: float = 0.01,
        final_lambda: float = 0.001,
        warmup_steps: int = 100,
        total_steps: int = 1000,
    ):
        self.optimizer = optimizer
        self.initial_lambda = initial_lambda
        self.final_lambda = final_lambda
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0

    def step(self) -> float:
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lambda_reg = self.initial_lambda
        else:
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            lambda_reg = (
                self.initial_lambda * (1 - progress) + self.final_lambda * progress
            )
        return lambda_reg

    def get_current_lambda(self) -> float:
        if self.current_step < self.warmup_steps:
            return self.initial_lambda
        progress = (self.current_step - self.warmup_steps) / (
            self.total_steps - self.warmup_steps
        )
        return self.initial_lambda * (1 - progress) + self.final_lambda * progress
