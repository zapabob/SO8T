"""
DGPO/GRPO trainer scaffolding.
References:
    Dai et al. (2026) arXiv:2601.20614
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .reward import ShapedGRPOReward, RewardConfig
from ...utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""

    group_size: int = 8
    learning_rate: float = 1e-4
    entropy_coef: float = 0.01
    clip_epsilon: float = 0.2


class GRPOTrainer:
    """Minimal GRPO trainer scaffold."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Optional[GRPOConfig] = None,
        reward_config: Optional[RewardConfig] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config or GRPOConfig()
        self.reward = ShapedGRPOReward(reward_config)

    def train_step(self, batch: dict) -> dict:
        """Run a single GRPO training step.
        This is a scaffold; caller supplies batch with logits and reward fields.
        """
        self.model.train()
        logits = batch["logits"]
        advantages = batch["advantages"]
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -(log_probs * advantages).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        logger.debug("GRPO step loss: %s", loss.item())
        return {"loss": loss.item()}
