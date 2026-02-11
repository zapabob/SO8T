"""DGPO trainer wrapper."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..core.grpo import GRPOConfig, GRPOTrainer
from ..core.reward import RewardConfig


class DGRPOTrainer(GRPOTrainer):
    """DGPO trainer specialization."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Optional[GRPOConfig] = None,
        reward_config: Optional[RewardConfig] = None,
    ) -> None:
        super().__init__(model, optimizer, config=config, reward_config=reward_config)
