"""Difficulty estimation utilities for DGPO."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DifficultyEstimatorConfig:
    """Configuration for difficulty estimation."""

    scale: float = 1.0
    offset: float = 0.0


class DifficultyEstimator:
    """Simple difficulty estimator based on loss or score."""

    def __init__(self, config: Optional[DifficultyEstimatorConfig] = None) -> None:
        self.config = config or DifficultyEstimatorConfig()

    def estimate(self, scores: torch.Tensor) -> torch.Tensor:
        """Estimate difficulty (higher means harder)."""
        return torch.clamp((1.0 - scores) * self.config.scale + self.config.offset, 0.0, 1.0)
