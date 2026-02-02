"""
PET (Positional Energy Tapering) regularizer for stabilising training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn


@dataclass
class PETSchedule:
    """
    Three phase PET strength schedule expressed as total training progress ratios.
    """

    phase_boundaries: Tuple[float, float] = (0.3, 0.7)
    lambdas: Tuple[float, float, float] = (0.05, 0.2, 0.5)

    def strength(self, progress: float) -> float:
        low, high = self.phase_boundaries
        start, mid, end = self.lambdas
        if progress < low:
            return start
        if progress < high:
            return mid
        return end


class PETRegularizer(nn.Module):
    """
    Second order discrete Laplacian PET regularizer.
    """

    def __init__(self, schedule: PETSchedule | None = None) -> None:
        super().__init__()
        self.schedule = schedule or PETSchedule()

    def forward(self, hidden_states: Tensor, progress: float) -> Tensor:
        """
        Args:
            hidden_states: Tensor[B, T, D]
            progress: float in [0, 1] representing training progress.
        """
        if hidden_states.size(1) < 3:
            return hidden_states.new_zeros(())

        lap = (
            hidden_states[:, 2:, :]
            - 2 * hidden_states[:, 1:-1, :]
            + hidden_states[:, :-2, :]
        )
        penalty = lap.pow(2).mean()
        return penalty * self.schedule.strength(progress)
