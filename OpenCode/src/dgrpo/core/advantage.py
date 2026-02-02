"""Advantage computation for DGPO."""
from __future__ import annotations

import torch


def compute_group_advantage(
    rewards: torch.Tensor,
    difficulties: torch.Tensor,
    tool_usage_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute difficulty-aware group advantage."""
    difficulty_weight = torch.sigmoid(difficulties - 0.5)
    group_mean = rewards.mean(dim=-1, keepdim=True)
    group_std = rewards.std(dim=-1, keepdim=True) + 1e-8
    normalized_rewards = (rewards - group_mean) / group_std
    tool_penalty = tool_usage_mask.float() * 0.5
    return normalized_rewards * difficulty_weight * (1.0 - tool_penalty)
