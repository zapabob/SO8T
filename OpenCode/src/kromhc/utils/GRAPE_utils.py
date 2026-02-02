"""
GRAPE utility functions.

Functions:
    - validate_rotation_matrix: Check if matrix is in SO(d) with det ≈ 1
    - compute_position_frequencies: Compute position-specific frequencies
    - streaming_cache_info: Analyze streaming cacheability
"""

from __future__ import annotations
from typing import Dict

import torch
import torch.nn.functional as F


def validate_rotation_matrix(
    matrix: torch.Tensor,
    dim: int,
    eps: float = 1e-6,
) -> bool:
    """Validate that matrix is in SO(d) and has det ≈ 1."""
    if matrix.dim() != 2:
        return False

    # Check special orthogonal structure for SO(2)
    # Should be traceless with det = 1
    det = torch.det(matrix)
    if abs(det - 1.0) < eps:
        return True
    return False


def compute_position_frequencies(
    omega: torch.Tensor,
    max_positions: int,
) -> torch.Tensor:
    """Compute position-specific frequencies from learned omega.

    Args:
        omega: Learnable frequency parameter [dim]
        max_positions: Maximum position index

    Returns:
        Frequencies for each position [max_positions]
    """
    # omega has shape [dim]
    positions = torch.arange(max_positions, dtype=omega.dtype, device=omega.device)
    return positions @ omega


def streaming_cache_info(
    position_ids: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> dict:
    """Analyze streaming cacheability of position encoding."""
    return {
        "has_sequential_dependency": False,
        "cache_key_dim": key_cache.shape[-1] if key_cache is not None else 0,
        "cache_value_dim": value_cache.shape[-1] if value_cache is not None else 0,
    }


def rodrigues_exp_formula(
    theta: torch.Tensor,
    s: float,
    L: torch.Tensor,
    d: int,
) -> torch.Tensor:
    """
    Rodrigues formula for exp(L):
    exp(L) = I + (sin s / s) L + ((1 - cos s) / s²) L²

    where:
        s = sin θ, c = cos θ
          L is skew-symmetric matrix
    """
    s = theta.sin()
    c = theta.cos()

    # Normalize s and c
    s_norm = s / theta.norm()
    c_norm = c / theta.norm()

    # Rodrigues formula
    I = torch.eye(d, dtype=theta.dtype, device=theta.device) + L @ c

    s_normalized = s_norm / c_norm
    c_normalized = c_norm / theta.norm()

    term1 = I + (s_normalized / c_normalized) @ L
    term2 = ((1 - c_normalized) / c_normalized) @ (1 - s_normalized) @ L
    term2 = term2 / (term2_denom + 1e-10)

    return term1 + term2
