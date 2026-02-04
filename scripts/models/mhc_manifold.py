"""mHC manifold utilities (Birkhoff projection helpers)."""
from __future__ import annotations

from typing import Iterable, List, Optional

import torch


def birkhoff_project(weight: torch.Tensor, max_iter: int = 20, epsilon: float = 1e-8) -> torch.Tensor:
    """Project a weight matrix onto a doubly-stochastic-like manifold.

    Uses Sinkhorn-Knopp normalization on the absolute values and re-applies the sign.
    """
    if weight.ndim != 2:
        return weight
    dtype = weight.dtype
    device = weight.device
    w = weight.detach().float().abs().clamp_min(epsilon)

    for _ in range(max_iter):
        w = w / (w.sum(dim=1, keepdim=True) + epsilon)
        w = w / (w.sum(dim=0, keepdim=True) + epsilon)

    projected = w * weight.detach().sign().float()
    return projected.to(device=device, dtype=dtype)


def apply_mhc_projection_to_model(
    model,
    target_modules: Optional[Iterable[str]] = None,
    max_iter: int = 20,
    epsilon: float = 1e-8,
    blend: float = 0.1,
) -> List[str]:
    """Apply Birkhoff projection to selected linear weights.

    Returns list of module names updated.
    """
    if target_modules is None:
        target_modules = [
            "o_proj",
            "out_proj",
            "down_proj",
            "up_proj",
            "gate_proj",
        ]

    updated: List[str] = []
    for name, module in model.named_modules():
        if not hasattr(module, "weight"):
            continue
        if not any(key in name for key in target_modules):
            continue
        weight = module.weight.data
        if weight.ndim != 2:
            continue
        projected = birkhoff_project(weight, max_iter=max_iter, epsilon=epsilon)
        module.weight.data = (1.0 - blend) * weight + blend * projected
        updated.append(name)

    return updated
