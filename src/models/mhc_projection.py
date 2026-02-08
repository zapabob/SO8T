import torch
import torch.nn as nn

@torch.no_grad()
def project_mhc_l2(model: nn.Module, max_norm: float = 1.0):
    """
    Minimal Manifold-Constrained Hyper-Connections (mHC) inspired projection.
    Applies L2 norm constraint to SO8T adapter gate coefficients.
    """
    if hasattr(model, "so8t_adapter_bank"):
        alpha = model.so8t_adapter_bank.alpha
        norm = torch.norm(alpha)
        if norm > max_norm:
            alpha.mul_(max_norm / (norm + 1e-6))

@torch.no_grad()
def project_mhc_clip(model: nn.Module, min_val: float = -1.0, max_val: float = 1.0):
    """
    Box-constrained manifold projection.
    Clips gate coefficients to a stable range.
    """
    if hasattr(model, "so8t_adapter_bank"):
        model.so8t_adapter_bank.alpha.clamp_(min=min_val, max=max_val)
