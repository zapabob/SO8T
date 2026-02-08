import torch
import torch.nn as nn

def pet_loss(alpha: torch.Tensor, lam_p: float = 1e-3, lam_d: float = 0.0) -> torch.Tensor:
    """
    Second-Order Discrete Difference Regularization for SO8T Adapters.
    
    Args:
        alpha: Gate coefficients tensor of shape [num_layers, num_passes]
        lam_p: Lambda for pass-direction smoothness (mandatory)
        lam_d: Lambda for layer-direction smoothness (optional)
        
    Returns:
        torch.Tensor: Scalar PET loss
    """
    # alpha shape: [L, 4]
    num_layers, num_passes = alpha.shape
    loss = torch.tensor(0.0, device=alpha.device)
    
    # Pass-direction second-order difference (Smoothness over inference time)
    if num_passes >= 3:
        # Delta^2 alpha_p = alpha_{p+1} - 2*alpha_p + alpha_{p-1}
        # For P=4, central p are 1, 2 (index 1, 2)
        d2_p1 = alpha[:, 2] - 2 * alpha[:, 1] + alpha[:, 0]
        d2_p2 = alpha[:, 3] - 2 * alpha[:, 2] + alpha[:, 1]
        loss = loss + lam_p * (d2_p1.pow(2).mean() + d2_p2.pow(2).mean())
        
    # Layer-direction second-order difference (Smoothness over depth)
    if lam_d > 0 and num_layers >= 3:
        # Delta^2 alpha_l = alpha_{l+1} - 2*alpha_l + alpha_{l-1}
        d2_l = alpha[2:, :] - 2 * alpha[1:-1, :] + alpha[:-2, :]
        loss = loss + lam_d * d2_l.pow(2).mean()
        
    return loss
