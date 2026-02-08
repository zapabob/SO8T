import torch
import torch.nn as nn
from typing import Optional

class PETAdapter(nn.Module):
    """
    Minimal residual adapter for a single layer and pass.
    Includes mHC-inspired manifold scaling for stability.
    """
    def __init__(self, hidden_size: int, rank: int = 16):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, rank, bias=False)
        self.up_proj = nn.Linear(rank, hidden_size, bias=False)
        self.layernorm = nn.LayerNorm(hidden_size)
        
        # Manifold Scale: initialized to 0 to preserve identity initially
        self.manifold_scale = nn.Parameter(torch.zeros(1))
        
        # Initialize with zeros to start as identity residual
        nn.init.zeros_(self.up_proj.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Manifold-constrained scaling: tanh maintains scale in [-1, 1]
        scale = torch.tanh(self.manifold_scale)
        adapter_out = self.up_proj(self.down_proj(self.layernorm(x)))
        return scale * adapter_out

class PETAdapterBank(nn.Module):
    """
    Adapter Bank for PET (Second-Order Discrete Difference) SO8T Quadruple Inference.
    Contains (layer x pass) adapters and gate coefficients alpha.
    """
    def __init__(self, num_layers: int, hidden_size: int, num_passes: int = 4, rank: int = 16):
        super().__init__()
        self.num_layers = num_layers
        self.num_passes = num_passes
        
        # Adapter Bank: ModuleDict of ModuleLists
        # Format: adapters.layer_idx.pass_idx
        self.adapters = nn.ModuleList([
            nn.ModuleList([PETAdapter(hidden_size, rank) for _ in range(num_passes)])
            for _ in range(num_layers)
        ])
        
        # Gate coefficients alpha: (num_layers, num_passes)
        # Initialized to 0.0
        self.alpha = nn.Parameter(torch.zeros(num_layers, num_passes))
        
    def get_adapter_output(self, hidden_states: torch.Tensor, layer_idx: int, pass_id: int) -> torch.Tensor:
        """
        Calculates alpha_{l,p} * Adapter_{l,p}(LN(x))
        """
        if layer_idx >= self.num_layers or pass_id >= self.num_passes:
            return torch.zeros_like(hidden_states)
            
        adapter = self.adapters[layer_idx][pass_id]
        alpha = self.alpha[layer_idx, pass_id]
        
        return alpha * adapter(hidden_states)

    def calculate_pet_loss(self, lambda_g: float = 0.01) -> torch.Tensor:
        """
        L_PET = λg Σ_ℓ Σ_{p=2..3} (α_{ℓ,p+1}-2α_{ℓ,p}+α_{ℓ,p-1})^2
        """
        if self.num_passes < 3:
            return torch.tensor(0.0, device=self.alpha.device)
            
        # alpha shape: (num_layers, num_passes)
        # Second-order difference across pass dimension (dim=1)
        # Difference at p: alpha[p+1] - 2*alpha[p] + alpha[p-1]
        
        # For p=2,3 (1-indexed) -> indices 1, 2 (0-indexed)
        # We need p-1, p, p+1 to be valid.
        # If num_passes = 4, p can be 2, 3.
        # p=2: indices (0, 1, 2)
        # p=3: indices (1, 2, 3)
        
        diff = self.alpha[:, 2:] - 2 * self.alpha[:, 1:-1] + self.alpha[:, :-2]
        pet_loss = torch.sum(diff**2)
        
        return lambda_g * pet_loss
