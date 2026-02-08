import torch
import torch.nn as nn
from typing import Optional

class ResidualAdapter(nn.Module):
    """
    Sub-module: down -> GELU -> up.
    Inserted into MLP residual connection.
    """
    def __init__(self, d_model: int, r: int = 16):
        super().__init__()
        self.down_proj = nn.Linear(d_model, r, bias=False)
        self.up_proj = nn.Linear(r, d_model, bias=False)
        self.activation = nn.GELU()
        
        # Initialize up_proj weights to zero to ensure identity residual at start
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up_proj(self.activation(self.down_proj(x)))

class SO8TAdapterBank(nn.Module):
    """
    Bank of L x P adapters with learnable gate coefficients alpha.
    """
    def __init__(self, num_layers: int, d_model: int, r: int = 16, num_passes: int = 4):
        super().__init__()
        self.num_layers = num_layers
        self.num_passes = num_passes
        
        # Adapters per layer and per pass
        self.adapters = nn.ModuleList([
            nn.ModuleList([ResidualAdapter(d_model, r) for _ in range(num_passes)])
            for _ in range(num_layers)
        ])
        
        # Trainable alpha parameters [L, P]
        self.alpha = nn.Parameter(torch.zeros(num_layers, num_passes))
        
    def forward(self, x: torch.Tensor, layer_idx: int, pass_id: int) -> torch.Tensor:
        """
        x_new = x + alpha[layer, pass] * adapter[layer][pass](LN(x))
        (LayerNorm should be applied before calling this if following Pre-LN)
        """
        # pass_id assumed to be 0-indexed here
        return self.alpha[layer_idx, pass_id] * self.adapters[layer_idx][pass_id](x)
