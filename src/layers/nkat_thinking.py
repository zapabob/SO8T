import torch
import torch.nn as nn
import torch.nn.functional as F

class NKAT_ThinkingBlock(nn.Module):
    """
    NKAT Core Block: Replaces the standard FFN in Transformers.

    - Geometry: SO(8) Rotation
    - Physics: Heat Kernel Decay
    - Activation: Squared ReLU (Energy)
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()

        assert dim % 8 == 0, f"Dim {dim} must be divisible by 8"
        self.dim = dim
        self.num_blocks = dim // 8

        # SO(8) Parameters
        self.so8_raw = nn.Parameter(0.01 * torch.randn(self.num_blocks, 8, 8))
        # Heat Kernel Decay
        self.decay_params = nn.Parameter(torch.zeros(self.num_blocks, 8))

        self.mixing = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Seq, Dim) or (Seq, Batch, Dim)
        residual = x
        original_shape = x.shape

        # Flatten to (N, Dim) for processing
        x_flat = x.view(-1, self.dim)
        B_total = x_flat.shape[0]

        # 1. Geometry
        x_blocks = x_flat.view(B_total, self.num_blocks, 8)
        A = 0.5 * (self.so8_raw - self.so8_raw.transpose(1, 2))
        R = torch.matrix_exp(A) # (Blocks, 8, 8)
        x_rot = torch.einsum('bki,kji->bkj', x_blocks, R)

        # 2. Physics
        gamma = torch.sigmoid(self.decay_params)
        x_phys = x_rot * gamma

        # 3. Mixing & Energy Activation
        out = x_phys.view(B_total, self.dim)
        out = self.mixing(out)
        out = F.relu(out) ** 2 # Squared ReLU
        out = self.dropout(out)

        # Reshape back
        out = out.view(original_shape)
        return self.norm(residual + out)
