import sys
import os

# Add project root to path for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from src.layers.nkat_thinking import NKAT_ThinkingBlock

class NKATPendulumModel(nn.Module):
    """
    Physics-aware model using NKAT Thinking Blocks.
    """
    def __init__(self, in_dim: int = 40, model_dim: int = 64, out_dim: int = 4, num_layers: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, model_dim)

        self.blocks = nn.ModuleList([
            NKAT_ThinkingBlock(model_dim) for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(model_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)





class BaselineMLP(nn.Module):
    """
    Standard MLP Baseline (Brute-force approach).
    """
    def __init__(self, in_dim: int = 40, hidden_dim: int = 64, out_dim: int = 4, num_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]

        for _ in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ])

        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    print("Testing NKAT-SO8T Pendulum Models...")
    print("NKAT Thinking Block imported successfully!")
    print("Models created successfully!")

    # Test model creation
    nkat_model = NKATPendulumModel()
    baseline_model = BaselineMLP()

    print(f"NKAT Model: {nkat_model}")
    print(f"Baseline Model: {baseline_model}")
    print("All imports and model creation successful!")
