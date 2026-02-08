import torch
import torch.nn as nn
from typing import Optional

class SO8ViTProjector(nn.Module):
    """
    Projector connecting Vision Features to LLM Space.
    Enhanced to support quadrality thinking paths.
    """
    def __init__(self, vision_hidden_size: int, llm_hidden_size: int):
        super().__init__()
        # Multi-stage MLP for better feature alignment
        self.projector = nn.Sequential(
            nn.Linear(vision_hidden_size, llm_hidden_size),
            nn.LayerNorm(llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )
        
        # Quadrant-specific conditioning layers (Conceptual)
        # These could bias the vision features based on the current thinking pass
        self.pass_gates = nn.Parameter(torch.ones(4, llm_hidden_size))

    def forward(self, vision_outputs: torch.Tensor, pass_id: int = 0) -> torch.Tensor:
        """
        vision_outputs: [batch, seq, vision_hidden]
        pass_id: Current SO8T pass (0-3)
        """
        x = self.projector(vision_outputs)
        
        # Apply pass-specific gating/conditioning
        gate = self.pass_gates[pass_id].view(1, 1, -1)
        return x * gate
