import torch
import torch.nn as nn
from typing import Optional

from src.models.projector import SO8ViTProjector

class VisionEncoderWrapper(nn.Module):
    """
    Wrapper for a frozen ViT with a trainable projector.
    """
    def __init__(self, vit_model: nn.Module, llm_hidden_size: int):
        super().__init__()
        self.vit = vit_model
        # Freeze ViT weights by default
        for param in self.vit.parameters():
            param.requires_grad = False
            
        self.projector = SO8ViTProjector(
            vision_hidden_size=getattr(vit_model.config, "hidden_size", 1024),
            llm_hidden_size=llm_hidden_size
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Get patch embeddings/features from ViT
        vision_outputs = self.vit(pixel_values).last_hidden_state
        # Project to LLM space
        return self.projector(vision_outputs)
