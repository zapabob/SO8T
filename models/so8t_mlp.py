"""
SO8T MLP: SO(8)群構造を持つMLP層

This module implements MLP layers with SO(8) group structure,
replacing standard MLP with group-theoretic operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
import math


class SO8TMLP(nn.Module):
    """SO8T MLP with SO(8) group structure."""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        group_structure: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.group_structure = group_structure
        
        # MLP layers
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
        # SO8T group structure parameters
        if group_structure:
            self.rotation_dim = 8
            self.register_buffer("rotation_matrix", self._create_rotation_matrix())
            self.group_scale = nn.Parameter(torch.ones(1))
            
    def _create_rotation_matrix(self) -> torch.Tensor:
        """Create SO(8) rotation matrix for MLP."""
        # Create 8x8 rotation matrix
        rotation_matrix = torch.eye(8)
        
        # Add small random rotation for group structure
        angle = torch.randn(1) * 0.1
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        
        # 2D rotation in first two dimensions
        rotation_matrix[0, 0] = cos_angle
        rotation_matrix[0, 1] = -sin_angle
        rotation_matrix[1, 0] = sin_angle
        rotation_matrix[1, 1] = cos_angle
        
        return rotation_matrix
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SO8T MLP.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        # Apply SO8T group structure if enabled
        if self.group_structure:
            x = self._apply_group_structure(x)
            
        # Gate projection
        gate = self.gate_proj(x)
        
        # Up projection
        up = self.up_proj(x)
        
        # Apply activation
        gate = self._apply_activation(gate)
        
        # Element-wise multiplication
        intermediate = gate * up
        
        # Down projection
        output = self.down_proj(intermediate)
        
        return output
        
    def _apply_group_structure(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SO8T group structure to input."""
        # Get rotation matrix
        rotation_matrix = self.rotation_matrix
        
        # Apply rotation to the last dimension
        x_rotated = torch.matmul(x, rotation_matrix[:x.size(-1), :x.size(-1)])
        
        # Apply group scaling
        x_rotated = x_rotated * self.group_scale
        
        return x_rotated
        
    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation function."""
        if self.hidden_act == "silu":
            return F.silu(x)
        elif self.hidden_act == "gelu":
            return F.gelu(x)
        elif self.hidden_act == "relu":
            return F.relu(x)
        elif self.hidden_act == "swish":
            return x * torch.sigmoid(x)
        else:
            raise ValueError(f"Unsupported activation: {self.hidden_act}")


class SO8TGroupMLP(nn.Module):
    """SO8T Group MLP with multiple group structures."""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_groups: int = 4,
        hidden_act: str = "silu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_groups = num_groups
        self.hidden_act = hidden_act
        
        # Group-specific MLPs
        self.group_mlps = nn.ModuleList([
            SO8TMLP(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size // num_groups,
                hidden_act=hidden_act,
                group_structure=True
            )
            for _ in range(num_groups)
        ])
        
        # Group combination layer
        self.group_combine = nn.Linear(intermediate_size, intermediate_size, bias=False)
        
        # Final projection
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SO8T Group MLP.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        # Apply group-specific MLPs
        group_outputs = []
        for group_mlp in self.group_mlps:
            group_output = group_mlp(x)
            group_outputs.append(group_output)
            
        # Concatenate group outputs
        combined = torch.cat(group_outputs, dim=-1)
        
        # Apply group combination
        combined = self.group_combine(combined)
        
        # Apply activation
        combined = self._apply_activation(combined)
        
        # Final projection
        output = self.down_proj(combined)
        
        return output
        
    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation function."""
        if self.hidden_act == "silu":
            return F.silu(x)
        elif self.hidden_act == "gelu":
            return F.gelu(x)
        elif self.hidden_act == "relu":
            return F.relu(x)
        elif self.hidden_act == "swish":
            return x * torch.sigmoid(x)
        else:
            raise ValueError(f"Unsupported activation: {self.hidden_act}")


class SO8TAdaptiveMLP(nn.Module):
    """SO8T Adaptive MLP with dynamic group structure."""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        adaptive_groups: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.adaptive_groups = adaptive_groups
        
        # Base MLP
        self.base_mlp = SO8TMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            group_structure=True
        )
        
        # Adaptive group selection
        if adaptive_groups:
            self.group_selector = nn.Linear(hidden_size, 8, bias=False)  # 8 groups max
            self.group_weights = nn.Parameter(torch.ones(8))
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SO8T Adaptive MLP.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        # Base MLP forward pass
        base_output = self.base_mlp(x)
        
        # Adaptive group selection
        if self.adaptive_groups:
            # Get group selection weights
            group_logits = self.group_selector(x)
            group_weights = F.softmax(group_logits, dim=-1)
            
            # Apply group weights
            group_weights = group_weights * self.group_weights
            group_weights = group_weights / group_weights.sum(dim=-1, keepdim=True)
            
            # Apply adaptive weighting
            base_output = base_output * group_weights.unsqueeze(-1)
            
        return base_output
