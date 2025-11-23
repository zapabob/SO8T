"""
SO8T Embedding: SO(8)群構造を持つEmbedding層

This module implements embedding layers with SO(8) group structure,
replacing standard embeddings with group-theoretic operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
import math


class SO8TEmbedding(nn.Module):
    """SO8T Embedding with SO(8) group structure."""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        padding_idx: Optional[int] = None,
        group_structure: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.group_structure = group_structure
        
        # Token embedding
        self.token_embedding = nn.Embedding(
            vocab_size, 
            hidden_size, 
            padding_idx=padding_idx
        )
        
        # SO8T group structure parameters
        if group_structure:
            self.rotation_dim = 8
            self.register_buffer("rotation_matrix", self._create_rotation_matrix())
            self.group_scale = nn.Parameter(torch.ones(1))
            self.group_bias = nn.Parameter(torch.zeros(hidden_size))
            
    def _create_rotation_matrix(self) -> torch.Tensor:
        """Create SO(8) rotation matrix for embeddings."""
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
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SO8T Embedding.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            
        Returns:
            Embeddings [batch_size, seq_len, hidden_size]
        """
        # Get token embeddings
        embeddings = self.token_embedding(input_ids)
        
        # Apply SO8T group structure if enabled
        if self.group_structure:
            embeddings = self._apply_group_structure(embeddings)
            
        return embeddings
        
    def _apply_group_structure(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SO8T group structure to embeddings."""
        # Get rotation matrix
        rotation_matrix = self.rotation_matrix
        
        # Apply rotation to the last dimension
        x_rotated = torch.matmul(x, rotation_matrix[:x.size(-1), :x.size(-1)])
        
        # Apply group scaling and bias
        x_rotated = x_rotated * self.group_scale + self.group_bias
        
        return x_rotated


class SO8TPositionalEmbedding(nn.Module):
    """SO8T Positional Embedding with SO(8) group structure."""
    
    def __init__(
        self,
        hidden_size: int,
        max_position_embeddings: int = 32768,
        base: float = 10000.0,
        group_structure: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.group_structure = group_structure
        
        # Create frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, hidden_size, 2).float() / hidden_size))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # SO8T group structure parameters
        if group_structure:
            self.rotation_dim = 8
            self.register_buffer("rotation_matrix", self._create_rotation_matrix())
            self.group_scale = nn.Parameter(torch.ones(1))
            
    def _create_rotation_matrix(self) -> torch.Tensor:
        """Create SO(8) rotation matrix for positional embeddings."""
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
        
    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SO8T Positional Embedding.
        
        Args:
            position_ids: Position indices [batch_size, seq_len]
            
        Returns:
            Positional embeddings [batch_size, seq_len, hidden_size]
        """
        # Create position embeddings
        freqs = torch.outer(position_ids.float(), self.inv_freq)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        
        # Concatenate cos and sin
        pos_emb = torch.cat([cos, sin], dim=-1)
        
        # Apply SO8T group structure if enabled
        if self.group_structure:
            pos_emb = self._apply_group_structure(pos_emb)
            
        return pos_emb
        
    def _apply_group_structure(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SO8T group structure to positional embeddings."""
        # Get rotation matrix
        rotation_matrix = self.rotation_matrix
        
        # Apply rotation to the last dimension
        x_rotated = torch.matmul(x, rotation_matrix[:x.size(-1), :x.size(-1)])
        
        # Apply group scaling
        x_rotated = x_rotated * self.group_scale
        
        return x_rotated


class SO8TCombinedEmbedding(nn.Module):
    """SO8T Combined Embedding with token and positional embeddings."""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int = 32768,
        padding_idx: Optional[int] = None,
        group_structure: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.padding_idx = padding_idx
        self.group_structure = group_structure
        
        # Token embedding
        self.token_embedding = SO8TEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            padding_idx=padding_idx,
            group_structure=group_structure
        )
        
        # Positional embedding
        self.position_embedding = SO8TPositionalEmbedding(
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            group_structure=group_structure
        )
        
        # Combination layer
        self.combine = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for SO8T Combined Embedding.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            position_ids: Position indices [batch_size, seq_len]
            
        Returns:
            Combined embeddings [batch_size, seq_len, hidden_size]
        """
        # Get token embeddings
        token_emb = self.token_embedding(input_ids)
        
        # Get position embeddings
        if position_ids is None:
            position_ids = torch.arange(
                input_ids.size(1), 
                device=input_ids.device
            ).unsqueeze(0).expand(input_ids.size(0), -1)
            
        pos_emb = self.position_embedding(position_ids)
        
        # Combine token and position embeddings
        combined = torch.cat([token_emb, pos_emb], dim=-1)
        combined = self.combine(combined)
        
        return combined


class SO8TAdaptiveEmbedding(nn.Module):
    """SO8T Adaptive Embedding with dynamic group structure."""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int = 32768,
        padding_idx: Optional[int] = None,
        adaptive_groups: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.padding_idx = padding_idx
        self.adaptive_groups = adaptive_groups
        
        # Base embedding
        self.base_embedding = SO8TEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            padding_idx=padding_idx,
            group_structure=True
        )
        
        # Adaptive group selection
        if adaptive_groups:
            self.group_selector = nn.Linear(hidden_size, 8, bias=False)  # 8 groups max
            self.group_weights = nn.Parameter(torch.ones(8))
            
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SO8T Adaptive Embedding.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            
        Returns:
            Adaptive embeddings [batch_size, seq_len, hidden_size]
        """
        # Base embedding forward pass
        base_emb = self.base_embedding(input_ids)
        
        # Adaptive group selection
        if self.adaptive_groups:
            # Get group selection weights
            group_logits = self.group_selector(base_emb)
            group_weights = F.softmax(group_logits, dim=-1)
            
            # Apply group weights
            group_weights = group_weights * self.group_weights
            group_weights = group_weights / group_weights.sum(dim=-1, keepdim=True)
            
            # Apply adaptive weighting
            base_emb = base_emb * group_weights.unsqueeze(-1)
            
        return base_emb
