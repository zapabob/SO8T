"""
GRAPE (Group Representational Position Encoding)
References:
    Zhang et al. (2026) arXiv:2512.07805, ICLR 2026
    Apache 2.0 License

This module implements GRAPE, a unified framework for positional encoding
based on group actions. GRAPE brings together two families of mechanisms:
(i) Multiplicative rotations (Multiplicative GRAPE) in SO(d)
(ii) Additive logit biases (Additive GRAPE) in GL(d+k)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.errors import KromHCError
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GRAPEConfig:
    """GRAPE configuration."""
    # Group type: 'so3' for multiplicative SO(d), 'gl' for additive
    group_type: Literal["so3", "gl"] = "so3"
    
    # Dimension for SO(d) or SO(3)
    dim: int = 2
    
    # Learning rate for frequency parameters
    freq_lr: float = 1e-4
    
    # Maximum position index
    max_positions: int = 8192
    
    # Learnable frequency
    learnable_freq: bool = False
    
    # Skew factor for generator initialization
    skew_factor: float = 0.1
    
    # Use bias term (for additive variant)
    use_bias: bool = False


class GRAPEMultiplicative(nn.Module):
    """Multiplicative GRAPE: SO(d) rotations with closed-form matrix exponential.
    
    Key formula: exp(L) = I + (sin s / s) L + ((1 - cos s) / s²) L²
    
    where:
        s = sin θ, c = cos θ
        L is a skew-symmetric generator
    """
    
    def __init__(self, config: GRAPEConfig):
        super().__init__()
        self.dim = config.dim
        self.max_pos = config.max_positions
        self.learnable_freq = config.learnable_freq
        self.use_bias = config.use_bias
        
        # Position indices
        self.register_buffer(
            'pos_indices',
            torch.arange(config.max_positions, dtype=torch.float32)
        )
        
        # Learnable frequency parameters (skew-symmetric generator L)
        # Initialize from normal distribution
        L = torch.randn(self.dim, dtype=torch.float32) * config.skew_factor
        
        if config.learnable_freq:
            self.L = nn.Parameter(L)
        else:
            self.register_buffer('L', L)
        
        # Bias for additive variant
        self.bias = nn.Parameter(torch.zeros(self.dim)) if config.use_bias else None
        
        logger.info(f"GRAPE init: dim={config.dim}, max_pos={config.max_pos}, "
                    f"learnable_freq={config.learnable_freq}")
    
    def _rodrigues_formula(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Closed-form Rodriguez-type formula for exp(L):
        exp(L) = I + (sin s / s) L + ((1 - cos s) / s²) L²
        
        where:
            s = sin θ, c = cos θ
        """
        s = theta.sin()
        c = theta.cos()
        
        # L is in R^{d×d} with skew-symmetric structure
        if self.dim == 2:
            # SO(2) skew-symmetric: L = [[0, -a, b], [a, 0, -c], [-b, -c, 0]]
            L_traceless = torch.tensor([
                [[0, -L[0,1]], L[0,2]],
                [L[0,1], 0, [-L[0,1], 0],
                [0, 0, -L[1,2]], [-L[0,1], 0]
            ], dtype=theta.dtype, device=theta.device)
        elif self.dim == 3:
            # SO(3): L has special structure (antisymmetric traceless part)
            L_traceless = torch.tensor([
                [[0, -L[0,1], 0],
                [0, L[0,2], 0],
                [0, 0, -L[0,1], 0],
                [0, 0, -L[1,2], 0]
            ], dtype=theta.dtype, device=theta.device)
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}")
        
        # Rodrigues formula
        I = torch.eye(self.dim, dtype=theta.dtype, device=theta.device) + L_traceless
        
        s_normalized = s / theta.norm()
        c_normalized = c / theta.norm()
        
        term1 = I + (s_normalized / c_normalized) @ self.L
        term2 = ((1 - c_normalized) / c_normalized) @ self.L @ self.L
        
        return term1 + term2
    
    def _get_rotation_matrices(self, n: torch.Tensor) -> torch.Tensor:
        """Compute rotation matrix G(n) for each position n."""
        # Theta = n * ω L where ω is frequency parameter
        theta = n[:, None] * self.L
        
        # Compute exp(L) using closed-form formula
        exp_L = self._rodrigues_formula(theta)
        
        return exp_L
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embeddings [batch, seq_len, dim]
        
        Returns:
            Position-encoded embeddings [batch, seq_len, dim]
        """
        seq_len = x.shape[1]
        positions = self.pos_indices[:seq_len]
        
        # Compute rotation matrices G(n) for all positions
        # Shape: [seq_len, dim]
        rotations = self._get_rotation_matrices(positions)
        
        # Apply G(n) to embeddings: x' = G(n) * x
        # Broadcasting: [batch, seq_len, dim] @ [seq_len, dim]
        # Result: [batch, seq_len, dim]
        pos_encoded = torch.einsum('djk', x, rotations)
        
        # Add bias if enabled
        if self.bias is not None:
            pos_encoded = pos_encoded + self.bias
        
        return pos_encoded


class GRAPEAdditive(nn.Module):
    """Additive GRAPE: GL(d+k) unipotent actions (ALiBi/FoX variant).
    
    def __init__(self, config: GRAPEConfig, dim: int):
        super().__init__()
        self.dim = dim
        self.max_pos = config.max_positions
        self.use_bias = config.use_bias
        
        # Unipotent generator A (A² = 0 in GL(d+k))
        # Project to GL(d+k): embed in homogeneous space, then apply A, then project back
        
        # Learnable generator A (nilpotent in GL(d+k))
        A_init = torch.randn(dim, dim, dtype=torch.float32) * 0.01
        
        if config.learnable_freq:
            self.A = nn.Parameter(A_init)
        else:
            self.register_buffer('A', A_init)
        
        self.bias = nn.Parameter(torch.zeros(dim)) if config.use_bias else None
        
        # Projection matrices for homogeneous coordinates
        self.register_buffer('I', torch.eye(dim + 1, dtype=torch.float32))
        self.register_buffer('P', torch.eye(dim, dtype=torch.float32))
        
        logger.info(f"GRAPE Additive init: dim={dim}, use_bias={config.use_bias}")
    
    def _apply_unipotent(self, x: torch.Tensor) -> torch.Tensor:
        """Apply unipotent action: x' = A @ x (homogeneous space)."""
        # Project to homogeneous: [batch, dim] -> [batch, dim+1, 1]
        x_hom = F.pad(x, (0, 0, 1), value=1.0)
        
        # Apply A: [dim+1, dim] @ [dim+1, 1] -> [dim+1, dim]
        x_prime_hom = torch.matmul(self.A, x_hom)
        
        # Project back: [dim+1, dim] -> [dim, dim] 
        x_prime = torch.matmul(self.P, x_prime_hom)
        
        return x_prime
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply additive position encoding."""
        seq_len = x.shape[1]
        
        # Create position indices
        positions = torch.arange(seq_len, dtype=torch.float32, device=x.device)
        
        # Compute position bias term: G(n) = exp(n ω A)
        # For additive variant, we compute: bias(n) = n ω A
        
        # Since A is nilpotent: exp(n ω A) = I + n @ A (first-order Taylor)
        # = I + n @ A (second-order approximation)
        
        n_matrix = positions.view(-1, 1)  # [seq_len, 1, 1]
        exp_term = torch.matrix_exp(n_matrix @ self.A)  # [seq_len, 1, dim]
        
        bias = exp_term
        
        return x + bias


class GRAPEHybrid(nn.Module):
    """Hybrid GRAPE: Multiplicative GRAPE with bias term (for flexibility)."""
    
    def __init__(
        self,
        multiplicative_config: GRAPEConfig,
        additive_config: Optional[GRAPEConfig] = None,
    ):
        super().__init__()
        self.multiplicative = GRAPEMultiplicative(multiplicative_config)
        if additive_config is not None:
            self.additive = GRAPEAdditive(additive_config, dim=multiplicative_config.dim)
        else:
            self.additive = None
        logger.info("GRAPEHybrid: Multiplicative GRAPE with optional additive term")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos_encoded = self.multiplicative(x)
        if self.additive is not None:
            pos_encoded = self.additive(x)
            pos_encoded = pos_encoded + self.additive.bias
        return pos_encoded
        return pos_encoded


def validate_rotation_matrix(
    matrix: torch.Tensor,
    dim: int,
    eps: float = 1e-6,
) -> bool:
    """Validate that matrix is in SO(d) and has det ≈ 1."""
    if matrix.dim() != 2:
        return False
    
    # Check special orthogonal structure
    if dim == 2:
        # Should be traceless with det = a² - (-a²) = 1
        det = torch.det(matrix)
        if abs(det - 1.0) < eps:
            return True
        return False
    
    def compute_position_frequencies(
    omega: torch.Tensor,
    max_positions: int,
) -> torch.Tensor:
    """Compute position-specific frequencies from learned omega.
    
    Args:
        omega: Learnable frequency parameter [dim]
        max_positions: Maximum position index
    
    Returns:
        Frequencies for each position [max_positions]
    """
    # omega has shape [dim]
    # For multiplicative GRAPE, we use: θ_n = n @ ω
    positions = torch.arange(max_positions, dtype=omega.dtype, device=omega.device)
    return positions @ omega


def streaming_cache_info(
    position_ids: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> dict:
    """Analyze streaming cacheability of position encoding."""
    return {
        'has_sequential_dependency': False,
        'cache_key_dim': key_cache.shape[-1] if key_cache is not None else 0,
        'cache_value_dim': value_cache.shape[-1] if value_cache is not None else 0,
    }
