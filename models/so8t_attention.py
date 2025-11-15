"""
SO8T Attention: SO(8)群構造を持つAttentionメカニズム

This module implements attention mechanisms with SO(8) group structure,
replacing standard attention with group-theoretic operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math
import warnings

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    warnings.warn("Flash Attention not available. Using standard attention.")

# Import EfficientAttention as fallback
try:
    import sys
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).parent.parent
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "utils"))
    from efficient_attention import EfficientAttention, efficient_attention_func
    EFFICIENT_ATTN_AVAILABLE = True
except ImportError:
    EFFICIENT_ATTN_AVAILABLE = False
    EfficientAttention = None
    efficient_attention_func = None


class SO8TRotaryEmbedding(nn.Module):
    """SO8T Rotary Position Embedding with group structure."""
    
    def __init__(self, dim: int, max_position_embeddings: int = 32768, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Create rotation matrices for SO(8) group
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # SO(8) rotation parameters
        self.rotation_dim = 8
        self.register_buffer("rotation_matrix", self._create_rotation_matrix())
        
    def _create_rotation_matrix(self) -> torch.Tensor:
        """Create SO(8) rotation matrix for position embeddings."""
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
        
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply SO8T rotary position embedding.
        
        Args:
            x: Input tensor [batch_size, num_heads, seq_len, head_dim]
            seq_len: Sequence length
            
        Returns:
            Tuple of (cos, sin) for rotary embedding
        """
        device = x.device
        dtype = x.dtype
        
        # Create position indices
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        
        # Apply SO(8) rotation
        freqs = freqs.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim//2]
        
        # Create cos and sin
        cos = torch.cos(freqs).to(dtype)
        sin = torch.sin(freqs).to(dtype)
        
        # Apply group rotation
        cos = torch.matmul(cos, self.rotation_matrix[:cos.size(-1), :cos.size(-1)])
        sin = torch.matmul(sin, self.rotation_matrix[:sin.size(-1), :sin.size(-1)])
        
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SO8TAttention(nn.Module):
    """SO8T Attention with SO(8) group structure."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        attention_dropout: float = 0.0,
        bias: bool = False,
        use_flash_attention: bool = False,
        use_efficient_attention: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.attention_dropout = attention_dropout
        self.bias = bias
        self.use_flash_attention = use_flash_attention and FLASH_ATTN_AVAILABLE
        self.use_efficient_attention = use_efficient_attention and EFFICIENT_ATTN_AVAILABLE and not self.use_flash_attention
        
        # Initialize EfficientAttention if available
        if self.use_efficient_attention and EfficientAttention is not None:
            self.head_dim = hidden_size // num_heads
            self.efficient_attn = EfficientAttention(
                head_dim=self.head_dim,
                dropout=attention_dropout,
                chunk_size=512,
                use_tiling=True,
            )
        else:
            self.efficient_attn = None
        
        # Calculate head dimensions
        self.head_dim = hidden_size // num_heads
        self.num_queries_per_kv = num_heads // num_key_value_heads
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=bias)
        
        # SO8T Rotary Position Embedding
        self.rotary_emb = SO8TRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=32768,
            base=10000.0
        )
        
        # Attention dropout
        self.attn_dropout = nn.Dropout(attention_dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass for SO8T Attention.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len, seq_len]
            position_ids: Position indices [batch_size, seq_len]
            past_key_value: Past key-value cache
            output_attentions: Whether to output attention weights
            use_cache: Whether to use key-value cache
            cache_position: Cache position indices
            
        Returns:
            Tuple of (hidden_states, attention_weights, present_key_value)
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary position embedding
        cos, sin = self.rotary_emb(query_states, seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Repeat key and value for grouped query attention
        if self.num_queries_per_kv > 1:
            key_states = key_states.repeat_interleave(self.num_queries_per_kv, dim=1)
            value_states = value_states.repeat_interleave(self.num_queries_per_kv, dim=1)
            
        # Handle past key-value cache
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
            
        # Apply SO8T group structure to attention
        query_states, key_states, value_states = self._apply_group_structure(
            query_states, key_states, value_states
        )
        
        # Compute attention
        if self.use_flash_attention:
            attn_output = self._flash_attention(query_states, key_states, value_states, attention_mask)
            attn_weights = None
        elif self.use_efficient_attention and self.efficient_attn is not None:
            # Use EfficientAttention
            attn_output = self.efficient_attn(
                query_states,
                key_states,
                value_states,
                attention_mask=attention_mask,
                causal=True,
                softmax_scale=1.0 / math.sqrt(self.head_dim),
            )
            attn_weights = None  # EfficientAttention doesn't return attention weights
        else:
            attn_output, attn_weights = self._standard_attention(
                query_states, key_states, value_states, attention_mask
            )
            
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        # Prepare outputs
        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += ((key_states, value_states),)
            
        return outputs
        
    def _apply_group_structure(
        self, 
        query_states: torch.Tensor, 
        key_states: torch.Tensor, 
        value_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply SO8T group structure to attention states."""
        # Apply SO(8) rotation to query states
        query_states = self._apply_so8_rotation(query_states)
        
        # Apply SO(8) rotation to key states
        key_states = self._apply_so8_rotation(key_states)
        
        # Apply SO(8) rotation to value states
        value_states = self._apply_so8_rotation(value_states)
        
        return query_states, key_states, value_states
        
    def _apply_so8_rotation(self, x: torch.Tensor, rotation_type: str = "vector") -> torch.Tensor:
        """Apply SO8 rotation to input tensor with Triality symmetry."""
        # Create different rotation matrices based on Triality symmetry
        if rotation_type == "vector":
            # Vector representation (Task reasoning)
            rotation_matrix = self._create_vector_rotation_matrix(x.device, x.dtype)
        elif rotation_type == "spinor_plus":
            # Spinor representation S+ (Safety reasoning)
            rotation_matrix = self._create_spinor_plus_rotation_matrix(x.device, x.dtype)
        elif rotation_type == "spinor_minus":
            # Spinor representation S- (Authority reasoning)
            rotation_matrix = self._create_spinor_minus_rotation_matrix(x.device, x.dtype)
        else:
            # Default vector rotation
            rotation_matrix = self._create_vector_rotation_matrix(x.device, x.dtype)
        
        # Apply rotation to the last dimension
        x_rotated = torch.matmul(x, rotation_matrix[:x.size(-1), :x.size(-1)])
        
        return x_rotated
    
    def _create_vector_rotation_matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Create SO8 rotation matrix for vector representation."""
        rotation_matrix = torch.eye(8, device=device, dtype=dtype)
        # Apply task-specific rotation
        rotation_matrix += 0.01 * torch.randn_like(rotation_matrix)
        return self._orthogonalize(rotation_matrix)
    
    def _create_spinor_plus_rotation_matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Create SO8 rotation matrix for spinor S+ representation."""
        rotation_matrix = torch.eye(8, device=device, dtype=dtype)
        # Apply safety-specific rotation
        rotation_matrix += 0.01 * torch.randn_like(rotation_matrix)
        # Add safety-specific bias
        rotation_matrix[0, 0] += 0.1  # Safety bias
        return self._orthogonalize(rotation_matrix)
    
    def _create_spinor_minus_rotation_matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Create SO8 rotation matrix for spinor S- representation."""
        rotation_matrix = torch.eye(8, device=device, dtype=dtype)
        # Apply authority-specific rotation
        rotation_matrix += 0.01 * torch.randn_like(rotation_matrix)
        # Add authority-specific bias
        rotation_matrix[1, 1] += 0.1  # Authority bias
        return self._orthogonalize(rotation_matrix)
    
    def _orthogonalize(self, matrix: torch.Tensor) -> torch.Tensor:
        """Orthogonalize matrix to maintain SO8 group properties."""
        # QR decomposition for orthogonalization
        Q, R = torch.qr(matrix)
        # Ensure determinant is +1 (special orthogonal group)
        det = torch.det(Q)
        if det < 0:
            Q[:, 0] *= -1
        return Q
        
    def _flash_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply Flash Attention."""
        # Flash Attention expects [batch_size, seq_len, num_heads, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        # Apply Flash Attention
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout_p=self.attention_dropout if self.training else 0.0,
            softmax_scale=1.0 / math.sqrt(self.head_dim),
            causal=True,
        )
        
        return attn_output
        
    def _standard_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply SO8T Multi-Head Attention with group structure (optimized version).
        
        Optimizations:
        - Chunked attention computation for memory efficiency
        - Efficient causal mask processing
        - Batch matrix operations instead of per-head loops
        """
        batch_size, num_heads, seq_len, head_dim = query_states.shape
        
        # Optimized: Use chunked attention for long sequences
        chunk_size = 512  # Process in chunks for memory efficiency
        use_chunked = seq_len > chunk_size
        
        if use_chunked:
            # Chunked attention computation
            attn_outputs = []
            attn_weights_list = []
            
            for chunk_start in range(0, seq_len, chunk_size):
                chunk_end = min(chunk_start + chunk_size, seq_len)
                
                # Extract chunk
                q_chunk = query_states[:, :, chunk_start:chunk_end, :]
                k_chunk = key_states[:, :, :chunk_end, :]  # Causal: only up to chunk_end
                v_chunk = value_states[:, :, :chunk_end, :]
                
                # Apply SO8 group structure to chunks
                q_chunk, k_chunk, v_chunk = self._apply_group_structure(
                    q_chunk, k_chunk, v_chunk
                )
                
                # Compute attention for chunk
                attn_weights_chunk = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) / math.sqrt(head_dim)
                
                # Apply causal mask for chunk
                if attention_mask is not None:
                    mask_chunk = attention_mask[:, chunk_start:chunk_end, :chunk_end]
                    attn_weights_chunk = attn_weights_chunk + mask_chunk
                else:
                    # Create causal mask for chunk
                    causal_mask = torch.triu(torch.ones(chunk_end - chunk_start, chunk_end, device=q_chunk.device, dtype=q_chunk.dtype), diagonal=1) * -1e9
                    attn_weights_chunk = attn_weights_chunk + causal_mask.unsqueeze(0).unsqueeze(0)
                
                # Apply softmax
                attn_weights_chunk = F.softmax(attn_weights_chunk, dim=-1, dtype=torch.float32).to(q_chunk.dtype)
                
                # Apply dropout
                attn_weights_chunk = self.attn_dropout(attn_weights_chunk)
                
                # Apply attention to values
                attn_output_chunk = torch.matmul(attn_weights_chunk, v_chunk)
                
                attn_outputs.append(attn_output_chunk)
                attn_weights_list.append(attn_weights_chunk)
            
            # Concatenate chunks
            attn_output = torch.cat(attn_outputs, dim=2)
            attn_weights = torch.cat(attn_weights_list, dim=2)
        else:
            # Standard batch attention (optimized for shorter sequences)
            # Apply SO8 group structure to all heads at once
            query_states, key_states, value_states = self._apply_group_structure(
                query_states, key_states, value_states
            )
            
            # Compute attention scores for all heads at once
            # [batch_size, num_heads, seq_len, head_dim] @ [batch_size, num_heads, head_dim, seq_len]
            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(head_dim)
            
            # Apply attention mask efficiently
            if attention_mask is not None:
                # Expand mask to match attention weights shape
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
                elif attention_mask.dim() == 3:
                    attention_mask = attention_mask.unsqueeze(1)
                attn_weights = attn_weights + attention_mask
            else:
                # Create causal mask efficiently
                causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=query_states.device, dtype=query_states.dtype), diagonal=1) * -1e9
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
                attn_weights = attn_weights + causal_mask
            
            # Apply softmax
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            
            # Apply dropout
            attn_weights = self.attn_dropout(attn_weights)
            
            # Apply attention to values
            attn_output = torch.matmul(attn_weights, value_states)
        
        # Apply cross-head SO8 group interaction
        attn_output = self._apply_cross_head_group_interaction(attn_output)
        
        return attn_output, attn_weights
    
    def _apply_head_group_structure(
        self,
        q_head: torch.Tensor,
        k_head: torch.Tensor,
        v_head: torch.Tensor,
        head_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply SO8 group structure to individual attention head."""
        # Apply different SO8 rotations based on head index (Triality symmetry)
        if head_idx % 3 == 0:
            # Vector representation (Task reasoning)
            q_head = self._apply_so8_rotation(q_head, rotation_type="vector")
            k_head = self._apply_so8_rotation(k_head, rotation_type="vector")
            v_head = self._apply_so8_rotation(v_head, rotation_type="vector")
        elif head_idx % 3 == 1:
            # Spinor representation S+ (Safety reasoning)
            q_head = self._apply_so8_rotation(q_head, rotation_type="spinor_plus")
            k_head = self._apply_so8_rotation(k_head, rotation_type="spinor_plus")
            v_head = self._apply_so8_rotation(v_head, rotation_type="spinor_plus")
        else:
            # Spinor representation S- (Authority reasoning)
            q_head = self._apply_so8_rotation(q_head, rotation_type="spinor_minus")
            k_head = self._apply_so8_rotation(k_head, rotation_type="spinor_minus")
            v_head = self._apply_so8_rotation(v_head, rotation_type="spinor_minus")
            
        return q_head, k_head, v_head
    
    def _apply_cross_head_group_interaction(
        self,
        attn_output: torch.Tensor
    ) -> torch.Tensor:
        """Apply SO8 group interaction between attention heads."""
        batch_size, num_heads, seq_len, head_dim = attn_output.shape
        
        # Apply non-commutative group operations between heads
        for head_idx in range(num_heads - 1):
            current_head = attn_output[:, head_idx, :, :]
            next_head = attn_output[:, head_idx + 1, :, :]
            
            # Apply non-commutative group operation
            interaction = self._apply_non_commutative_interaction(current_head, next_head)
            
            # Update heads with group interaction
            attn_output[:, head_idx, :, :] = current_head + 0.1 * interaction
            attn_output[:, head_idx + 1, :, :] = next_head + 0.1 * interaction.transpose(-2, -1)
            
        return attn_output
    
    def _apply_non_commutative_interaction(
        self,
        head1: torch.Tensor,
        head2: torch.Tensor
    ) -> torch.Tensor:
        """Apply non-commutative group interaction between two heads."""
        # Non-commutative operation: head1 @ head2 - head2 @ head1
        interaction = torch.matmul(head1, head2.transpose(-2, -1)) - torch.matmul(head2, head1.transpose(-2, -1))
        return interaction
