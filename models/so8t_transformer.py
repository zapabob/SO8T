"""
SO8T Transformer: SO(8)群構造を持つ完全なTransformerアーキテクチャ

This module implements a complete Transformer architecture based on SO(8) group structure,
replacing the base Qwen2.5 model with SO8T-native components while maintaining
compatibility with the original model interface.

Key Features:
- SO(8) group structure in attention mechanisms
- Triality-based three-way reasoning (task, safety, authority)
- PET regularization for temporal consistency
- Non-commutative gates for safety persona protection
- Memory-efficient implementation for RTX3060
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Union
import math
import warnings

from .so8t_group_structure import SO8TGroupStructure
from .so8t_attention import SO8TAttention
from .so8t_mlp import SO8TMLP
from .so8t_embedding import SO8TEmbedding


class SO8TTransformerConfig:
    """Configuration for SO8T Transformer model."""
    
    def __init__(
        self,
        vocab_size: int = 152064,  # Qwen2.5-7B-Instruct準拠
        hidden_size: int = 3584,   # Qwen2.5-7B-Instruct準拠
        intermediate_size: int = 18944,  # Qwen2.5-7B-Instruct準拠
        num_hidden_layers: int = 28,     # Qwen2.5-7B-Instruct準拠
        num_attention_heads: int = 28,   # Qwen2.5-7B-Instruct準拠
        num_key_value_heads: int = 4,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,   # Qwen2.5-7B-Instruct準拠
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        use_cache: bool = False,
        # SO8T specific parameters
        rotation_dim: int = 8,
        safety_weight: float = 0.1,
        cmd_weight: float = 0.9,
        pet_lambda: float = 0.01,
        group_monitoring: bool = True,
        # Memory optimization
        gradient_checkpointing: bool = True,
        use_flash_attention: bool = False,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_cache = use_cache
        
        # SO8T specific parameters
        self.rotation_dim = rotation_dim
        self.safety_weight = safety_weight
        self.cmd_weight = cmd_weight
        self.pet_lambda = pet_lambda
        self.group_monitoring = group_monitoring
        
        # Memory optimization
        self.gradient_checkpointing = gradient_checkpointing
        self.use_flash_attention = use_flash_attention


class SO8TTransformerLayer(nn.Module):
    """Single SO8T Transformer layer with SO(8) group structure."""
    
    def __init__(self, config: SO8TTransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # SO8T Group Structure
        self.group_structure = SO8TGroupStructure(
            hidden_size=config.hidden_size,
            lambda_pet=config.pet_lambda
        )
        
        # SO8T Attention
        self.self_attn = SO8TAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            attention_dropout=config.attention_dropout,
            bias=config.attention_bias,
            use_flash_attention=config.use_flash_attention
        )
        
        # SO8T MLP
        self.mlp = SO8TMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act
        )
        
        # Layer normalization
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
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
        Forward pass for SO8T Transformer layer.
        
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
        residual = hidden_states
        
        # Pre-attention layer norm
        hidden_states = self.input_layernorm(hidden_states)
        
        # Apply SO8T group structure to hidden states
        group_output, group_info = self.group_structure(
            hidden_states, 
            return_group_info=True
        )
        
        # Self-attention with SO8T group structure
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=group_output,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        
        # Residual connection
        hidden_states = residual + hidden_states
        
        # Post-attention layer norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # MLP
        hidden_states = self.mlp(hidden_states)
        
        # Residual connection
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
            
        # Add group information for monitoring
        if hasattr(self, 'group_info'):
            self.group_info = group_info
            
        return outputs


class SO8TTransformerModel(nn.Module):
    """Complete SO8T Transformer model replacing Qwen2.5."""
    
    def __init__(self, config: SO8TTransformerConfig):
        super().__init__()
        self.config = config
        
        # SO8T Embedding
        self.embed_tokens = SO8TEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size
        )
        
        # SO8T Transformer layers
        self.layers = nn.ModuleList([
            SO8TTransformerLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Initialize weights
        self.post_init()
        
    def post_init(self):
        """Initialize weights after model creation."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Initialize linear layers with Xavier uniform
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # Initialize embeddings with normal distribution
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.RMSNorm):
                # Initialize RMSNorm
                nn.init.ones_(module.weight)
                
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for SO8T Transformer model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            position_ids: Position indices [batch_size, seq_len]
            past_key_values: Past key-value caches
            inputs_embeds: Input embeddings [batch_size, seq_len, hidden_size]
            use_cache: Whether to use key-value cache
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dictionary
            
        Returns:
            Dictionary containing model outputs
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.attention_bias
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else True
        
        # Get input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            
        # Get position embeddings
        if position_ids is None:
            position_ids = self._get_position_ids(input_ids, attention_mask)
            
        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = self._prepare_attention_mask(attention_mask, input_ids.shape)
            
        # Initialize hidden states
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        # Forward through SO8T Transformer layers
        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            # Get past key-value for this layer
            past_key_value = past_key_values[layer_idx] if past_key_values is not None else None
            
            # Forward through layer
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
                
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            
        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attns] if v is not None)
            
        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_decoder_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attns,
        }
        
    def _get_position_ids(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get position IDs for input tokens."""
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        position_ids = torch.arange(
            self.config.max_position_embeddings,
            device=device,
            dtype=torch.long
        )
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        if attention_mask is not None:
            # Mask out padding positions
            position_ids = position_ids.masked_fill(attention_mask == 0, 0)
            
        return position_ids[:, :seq_length]
        
    def _prepare_attention_mask(self, attention_mask: torch.Tensor, input_shape: Tuple[int, int]) -> torch.Tensor:
        """Prepare attention mask for attention computation."""
        batch_size, seq_length = input_shape
        device = attention_mask.device
        
        # Create causal mask
        causal_mask = torch.tril(torch.ones(seq_length, seq_length, device=device))
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_length, -1)
            causal_mask = causal_mask * attention_mask
            
        return causal_mask


class SO8TTransformerForCausalLM(nn.Module):
    """SO8T Transformer for causal language modeling with triality reasoning."""
    
    def __init__(self, config: SO8TTransformerConfig):
        super().__init__()
        self.config = config
        
        # SO8T Transformer model
        self.model = SO8TTransformerModel(config)
        
        # Triality reasoning heads
        self.task_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.safety_head = nn.Linear(config.hidden_size, 2, bias=True)  # 0: safe, 1: unsafe
        self.authority_head = nn.Linear(config.hidden_size, 2, bias=True)  # 0: handle, 1: escalate
        
        # Initialize weights
        self.post_init()
        
    def post_init(self):
        """Initialize weights after model creation."""
        # Initialize task head (language modeling)
        nn.init.normal_(self.task_head.weight, mean=0.0, std=0.02)
        
        # Initialize safety head
        nn.init.normal_(self.safety_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.safety_head.bias)
        
        # Initialize authority head
        nn.init.normal_(self.authority_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.authority_head.bias)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        safety_labels: Optional[torch.Tensor] = None,
        authority_labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for SO8T Transformer with triality reasoning.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            position_ids: Position indices [batch_size, seq_len]
            past_key_values: Past key-value caches
            inputs_embeds: Input embeddings [batch_size, seq_len, hidden_size]
            labels: Language modeling labels [batch_size, seq_len]
            safety_labels: Safety classification labels [batch_size]
            authority_labels: Authority classification labels [batch_size]
            use_cache: Whether to use key-value cache
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dictionary
            
        Returns:
            Dictionary containing model outputs
        """
        # Forward through SO8T Transformer
        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        hidden_states = transformer_outputs["last_hidden_state"]
        
        # Triality reasoning heads
        task_logits = self.task_head(hidden_states)
        safety_logits = self.safety_head(hidden_states)
        authority_logits = self.authority_head(hidden_states)
        
        # Calculate losses
        total_loss = None
        task_loss = None
        safety_loss = None
        authority_loss = None
        
        if labels is not None:
            # Language modeling loss
            shift_logits = task_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            task_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            total_loss = task_loss
            
        if safety_labels is not None:
            # Safety classification loss
            safety_loss = F.cross_entropy(safety_logits, safety_labels)
            if total_loss is None:
                total_loss = safety_loss
            else:
                total_loss += safety_loss
                
        if authority_labels is not None:
            # Authority classification loss
            authority_loss = F.cross_entropy(authority_logits, authority_labels)
            if total_loss is None:
                total_loss = authority_loss
            else:
                total_loss += authority_loss
                
        if not return_dict:
            return tuple(v for v in [total_loss, task_logits, safety_logits, authority_logits] if v is not None)
            
        return {
            "loss": total_loss,
            "task_loss": task_loss,
            "safety_loss": safety_loss,
            "authority_loss": authority_loss,
            "task_logits": task_logits,
            "safety_logits": safety_logits,
            "authority_logits": authority_logits,
            "hidden_states": transformer_outputs["hidden_states"],
            "attentions": transformer_outputs["attentions"],
            "past_key_values": transformer_outputs["past_key_values"],
        }
        
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text using SO8T Transformer with triality reasoning.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            
        Returns:
            Generated token IDs [batch_size, generated_length]
        """
        # Simple greedy generation for now
        # TODO: Implement proper generation with triality reasoning
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        generated_ids = input_ids.clone()
        
        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass
            outputs = self.forward(
                input_ids=generated_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Get next token logits
            next_token_logits = outputs["task_logits"][:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
                
            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
                
            # Sample next token
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(batch_size, 1, device=device, dtype=attention_mask.dtype)
                ], dim=1)
                
            # Check for EOS token
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
                
        return generated_ids
