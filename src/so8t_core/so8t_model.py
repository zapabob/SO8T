"""
SO8T/thinking Model Implementation

Complete SO(8) geometric reasoning model with triality principle.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from typing import Optional, Tuple, List
import math

from .so8t_layer import (
    SO8TReasoningLayer,
    SO8RotationGate,
    SO8TGeometricAttention,
    orthogonality_loss,
    triality_consistency_loss,
    SO8TConfig
)

class SO8TPretrainedConfig(PretrainedConfig):
    """Configuration class for SO8T model."""

    model_type = "so8t"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=2048,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=8192,
        hidden_dropout_prob=0.1,
        attention_dropout_prob=0.1,
        max_position_embeddings=4096,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache

class SO8TEmbeddings(nn.Module):
    """SO8T Embeddings with geometric position encoding."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # SO(8) geometric position embeddings
        self.geo_position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def _create_geometric_positions(self, seq_length, device):
        """Create SO(8) geometric position encodings."""
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(1, -1)

        # SO(8) geometric transformation of positions
        geo_positions = torch.sin(position_ids.float() / 10000 ** (torch.arange(0, self.geo_position_embeddings.embedding_dim, 2, device=device).float() / self.geo_position_embeddings.embedding_dim))
        geo_positions = geo_positions.unsqueeze(0)

        return geo_positions

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device if input_ids is not None else inputs_embeds.device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape[:1])

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)

        # Add geometric position embeddings
        geo_positions = self._create_geometric_positions(seq_length, inputs_embeds.device)
        geo_position_embeddings = self.geo_position_embeddings(position_ids)

        embeddings = inputs_embeds + position_embeddings + geo_position_embeddings * geo_positions

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class SO8TModel(PreTrainedModel):
    """Core SO8T/thinking model."""

    config_class = SO8TPretrainedConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = SO8TEmbeddings(config)

        # SO(8) reasoning layers
        self.layers = nn.ModuleList([
            SO8TReasoningLayer(
                config.hidden_size,
                config.num_attention_heads,
                config.attention_dropout_prob
            )
            for _ in range(config.num_hidden_layers)
        ])

        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with geometric considerations."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Special initialization for SO(8) parameters
        if hasattr(module, 'rotation_params'):
            # Initialize rotation parameters to identity (no rotation)
            nn.init.zeros_(module.rotation_params)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=self.device)

        # Embeddings
        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds
        )

        all_hidden_states = () if output_hidden_states else None

        # SO(8) reasoning layers
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer(hidden_states, attention_mask=attention_mask)
            hidden_states = layer_outputs

        hidden_states = self.final_layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return {
            'last_hidden_state': hidden_states,
            'hidden_states': all_hidden_states,
        }

class SO8TForCausalLM(PreTrainedModel):
    """SO8T model for causal language modeling."""

    config_class = SO8TPretrainedConfig

    def __init__(self, config):
        super().__init__(config)

        self.so8t = SO8TModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.so8t(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs['last_hidden_state'] if isinstance(outputs, dict) else outputs[0]

        # Language modeling head
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return {
            'loss': loss,
            'logits': lm_logits,
            'hidden_states': outputs.get('hidden_states'),
        }

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        # Only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past
