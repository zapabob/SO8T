#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRAPE (Group Representational Position Encoding) utilities.

Implements multiplicative GRAPE (commuting MS-GRAPE) as a RoPE-compatible
replacement based on arXiv:2512.07805. This module provides a drop-in
rotary embedding and a patch helper for HF-style models.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Tuple
import logging
import math
import importlib
import types
import torch

logger = logging.getLogger(__name__)


@dataclass
class GrapePatchConfig:
    """Configuration for GRAPE rotary patching."""
    base: float = 10000.0
    learnable_freq: bool = True
    log_freq_scale: float = 16.0
    attention_scaling: float = 1.0
    variant: str = "commuting_ms_grape"
    additive_bias_max: float = 8.0
    additive_bias_type: str = "alibi"

    def as_dict(self) -> dict:
        return asdict(self)


class GrapeRotaryEmbedding(torch.nn.Module):
    """GRAPE (commuting MS-GRAPE) compatible rotary embedding.

    - Learnable log-frequency spectrum initialized with RoPE log-uniform base.
    - Drop-in replacement for LlamaRotaryEmbedding style forward(x, position_ids).
    - Implements multiplicative GRAPE (SO(d)) commuting subspace variant.
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        learnable_freq: bool = True,
        log_freq_scale: float = 16.0,
        attention_scaling: float = 1.0,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("GRAPE rotary dim must be even")

        self.dim = dim
        self.base = float(base)
        self.learnable_freq = learnable_freq
        self.log_freq_scale = float(log_freq_scale)
        self.attention_scaling = float(attention_scaling)

        # RoPE-style log-uniform spectrum
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        log_init = inv_freq.log().float() * self.log_freq_scale
        self.log_freq = torch.nn.Parameter(log_init, requires_grad=learnable_freq)

    def _apply(self, fn):
        # keep log_freq in fp32 even if module-wide dtype casts happen
        super()._apply(fn)
        if getattr(self, "log_freq", None) is not None:
            self.log_freq.data = self.log_freq.data.float()
            if self.log_freq.grad is not None:
                self.log_freq.grad.data = self.log_freq.grad.data.float()
        return self

    @property
    def freq(self) -> torch.Tensor:
        scaled_log_freq = self.log_freq / self.log_freq_scale
        return torch.exp(scaled_log_freq)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None,
    ):
        if position_ids is None:
            if seq_len is None:
                seq_len = x.shape[-2]
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)

        inv_freq_expanded = self.freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _resolve_head_dim(module, model_config) -> Optional[int]:
    rotary = getattr(module, "rotary_emb", None)
    head_dim = getattr(rotary, "dim", None) or getattr(module, "head_dim", None)
    if head_dim is None and model_config is not None:
        head_dim = getattr(model_config, "head_dim", None)
        if head_dim is None and hasattr(model_config, "hidden_size"):
            num_heads = getattr(model_config, "num_attention_heads", None)
            if num_heads:
                head_dim = int(model_config.hidden_size // num_heads)
    return head_dim


def _resolve_num_heads(model) -> Optional[int]:
    config = getattr(model, "config", None)
    num_heads = getattr(config, "num_attention_heads", None) if config else None
    if num_heads is None:
        num_heads = getattr(model, "num_attention_heads", None)
    return num_heads


def _get_alibi_slopes(num_heads: int) -> torch.Tensor:
    """Return ALiBi slopes (from Press et al.)."""
    def get_slopes_power_of_2(n):
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if num_heads < 1:
        raise ValueError("num_heads must be >= 1")
    if math.log2(num_heads).is_integer():
        slopes = get_slopes_power_of_2(num_heads)
    else:
        closest_power = 2 ** math.floor(math.log2(num_heads))
        slopes = get_slopes_power_of_2(closest_power)
        extra = _get_alibi_slopes(2 * closest_power)
        slopes += extra[0::2][: num_heads - closest_power]
    return torch.tensor(slopes, dtype=torch.float32)


def _build_alibi_bias(
    slopes: Optional[torch.Tensor],
    q_len: int,
    kv_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if slopes is None:
        return None
    if not isinstance(slopes, torch.Tensor):
        slopes = torch.tensor(slopes, dtype=torch.float32)
    slopes = slopes.to(device=device, dtype=torch.float32).view(1, -1, 1, 1)
    positions = torch.arange(1 - kv_len, 1, device=device, dtype=torch.float32).view(1, 1, 1, kv_len)
    alibi = slopes * positions
    if q_len != 1:
        alibi = alibi.expand(-1, -1, q_len, -1)
    return alibi.to(dtype=dtype)


def _resolve_phi3_ops(module: torch.nn.Module) -> Tuple[Optional[callable], Optional[callable]]:
    module_name = getattr(module.__class__, "__module__", "")
    if not module_name:
        return None, None
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        return None, None
    apply_rotary_pos_emb = getattr(mod, "apply_rotary_pos_emb", None)
    repeat_kv = getattr(mod, "repeat_kv", None)
    return apply_rotary_pos_emb, repeat_kv


def _phi3_attention_forward_with_alibi(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[object] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    bsz, q_len, _ = hidden_states.size()

    qkv = self.qkv_proj(hidden_states)
    query_pos = self.num_heads * self.head_dim
    query_states = qkv[..., :query_pos]
    key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
    value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                "The cache structure has changed since version v4.36."
                " Please initialize attention with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
    apply_rotary_pos_emb = getattr(self, "_grape_apply_rotary_pos_emb", None)
    repeat_kv = getattr(self, "_grape_repeat_kv", None)
    if apply_rotary_pos_emb is None or repeat_kv is None:
        raise RuntimeError("GRAPE additive patch missing rotary helpers")

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is None:
        min_val = torch.finfo(attn_weights.dtype).min
        diagonal = 1 + kv_seq_len - q_len
        causal = torch.zeros((q_len, kv_seq_len), dtype=attn_weights.dtype, device=attn_weights.device)
        if diagonal > 0:
            causal = causal + torch.triu(
                torch.full((q_len, kv_seq_len), min_val, dtype=attn_weights.dtype, device=attn_weights.device),
                diagonal=diagonal,
            )
        attention_mask = causal.view(1, 1, q_len, kv_seq_len).expand(bsz, 1, q_len, kv_seq_len)

    if attention_mask is not None:
        if attention_mask.size() not in {
            (bsz, 1, q_len, kv_seq_len),
            (bsz, self.num_heads, q_len, kv_seq_len),
        }:
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)} or "
                f"{(bsz, self.num_heads, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        if attention_mask.dtype == torch.bool:
            attention_mask = attention_mask.masked_fill(
                attention_mask, torch.finfo(attn_weights.dtype).min
            )
        attention_mask = attention_mask.to(dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = attn_weights + attention_mask

    alibi = _build_alibi_bias(
        slopes=getattr(self, "grape_additive_bias", None),
        q_len=q_len,
        kv_len=kv_seq_len,
        device=attn_weights.device,
        dtype=attn_weights.dtype,
    )
    if alibi is not None:
        attn_weights = attn_weights + alibi

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
    attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def _phi3_attention_forward_with_alibi_v2(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[object] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    qkv = self.qkv_proj(hidden_states)
    query_pos = self.config.num_attention_heads * self.head_dim
    query_states = qkv[..., :query_pos]
    key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
    value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

    query_states = query_states.view(hidden_shape).transpose(1, 2)
    key_states = key_states.view(hidden_shape).transpose(1, 2)
    value_states = value_states.view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    apply_rotary_pos_emb = getattr(self, "_grape_apply_rotary_pos_emb", None)
    repeat_kv = getattr(self, "_grape_repeat_kv", None)
    if apply_rotary_pos_emb is None or repeat_kv is None:
        raise RuntimeError("GRAPE additive patch missing rotary helpers")

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        if causal_mask.dtype == torch.bool:
            causal_mask = causal_mask.masked_fill(
                causal_mask, torch.finfo(attn_weights.dtype).min
            )
        attn_weights = attn_weights + causal_mask.to(dtype=attn_weights.dtype, device=attn_weights.device)
    else:
        min_val = torch.finfo(attn_weights.dtype).min
        q_len = attn_weights.shape[-2]
        kv_len = attn_weights.shape[-1]
        diagonal = 1 + kv_len - q_len
        causal = torch.zeros((q_len, kv_len), dtype=attn_weights.dtype, device=attn_weights.device)
        if diagonal > 0:
            causal = causal + torch.triu(
                torch.full((q_len, kv_len), min_val, dtype=attn_weights.dtype, device=attn_weights.device),
                diagonal=diagonal,
            )
        attn_weights = attn_weights + causal.view(1, 1, q_len, kv_len)

    q_len = attn_weights.shape[-2]
    kv_len = attn_weights.shape[-1]
    alibi = _build_alibi_bias(
        slopes=getattr(self, "grape_additive_bias", None),
        q_len=q_len,
        kv_len=kv_len,
        device=attn_weights.device,
        dtype=attn_weights.dtype,
    )
    if alibi is not None:
        attn_weights = attn_weights + alibi

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1)
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def enable_additive_grape(model, config: GrapePatchConfig) -> bool:
    """Enable additive GRAPE (ALiBi/FoX-style) by configuring model flags.

    This does not modify attention kernels directly; it stores ALiBi slopes
    and sets config hints so downstream kernels can consume them.
    """
    num_heads = _resolve_num_heads(model)
    if num_heads is None:
        logger.warning("Additive GRAPE skipped: num_attention_heads not found")
        return False

    slopes = _get_alibi_slopes(num_heads)
    model.grape_additive_bias = slopes

    config_obj = getattr(model, "config", None)
    if config_obj is not None:
        if hasattr(config_obj, "position_embedding_type"):
            config_obj.position_embedding_type = "alibi"
        if hasattr(config_obj, "alibi"):
            config_obj.alibi = True
        if hasattr(config_obj, "max_alibi_bias"):
            config_obj.max_alibi_bias = float(config.additive_bias_max)
        if hasattr(config_obj, "alibi_bias_max"):
            config_obj.alibi_bias_max = float(config.additive_bias_max)

    logger.info("Additive GRAPE enabled (%s, heads=%d)", config.additive_bias_type, num_heads)
    return True


def patch_attention_with_additive_grape(model, config: GrapePatchConfig) -> int:
    """Patch attention modules to apply additive GRAPE/ALiBi bias."""
    slopes = getattr(model, "grape_additive_bias", None)
    if slopes is None:
        num_heads = _resolve_num_heads(model)
        if num_heads is None:
            logger.warning("Additive GRAPE patch skipped: num_heads missing")
            return 0
        slopes = _get_alibi_slopes(num_heads)
        model.grape_additive_bias = slopes

    config_obj = getattr(model, "config", None)
    if config_obj is not None:
        if hasattr(config_obj, "use_flash_attention_2"):
            config_obj.use_flash_attention_2 = False
        if hasattr(config_obj, "_attn_implementation"):
            config_obj._attn_implementation = "eager"
        if hasattr(config_obj, "attn_implementation"):
            config_obj.attn_implementation = "eager"

    patched = 0
    required_attrs = (
        "qkv_proj",
        "o_proj",
        "num_key_value_heads",
        "num_key_value_groups",
        "head_dim",
        "attention_dropout",
    )
    for name, module in model.named_modules():
        if getattr(module, "_grape_additive_patched", False):
            continue
        if not all(hasattr(module, attr) for attr in required_attrs):
            continue
        apply_rotary_pos_emb, repeat_kv = _resolve_phi3_ops(module)
        if apply_rotary_pos_emb is None or repeat_kv is None:
            continue
        uses_position_embeddings = not hasattr(module, "rotary_emb")
        module.grape_additive_bias = slopes
        module._grape_apply_rotary_pos_emb = apply_rotary_pos_emb
        module._grape_repeat_kv = repeat_kv
        module._grape_additive_patched = True
        module._grape_original_forward = module.forward
        if uses_position_embeddings:
            module.forward = types.MethodType(_phi3_attention_forward_with_alibi_v2, module)
        else:
            module.forward = types.MethodType(_phi3_attention_forward_with_alibi, module)
        patched += 1

    if patched:
        logger.info("Additive GRAPE patched %d attention modules", patched)
    return patched


def patch_rotary_embeddings(model, config: GrapePatchConfig) -> int:
    """Patch model rotary embeddings in-place with GRAPE rotary embeddings."""
    patched = 0
    for name, module in model.named_modules():
        if not hasattr(module, "rotary_emb"):
            continue

        head_dim = _resolve_head_dim(module, getattr(model, "config", None))
        if head_dim is None or head_dim % 2 != 0:
            logger.warning(f"Skipping GRAPE patch for {name}: invalid head_dim={head_dim}")
            continue

        rotary = getattr(module, "rotary_emb", None)
        attention_scaling = getattr(rotary, "attention_scaling", config.attention_scaling)
        module.rotary_emb = GrapeRotaryEmbedding(
            dim=int(head_dim),
            base=config.base,
            learnable_freq=config.learnable_freq,
            log_freq_scale=config.log_freq_scale,
            attention_scaling=attention_scaling,
        )
        patched += 1

    return patched
