"""
Advanced patch utilities (GRAPE/mHC) for model augmentation.
Wraps existing implementations under scripts/models to keep src/ layout.
"""
from __future__ import annotations

from typing import Dict, Optional

try:
    from src.models.grape_position_encoding import (
        GrapePatchConfig,
        patch_rotary_embeddings,
        enable_additive_grape,
        patch_attention_with_additive_grape,
    )
except Exception:
    GrapePatchConfig = None
    patch_rotary_embeddings = None
    enable_additive_grape = None
    patch_attention_with_additive_grape = None

try:
    from src.models.mhc_manifold import apply_mhc_projection_to_model
except Exception:
    apply_mhc_projection_to_model = None


def apply_grape(model, cfg: Dict) -> bool:
    if not cfg.get("enabled"):
        return False
    if patch_rotary_embeddings is None or GrapePatchConfig is None:
        return False
    variant = cfg.get("variant", "multiplicative")
    config = GrapePatchConfig(variant=variant)
    variant_lower = (variant or "multiplicative").lower()
    did_patch = False
    if variant_lower in {"multiplicative", "commuting_ms_grape", "hybrid"}:
        patch_rotary_embeddings(model, config)
        did_patch = True
    if variant_lower in {"additive", "alibi", "fox", "hybrid"}:
        if enable_additive_grape is not None:
            enable_additive_grape(model, config)
            if patch_attention_with_additive_grape is not None:
                patch_attention_with_additive_grape(model, config)
            did_patch = True
    return did_patch


def apply_mhc(model, cfg: Dict) -> int:
    if not cfg.get("enabled"):
        return 0
    if apply_mhc_projection_to_model is None:
        return 0
    targets = cfg.get("targets", ["o_proj", "down_proj", "up_proj", "gate_proj"])
    blend = float(cfg.get("blend", 0.1))
    max_iter = int(cfg.get("max_iter", 20))
    updated = apply_mhc_projection_to_model(
        model, target_modules=targets, max_iter=max_iter, blend=blend
    )
    return len(updated)
