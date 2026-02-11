#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runtime requirement checks (Python/uv/torch/CUDA/flash-attn/Unsloth/HF hub).
"""
from __future__ import annotations

import os
import sys
import logging

logger = logging.getLogger(__name__)


def check_runtime_requirements() -> None:
    """Lightweight runtime checks with warnings (non-fatal)."""
    # Python version
    if sys.version_info < (3, 12):
        logger.warning("[RUNTIME] Python 3.12+ recommended (detected %s)", sys.version)

    # uv presence (optional)
    if os.environ.get("UV_CACHE_DIR") is None:
        logger.debug("[RUNTIME] UV_CACHE_DIR not set (uv optional)")

    # Torch + CUDA
    try:
        import torch  # type: ignore

        logger.info("[RUNTIME] torch=%s cuda_available=%s cuda=%s",
                    torch.__version__, torch.cuda.is_available(), torch.version.cuda)
        if torch.version.cuda is not None and float(torch.version.cuda.split(".")[0]) < 12:
            logger.warning("[RUNTIME] CUDA 12.8+ recommended (detected %s)", torch.version.cuda)
    except Exception as e:
        logger.warning("[RUNTIME] torch not available: %s", e)

    # flash-attn
    try:
        import flash_attn  # type: ignore
        logger.info("[RUNTIME] flash-attn available: %s", getattr(flash_attn, "__version__", "unknown"))
    except Exception as e:
        logger.warning("[RUNTIME] flash-attn not available: %s", e)

    # HF hub
    try:
        import huggingface_hub  # type: ignore
        logger.info("[RUNTIME] huggingface_hub available: %s", getattr(huggingface_hub, "__version__", "unknown"))
    except Exception as e:
        logger.warning("[RUNTIME] huggingface_hub not available: %s", e)

    # Unsloth
    try:
        import unsloth  # type: ignore
        logger.info("[RUNTIME] Unsloth available: %s", getattr(unsloth, "__version__", "unknown"))
    except Exception as e:
        logger.warning("[RUNTIME] Unsloth not available: %s", e)

