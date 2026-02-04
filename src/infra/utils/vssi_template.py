# -*- coding: utf-8 -*-
"""VSSI template helpers for SO8T.

Provides a unified 4-way thinking tag renderer (task/safety/policy/analysis)
with optional legacy <think> tags.
"""
from __future__ import annotations

from typing import Dict, Optional

DEFAULT_STYLE = "legacy"


def resolve_think_tags(style: Optional[str] = None) -> Dict[str, str]:
    """Resolve <think> tag style for legacy rendering."""
    style_norm = (style or DEFAULT_STYLE).strip().lower()
    if style_norm in ("openai", "thinking"):
        return {"start": "<think>", "end": "</thinking>"} if style_norm == "openai" else {"start": "<thinking>", "end": "</thinking>"}
    return {"start": "<think>", "end": "</think>"}


def render_thinking(
    task_block: str,
    safety_block: str,
    policy_block: str,
    analysis_block: Optional[str] = None,
    use_quadruple: bool = False,
    style: Optional[str] = None,
    alpha_gate: str = "<alpha_gate> VALIDATED </alpha_gate>",
) -> str:
    """Render VSSI thinking blocks.

    When use_quadruple=True, emits 4-way thinking tags:
    <think-task>, <think-analysis>, <think-safety>, <think-policy>.
    """
    analysis_block = analysis_block or "[Spinor_Plus_Logic]\n- Reasoning path: pending"  # safe fallback
    if use_quadruple:
        return (
            f"<think-task>\n{task_block}\n</think-task>\n\n"
            f"<think-analysis>\n{analysis_block}\n</think-analysis>\n\n"
            f"<think-safety>\n{safety_block}\n</think-safety>\n\n"
            f"<think-policy>\n{policy_block}\n</think-policy>\n\n"
            f"{alpha_gate}"
        )
    tags = resolve_think_tags(style)
    payload = "\n\n".join([task_block, analysis_block, safety_block, policy_block])
    return f"{tags['start']}\n{payload}\n{tags['end']}\n\n{alpha_gate}"


def normalize_prompt_text(text: str) -> str:
    """Normalize prompt text for mapping lookup."""
    return " ".join((text or "").strip().split()).lower()
