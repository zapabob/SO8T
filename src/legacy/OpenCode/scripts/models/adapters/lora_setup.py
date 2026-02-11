"""
LoRA / QLoRA helpers tuned for RTX3060 12GB workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
from torch import nn

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except ImportError:  # pragma: no cover - PEFT not strictly required for static analysis
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None


@dataclass
class LoRAParams:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Optional[Iterable[str]] = None
    bias: str = "none"


def default_target_modules() -> List[str]:
    return ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"]


def setup_lora(model: nn.Module, params: LoRAParams) -> nn.Module:
    if LoraConfig is None or get_peft_model is None:
        print("PEFT not available; returning model without LoRA adapters.")
        return model

    target = list(params.target_modules or default_target_modules())
    config = LoraConfig(
        r=params.r,
        lora_alpha=params.alpha,
        lora_dropout=params.dropout,
        target_modules=target,
        bias=params.bias,
        task_type="CAUSAL_LM",
    )

    if prepare_model_for_kbit_training is not None:
        model = prepare_model_for_kbit_training(model)

    return get_peft_model(model, config)


def freeze_except_lora(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def compute_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
