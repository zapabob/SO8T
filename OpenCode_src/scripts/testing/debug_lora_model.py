#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA適用後のPhi-3モデル構造デバッグ
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

def analyze_lora_model():
    print("=== Testing LoRA-applied model structure ===")
    model = AutoModelForCausalLM.from_pretrained('AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp', torch_dtype='auto')

    # LoRA設定
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    )

    model = get_peft_model(model, lora_config)
    print('After LoRA:')
    print('model type:', type(model))
    print('hasattr(model, "base_model"):', hasattr(model, 'base_model'))

    if hasattr(model, 'base_model'):
        print('model.base_model type:', type(model.base_model))
        print('hasattr(model.base_model, "model"):', hasattr(model.base_model, 'model'))
        print('hasattr(model.base_model, "layers"):', hasattr(model.base_model, 'layers'))

        if hasattr(model.base_model, 'model'):
            print('hasattr(model.base_model.model, "layers"):', hasattr(model.base_model.model, 'layers'))

    # attach_nkat_adapters のロジックをテスト
    print("\n=== Testing attach_nkat_adapters logic on LoRA model ===")

    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        print("Path 1: model.base_model.model.layers")
        try:
            layers = model.base_model.model.layers
            print("SUCCESS: Found layers via Path 1, len =", len(layers))
        except AttributeError as e:
            print("FAILED: Path 1 error:", e)

    elif hasattr(model, "base_model") and hasattr(model.base_model, "layers"):
        print("Path 2: model.base_model.layers")
        try:
            layers = model.base_model.layers
            print("SUCCESS: Found layers via Path 2, len =", len(layers))
        except AttributeError as e:
            print("FAILED: Path 2 error:", e)

    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        print("Path 3: model.model.layers")
        try:
            layers = model.model.layers
            print("SUCCESS: Found layers via Path 3, len =", len(layers))
        except AttributeError as e:
            print("FAILED: Path 3 error:", e)

    else:
        print("No valid path found")

if __name__ == "__main__":
    analyze_lora_model()

