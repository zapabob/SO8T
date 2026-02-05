#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phi-3モデル構造デバッグ
"""

from transformers import AutoModelForCausalLM

def analyze_model_structure():
    print("=== Model Structure Analysis ===")
    model = AutoModelForCausalLM.from_pretrained('AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp', torch_dtype='auto')

    print('model type:', type(model))
    print('model.base_model type:', type(model.base_model))
    print('hasattr(model.base_model, "model"):', hasattr(model.base_model, 'model'))
    print('hasattr(model.base_model, "layers"):', hasattr(model.base_model, 'layers'))
    print('hasattr(model, "model"):', hasattr(model, 'model'))

    if hasattr(model.base_model, 'model'):
        print('model.base_model.model type:', type(model.base_model.model))
        print('hasattr(model.base_model.model, "layers"):', hasattr(model.base_model.model, 'layers'))

    if hasattr(model.base_model, 'layers'):
        print('model.base_model.layers type:', type(model.base_model.layers))
        print('len(model.base_model.layers):', len(model.base_model.layers))

        # 最初の層の構造を確認
        if len(model.base_model.layers) > 0:
            first_layer = model.base_model.layers[0]
            print('first_layer type:', type(first_layer))

    # attach_nkat_adapters のロジックをシミュレート
    print("\n=== attach_nkat_adapters Logic Simulation ===")

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
    analyze_model_structure()

