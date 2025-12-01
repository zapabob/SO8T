#!/usr/bin/env python3
"""Unslothモデルの構造を正確にデバッグ"""

from unsloth import FastLanguageModel

def debug_model_structure():
    """Unslothモデルの構造をデバッグ"""
    try:
        print("Loading Unsloth model for debugging...")

        model, tokenizer = FastLanguageModel.from_pretrained(
            'models/Borea-Phi-3.5-mini-Instruct-Jp',
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
            device_map='auto'
        )

        print(f'Model type: {type(model)}')
        print(f'Model class: {model.__class__.__name__}')
        print(f'hasattr(model, "layers"): {hasattr(model, "layers")}')
        print(f'hasattr(model, "model"): {hasattr(model, "model")}')

        if hasattr(model, 'model'):
            print(f'hasattr(model.model, "layers"): {hasattr(model.model, "layers")}')
            if hasattr(model.model, 'layers'):
                print(f'len(model.model.layers): {len(model.model.layers)}')
                print(f'Type of first layer: {type(model.model.layers[0])}')
        else:
            print('model.model does not exist')

        # 直接属性アクセスを試す
        try:
            layers = model.layers
            print(f'model.layers exists, length: {len(layers)}')
        except AttributeError:
            print('model.layers does not exist')

        print("Debug completed")

    except Exception as e:
        print(f"Failed to debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model_structure()





