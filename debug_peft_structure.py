#!/usr/bin/env python3
"""PEFTモデルの構造をデバッグ"""

from unsloth import FastLanguageModel

def debug_peft_structure():
    """PEFTモデルの構造をデバッグ"""
    try:
        print("Loading PEFT model for debugging...")

        model, tokenizer = FastLanguageModel.from_pretrained(
            'models/Borea-Phi-3.5-mini-Instruct-Jp',
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
            device_map='auto'
        )

        print(f'Model type: {type(model)}')
        print(f'Model class: {model.__class__.__name__}')

        # PEFTモデルの構造を確認
        print(f'hasattr(model, "model"): {hasattr(model, "model")}')
        if hasattr(model, 'model'):
            print(f'hasattr(model.model, "layers"): {hasattr(model.model, "layers")}')
            if hasattr(model.model, 'layers'):
                print(f'len(model.model.layers): {len(model.model.layers)}')

        # base_model属性を確認
        print(f'hasattr(model, "base_model"): {hasattr(model, "base_model")}')
        if hasattr(model, 'base_model'):
            print(f'hasattr(base_model, "model"): {hasattr(model.base_model, "model")}')
            if hasattr(model.base_model, 'model'):
                print(f'hasattr(base_model.model, "layers"): {hasattr(model.base_model.model, "layers")}')
                if hasattr(model.base_model.model, 'layers'):
                    print(f'len(base_model.model.layers): {len(model.base_model.model.layers)}')

        # 直接属性探索
        print("Exploring model attributes:")
        for attr in dir(model):
            if not attr.startswith('_') and 'layer' in attr.lower():
                print(f"  {attr}: {type(getattr(model, attr))}")

        print("Debug completed")

    except Exception as e:
        print(f"Failed to debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_peft_structure()





