#!/usr/bin/env python3
"""PEFT適用後のモデルの構造をデバッグ"""

from unsloth import FastLanguageModel

def debug_peft_after_lora():
    """PEFT適用後のモデルの構造をデバッグ"""
    try:
        print("Loading and applying PEFT for debugging...")

        model, tokenizer = FastLanguageModel.from_pretrained(
            'models/Borea-Phi-3.5-mini-Instruct-Jp',
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
            device_map='auto'
        )

        print("Before PEFT:")
        print(f"  Model type: {type(model)}")
        print(f"  hasattr(model, 'model'): {hasattr(model, 'model')}")
        if hasattr(model, 'model'):
            print(f"  hasattr(model.model, 'layers'): {hasattr(model.model, 'layers')}")

        # PEFT適用
        print("Applying PEFT...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
        )

        print("After PEFT:")
        print(f"  Model type: {type(model)}")
        print(f"  Model class: {model.__class__.__name__}")
        print(f"  hasattr(model, 'model'): {hasattr(model, 'model')}")
        if hasattr(model, 'model'):
            print(f"  hasattr(model.model, 'layers'): {hasattr(model.model, 'layers')}")

        print(f"  hasattr(model, 'base_model'): {hasattr(model, 'base_model')}")
        if hasattr(model, 'base_model'):
            print(f"  base_model type: {type(model.base_model)}")
            print(f"  hasattr(base_model, 'model'): {hasattr(model.base_model, 'model')}")
            if hasattr(model.base_model, 'model'):
                print(f"  hasattr(base_model.model, 'layers'): {hasattr(model.base_model.model, 'layers')}")
                if hasattr(model.base_model.model, 'layers'):
                    print(f"  len(base_model.model.layers): {len(model.base_model.model.layers)}")

        # 様々なアクセス方法を試す
        print("Trying different access methods:")
        access_methods = [
            "model.model.layers",
            "model.base_model.model.layers",
            "model.layers",
            "model.base_model.layers"
        ]

        for method in access_methods:
            try:
                obj = model
                for attr in method.split('.'):
                    if attr == 'model':
                        continue
                    obj = getattr(obj, attr)
                if hasattr(obj, '__len__'):
                    print(f"  {method}: SUCCESS, length={len(obj)}")
                else:
                    print(f"  {method}: SUCCESS")
            except AttributeError as e:
                print(f"  {method}: FAILED - {e}")

        print("Debug completed")

    except Exception as e:
        print(f"Failed to debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_peft_after_lora()


