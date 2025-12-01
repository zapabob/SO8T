#!/usr/bin/env python3
"""PEFTモデルの詳細構造をデバッグ"""

from unsloth import FastLanguageModel

def debug_peft_detailed():
    """PEFTモデルの詳細構造をデバッグ"""
    try:
        print("Loading and applying PEFT for detailed debugging...")

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
        print(f"  hasattr(model, 'base_model'): {hasattr(model, 'base_model')}")

        if hasattr(model, 'base_model'):
            base_model = model.base_model
            print(f"  base_model type: {type(base_model)}")
            print(f"  hasattr(base_model, 'model'): {hasattr(base_model, 'model')}")

            if hasattr(base_model, 'model'):
                inner_model = base_model.model
                print(f"  base_model.model type: {type(inner_model)}")
                print(f"  hasattr(base_model.model, 'layers'): {hasattr(inner_model, 'layers')}")
                if hasattr(inner_model, 'layers'):
                    print(f"  len(base_model.model.layers): {len(inner_model.layers)}")
                    print(f"  layers[0] type: {type(inner_model.layers[0])}")

        # 様々なアクセス方法を試す
        print("\nTrying different access methods:")
        access_methods = [
            "model.model.layers",
            "model.base_model.model.layers",
            "model.base_model.layers",
            "model.layers"
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
                    if len(obj) > 0:
                        print(f"    First layer type: {type(obj[0])}")
                        print(f"    First layer attributes: {[attr for attr in dir(obj[0]) if not attr.startswith('_')][:5]}")
                else:
                    print(f"  {method}: SUCCESS")
            except AttributeError as e:
                print(f"  {method}: FAILED - {e}")

        # PEFT特有の属性をチェック
        print("\nPEFT specific attributes:")
        peft_attrs = [attr for attr in dir(model) if 'peft' in attr.lower() or 'lora' in attr.lower()]
        print(f"  PEFT related attributes: {peft_attrs}")

        if hasattr(model, 'peft_config'):
            print(f"  peft_config: {model.peft_config}")

        print("Detailed debug completed")

    except Exception as e:
        print(f"Failed to debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_peft_detailed()

