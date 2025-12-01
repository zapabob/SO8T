#!/usr/bin/env python3
"""PEFTモデルの構造をデバッグ"""

import torch
from unsloth import FastLanguageModel

def debug_peft_model_structure():
    """PEFTモデルの構造をデバッグ"""
    try:
        print("Loading PEFT model with Unsloth...")

        model_path = "models/Borea-Phi-3.5-mini-Instruct-Jp"

        # Unslothでモデルをロード
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=torch.float16,
            load_in_4bit=True,
            device_map="auto",
        )

        print(f"Model type: {type(model)}")
        print(f"Model class: {model.__class__.__name__}")
        print(f"Model MRO: {[cls.__name__ for cls in model.__class__.__mro__]}")

        # PEFT特有の属性を確認
        print("\nPEFT specific attributes:")
        print(f"  hasattr(model, 'base_model'): {hasattr(model, 'base_model')}")

        if hasattr(model, 'base_model'):
            print(f"  base_model type: {type(model.base_model)}")
            print(f"  hasattr(base_model, 'model'): {hasattr(model.base_model, 'model')}")
            if hasattr(model.base_model, 'model'):
                print(f"  hasattr(base_model.model, 'layers'): {hasattr(model.base_model.model, 'layers')}")
                if hasattr(model.base_model.model, 'layers'):
                    print(f"  len(base_model.model.layers): {len(model.base_model.model.layers)}")

        # model.modelの確認
        if hasattr(model, 'model'):
            print(f"  model.model type: {type(model.model)}")
            print(f"  hasattr(model.model, 'layers'): {hasattr(model.model, 'layers')}")
            if hasattr(model.model, 'layers'):
                print(f"  len(model.model.layers): {len(model.model.layers)}")

        # 利用可能な層アクセス方法を探す
        print("\nTrying different layer access methods:")

        # 方法1: model.base_model.model.layers
        try:
            if hasattr(model, 'base_model') and hasattr(model.base_model, 'model') and hasattr(model.base_model.model, 'layers'):
                layers = model.base_model.model.layers
                print(f"  model.base_model.model.layers: SUCCESS, length={len(layers)}")
                print(f"    Type of first layer: {type(layers[0])}")
            else:
                print("  model.base_model.model.layers: FAILED")
        except Exception as e:
            print(f"  model.base_model.model.layers: ERROR - {e}")

        # 方法2: model.model.layers (現在のコード)
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                layers = model.model.layers
                print(f"  model.model.layers: SUCCESS, length={len(layers)}")
            else:
                print("  model.model.layers: FAILED")
        except Exception as e:
            print(f"  model.model.layers: ERROR - {e}")

        # 方法3: 直接model.layers
        try:
            if hasattr(model, 'layers'):
                layers = model.layers
                print(f"  model.layers: SUCCESS, length={len(layers)}")
            else:
                print("  model.layers: FAILED")
        except Exception as e:
            print(f"  model.layers: ERROR - {e}")

        # 方法4: getattrで探す
        try:
            # 再帰的にlayersを探す
            def find_layers(obj, path=""):
                if hasattr(obj, 'layers'):
                    print(f"  Found layers at: {path}.layers, length={len(obj.layers)}")
                    return obj.layers
                for attr in dir(obj):
                    if not attr.startswith('_') and attr not in ['parameters', 'named_parameters', 'children', 'modules']:
                        try:
                            value = getattr(obj, attr)
                            if hasattr(value, '__len__') and len(value) > 10:  # レイヤー配列のようなもの
                                result = find_layers(value, f"{path}.{attr}")
                                if result is not None:
                                    return result
                        except:
                            pass
                return None

            layers = find_layers(model, "model")
            if layers:
                print("  Recursive search: SUCCESS")
            else:
                print("  Recursive search: FAILED")

        except Exception as e:
            print(f"  Recursive search: ERROR - {e}")

        print("Debug completed successfully")

    except Exception as e:
        print(f"Failed to debug PEFT model structure: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_peft_model_structure()





