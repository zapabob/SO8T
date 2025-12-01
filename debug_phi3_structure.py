#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phi3モデル構造デバッグスクリプト
"""

from unsloth import FastLanguageModel
import torch

def debug_phi3_structure():
    """Phi3モデルの構造をデバッグ"""
    print("=== Phi3 Model Structure Debug ===")

    try:
        # モデル読み込み
        print("Loading Phi-3.5-mini-instruct model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            'microsoft/Phi-3.5-mini-instruct',
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )

        print(f"Model type: {type(model)}")

        # トップレベル属性
        print("\n=== Top Level Attributes ===")
        top_attrs = [attr for attr in dir(model) if not attr.startswith('_') and not callable(getattr(model, attr))]
        print(f"Top level attributes: {top_attrs}")

        # base_model確認
        if hasattr(model, 'base_model'):
            print(f"\n=== base_model ===")
            print(f"base_model type: {type(model.base_model)}")

            base_attrs = [attr for attr in dir(model.base_model) if not attr.startswith('_') and not callable(getattr(model.base_model, attr))]
            print(f"base_model attributes: {base_attrs}")

            if hasattr(model.base_model, 'model'):
                print(f"\n=== base_model.model ===")
                print(f"base_model.model type: {type(model.base_model.model)}")

                base_model_attrs = [attr for attr in dir(model.base_model.model) if not attr.startswith('_') and not callable(getattr(model.base_model.model, attr))]
                print(f"base_model.model attributes: {base_model_attrs}")

                # layersを探す
                if hasattr(model.base_model.model, 'layers'):
                    print("
✅ Found layers in base_model.model.layers"                    print(f"Layers count: {len(model.base_model.model.layers)}")
                    print(f"First layer type: {type(model.base_model.model.layers[0])}")
                elif hasattr(model.base_model.model, 'model') and hasattr(model.base_model.model.model, 'layers'):
                    print("
✅ Found layers in base_model.model.model.layers"                    print(f"Layers count: {len(model.base_model.model.model.layers)}")
                    print(f"First layer type: {type(model.base_model.model.model.layers[0])}")
                else:
                    print("\n❌ Layers not found in expected locations")

                    # 再帰的に探す
                    def find_layers(obj, path=""):
                        if hasattr(obj, 'layers'):
                            print(f"✅ Found layers at: {path}.layers")
                            return True
                        for attr in dir(obj):
                            if not attr.startswith('_') and not callable(getattr(obj, attr)):
                                try:
                                    child = getattr(obj, attr)
                                    if hasattr(child, '__dict__') or hasattr(child, '__slots__'):
                                        if find_layers(child, f"{path}.{attr}"):
                                            return True
                                except:
                                    pass
                        return False

                    find_layers(model, "model")

        # 直接layers確認
        if hasattr(model, 'layers'):
            print(f"\n✅ Found layers directly in model.layers")
            print(f"Layers count: {len(model.layers)}")

    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_phi3_structure()