#!/usr/bin/env python3
"""Unslothモデルの詳細な構造をデバッグ"""

import torch
from unsloth import FastLanguageModel

def debug_model_structure():
    """Unslothモデルの構造を詳細にデバッグ"""
    try:
        print("Loading model with Unsloth for detailed debugging...")

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

        # 主要な属性をチェック
        print("\n=== Main Attributes ===")
        main_attrs = ['model', 'layers', 'embed_tokens', 'norm', 'lm_head']
        for attr in main_attrs:
            exists = hasattr(model, attr)
            print(f"hasattr(model, '{attr}'): {exists}")
            if exists:
                try:
                    value = getattr(model, attr)
                    print(f"  type({attr}): {type(value)}")
                    if hasattr(value, '__len__') and not isinstance(value, str):
                        try:
                            print(f"  len({attr}): {len(value)}")
                        except:
                            pass
                except Exception as e:
                    print(f"  Error accessing {attr}: {e}")

        # model属性がある場合
        if hasattr(model, 'model'):
            print("\n=== model.model Attributes ===")
            model_attrs = ['layers', 'embed_tokens', 'norm', 'lm_head']
            for attr in model_attrs:
                exists = hasattr(model.model, attr)
                print(f"hasattr(model.model, '{attr}'): {exists}")
                if exists:
                    try:
                        value = getattr(model.model, attr)
                        print(f"  type(model.model.{attr}): {type(value)}")
                        if hasattr(value, '__len__') and not isinstance(value, str):
                            try:
                                print(f"  len(model.model.{attr}): {len(value)}")
                            except:
                                pass
                    except Exception as e:
                        print(f"  Error accessing model.model.{attr}: {e}")

            # layersがある場合、最初の層を確認
            if hasattr(model.model, 'layers'):
                print("\n=== First Layer Analysis ===")
                first_layer = model.model.layers[0]
                print(f"First layer type: {type(first_layer)}")
                print(f"First layer class: {first_layer.__class__.__name__}")

                layer_attrs = ['self_attn', 'mlp', 'input_layernorm', 'post_attention_layernorm']
                for attr in layer_attrs:
                    exists = hasattr(first_layer, attr)
                    print(f"hasattr(first_layer, '{attr}'): {exists}")

        # dir()で全ての属性を確認
        print("\n=== All model attributes (first 20) ===")
        all_attrs = [attr for attr in dir(model) if not attr.startswith('_')]
        for attr in all_attrs[:20]:
            print(f"  {attr}")

        print("\n=== All model.model attributes (first 20) ===")
        if hasattr(model, 'model'):
            all_model_attrs = [attr for attr in dir(model.model) if not attr.startswith('_')]
            for attr in all_model_attrs[:20]:
                print(f"  {attr}")

        print("\nDebug completed successfully")

    except Exception as e:
        print(f"Failed to debug model structure: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model_structure()