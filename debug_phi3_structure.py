#!/usr/bin/env python3
"""Phi3モデルの構造を詳細にデバッグ"""

from unsloth import FastLanguageModel

def debug_phi3_structure():
    """Phi3モデルの構造を詳細にデバッグ"""
    try:
        print("Loading Phi3 model to understand its structure...")

        model, tokenizer = FastLanguageModel.from_pretrained(
            'models/Borea-Phi-3.5-mini-Instruct-Jp',
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
            device_map='auto'
        )

        print("Phi3 model structure analysis:")
        print(f"  Model type: {type(model)}")
        print(f"  Model class: {model.__class__.__name__}")

        # Phi3ForCausalLMの属性を調べる
        attrs = [attr for attr in dir(model) if not attr.startswith('_')]
        print(f"  All attributes ({len(attrs)}): {attrs[:10]}...")

        # 主なコンポーネントをチェック
        components = ['model', 'base_model', 'layers', 'embed_tokens', 'norm', 'lm_head']
        for comp in components:
            if hasattr(model, comp):
                print(f"  hasattr({comp}): True, type: {type(getattr(model, comp))}")
                obj = getattr(model, comp)
                if hasattr(obj, '__len__'):
                    try:
                        print(f"    length: {len(obj)}")
                    except:
                        pass
            else:
                print(f"  hasattr({comp}): False")

        # model属性の詳細
        if hasattr(model, 'model'):
            inner_model = model.model
            print(f"  model attribute type: {type(inner_model)}")
            inner_attrs = [attr for attr in dir(inner_model) if not attr.startswith('_')]
            print(f"  model attributes ({len(inner_attrs)}): {inner_attrs[:15]}...")

            # レイヤー関連の属性を探す
            layer_attrs = [attr for attr in inner_attrs if 'layer' in attr.lower()]
            print(f"  Layer-related attributes: {layer_attrs}")

            # 各属性をチェック
            for attr in ['layers', 'encoder', 'decoder', 'embeddings', 'pooler']:
                if hasattr(inner_model, attr):
                    print(f"    hasattr(model.{attr}): True, type: {type(getattr(inner_model, attr))}")
                    obj = getattr(inner_model, attr)
                    if hasattr(obj, '__len__'):
                        try:
                            print(f"      length: {len(obj)}")
                            if len(obj) > 0:
                                print(f"      first item type: {type(obj[0])}")
                        except:
                            pass
                else:
                    print(f"    hasattr(model.{attr}): False")

        # Phi3Configを確認
        if hasattr(model, 'config'):
            config = model.config
            print(f"  Config type: {type(config)}")
            config_attrs = [attr for attr in dir(config) if not attr.startswith('_')]
            print(f"  Config attributes: {config_attrs[:10]}...")

            if hasattr(config, 'num_hidden_layers'):
                print(f"    num_hidden_layers: {config.num_hidden_layers}")
            if hasattr(config, 'num_attention_heads'):
                print(f"    num_attention_heads: {config.num_attention_heads}")

        print("Phi3 structure analysis completed")

    except Exception as e:
        print(f"Failed to debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_phi3_structure()

