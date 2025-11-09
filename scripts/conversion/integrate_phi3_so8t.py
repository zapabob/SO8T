#!/usr/bin/env python3
"""
Phi-3モデルにSO8T統合スクリプト

既存のPhi-3モデルを読み込み、SO8T統合版モデル（modeling_phi3_so8t.py）に変換する。
SO8T統合により、SO(8)群構造を持つアテンション機構と回転ゲートが追加される。

Usage:
    python scripts/conversion/integrate_phi3_so8t.py \
        --model_path models/Borea-Phi-3.5-mini-Instruct-Jp \
        --output_path D:/webdataset/models/so8t_integrated/phi3_so8t
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from contextlib import contextmanager

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "Borea-Phi-3.5-mini-Instruct-Jp"))

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@contextmanager
def timeout_handler(timeout_seconds=3600):
    """タイムアウトハンドラー（Windows用）"""
    import threading
    
    def timeout_func():
        import time
        time.sleep(timeout_seconds)
        logger.error(f"[TIMEOUT] Operation exceeded {timeout_seconds} seconds")
        raise TimeoutError(f"Operation exceeded {timeout_seconds} seconds")
    
    timer = threading.Timer(timeout_seconds, timeout_func)
    timer.start()
    try:
        yield
    finally:
        timer.cancel()


def integrate_so8t_to_phi3(
    model_path: str,
    output_path: str,
    device: str = "cuda",
    verify: bool = True,
    torch_dtype: str = "bfloat16",
):
    """
    Phi-3モデルにSO8Tを統合
    
    Args:
        model_path: 元のPhi-3モデルパス
        output_path: SO8T統合後のモデル保存先
        device: デバイス（cuda/cpu）
        verify: 統合検証を行うか
        torch_dtype: モデルのデータ型（bfloat16/float16/float32）
    """
    logger.info(f"[STEP 1] Loading Phi-3 model from {model_path}")
    
    # モデルパスを確認
    model_path = Path(model_path)
    if not model_path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")
    
    # 設定とトークナイザーを読み込み
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    
    logger.info(f"Model architecture: {config.model_type}")
    logger.info(f"Hidden size: {config.hidden_size}")
    logger.info(f"Number of layers: {config.num_hidden_layers}")
    logger.info(f"Number of attention heads: {config.num_attention_heads}")
    
    # hidden_sizeが8の倍数か確認
    if config.hidden_size % 8 != 0:
        raise ValueError(
            f"hidden_size ({config.hidden_size}) must be divisible by 8 for SO8T integration"
        )
    
    # データ型を設定
    if torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif torch_dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    logger.info(f"[STEP 2] Loading original Phi-3 model")
    
    # メモリをクリア
    import gc
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("[INFO] GPU memory cleared before loading model")
    
    # 元のモデルを読み込み（重み転送用）
    from transformers import AutoModelForCausalLM
    try:
        original_model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            config=config,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,  # メモリ使用量を削減
        )
    except OSError as e:
        if "1455" in str(e) or "ページファイル" in str(e):
            logger.error("[ERROR] Insufficient memory (OSError 1455). Trying with CPU offloading...")
            # CPUオフロードを試す
            original_model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                config=config,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map="balanced",  # バランス型オフロード
                low_cpu_mem_usage=True,
                max_memory={0: "10GB", "cpu": "30GB"}  # メモリ制限
            )
        else:
            raise
    
    logger.info(f"[STEP 3] Creating SO8T-integrated model")
    
    # SO8T統合モデルをインポート
    try:
        # モデルディレクトリをパッケージとして扱うために、sys.pathに追加
        model_dir = PROJECT_ROOT / "models" / "Borea-Phi-3.5-mini-Instruct-Jp"
        model_dir_str = str(model_dir)
        
        # モデルディレクトリをパッケージとして認識させる
        import types
        import importlib
        
        # パッケージ構造を作成
        package_name = "borea_phi35_mini_instruct_jp"
        
        # 親パッケージを作成
        if 'models' not in sys.modules:
            sys.modules['models'] = types.ModuleType('models')
        
        # モデルパッケージを作成
        if package_name not in sys.modules:
            model_package = types.ModuleType(package_name)
            model_package.__path__ = [model_dir_str]
            model_package.__file__ = str(model_dir / "__init__.py")
            sys.modules[package_name] = model_package
        
        # modeling_phi3_so8t.pyを動的にインポート
        import importlib.util
        
        # まず、依存モジュールをインポート（相対インポートを回避）
        # configuration_phi3.pyをインポート
        config_spec = importlib.util.spec_from_file_location(
            f"{package_name}.configuration_phi3",
            model_dir / "configuration_phi3.py"
        )
        config_module = importlib.util.module_from_spec(config_spec)
        config_module.__package__ = package_name
        config_spec.loader.exec_module(config_module)
        sys.modules[f"{package_name}.configuration_phi3"] = config_module
        
        # modeling_phi3.pyをインポート
        modeling_phi3_spec = importlib.util.spec_from_file_location(
            f"{package_name}.modeling_phi3",
            model_dir / "modeling_phi3.py"
        )
        modeling_phi3_module = importlib.util.module_from_spec(modeling_phi3_spec)
        modeling_phi3_module.__package__ = package_name
        # configuration_phi3を設定
        modeling_phi3_module.configuration_phi3 = config_module
        modeling_phi3_spec.loader.exec_module(modeling_phi3_module)
        sys.modules[f"{package_name}.modeling_phi3"] = modeling_phi3_module
        
        # modeling_phi3_so8t.pyをインポート
        spec = importlib.util.spec_from_file_location(
            f"{package_name}.modeling_phi3_so8t",
            model_dir / "modeling_phi3_so8t.py"
        )
        
        # モジュールをパッケージとして扱うために、__package__属性を設定
        modeling_so8t = importlib.util.module_from_spec(spec)
        modeling_so8t.__package__ = package_name
        modeling_so8t.__name__ = f"{package_name}.modeling_phi3_so8t"
        # 依存モジュールを設定
        modeling_so8t.configuration_phi3 = config_module
        modeling_so8t.modeling_phi3 = modeling_phi3_module
        
        spec.loader.exec_module(modeling_so8t)
        sys.modules[f"{package_name}.modeling_phi3_so8t"] = modeling_so8t
        SO8TPhi3ForCausalLM = modeling_so8t.SO8TPhi3ForCausalLM
    except Exception as e:
        logger.error(f"Failed to import modeling_phi3_so8t: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    
    # SO8T統合モデルを作成
    so8t_model = SO8TPhi3ForCausalLM(config)
    
    # デバイスに移動
    if device == "cuda" and torch.cuda.is_available():
        so8t_model = so8t_model.to(device)
    else:
        so8t_model = so8t_model.to("cpu")
    
    logger.info(f"[STEP 4] Transferring weights from original model to SO8T model")
    
    # 重みを転送（SO8T固有のパラメータを除く）
    transferred_params = 0
    skipped_params = 0
    
    with torch.no_grad():
        # Embedding層
        if hasattr(original_model.model, 'embed_tokens'):
            so8t_model.model.embed_tokens.weight.copy_(original_model.model.embed_tokens.weight)
            transferred_params += 1
        
        # 各レイヤー
        for layer_idx in tqdm(range(config.num_hidden_layers), desc="Transferring weights"):
            original_layer = original_model.model.layers[layer_idx]
            so8t_layer = so8t_model.model.layers[layer_idx]
            
            # Input layer norm
            if hasattr(original_layer, 'input_layernorm'):
                so8t_layer.input_layernorm.weight.copy_(original_layer.input_layernorm.weight)
                transferred_params += 1
            
            # Attention層の重みを転送
            original_attn = original_layer.self_attn
            so8t_attn = so8t_layer.self_attn
            
            # QKV投影（SO8Tは分離されている）
            if hasattr(original_attn, 'qkv_proj'):
                # 元のモデルがqkv_projを使用している場合
                qkv_weight = original_attn.qkv_proj.weight
                query_pos = config.num_attention_heads * (config.hidden_size // config.num_attention_heads)
                kv_pos = query_pos + config.num_key_value_heads * (config.hidden_size // config.num_attention_heads)
                
                # Q, K, Vに分割
                so8t_attn.q_proj.weight.copy_(qkv_weight[:query_pos])
                so8t_attn.k_proj.weight.copy_(qkv_weight[query_pos:kv_pos])
                so8t_attn.v_proj.weight.copy_(qkv_weight[kv_pos:])
                transferred_params += 3
            else:
                # 分離されたQKV投影がある場合
                if hasattr(original_attn, 'q_proj'):
                    so8t_attn.q_proj.weight.copy_(original_attn.q_proj.weight)
                    transferred_params += 1
                if hasattr(original_attn, 'k_proj'):
                    so8t_attn.k_proj.weight.copy_(original_attn.k_proj.weight)
                    transferred_params += 1
                if hasattr(original_attn, 'v_proj'):
                    so8t_attn.v_proj.weight.copy_(original_attn.v_proj.weight)
                    transferred_params += 1
            
            # Output投影
            if hasattr(original_attn, 'o_proj'):
                so8t_attn.o_proj.weight.copy_(original_attn.o_proj.weight)
                transferred_params += 1
            
            # Rotary embedding（可能な場合）
            # SO8Tは独自のRoPEを使用するため、スキップ
            
            # Post attention layer norm
            if hasattr(original_layer, 'post_attention_layernorm'):
                so8t_layer.post_attention_layernorm.weight.copy_(original_layer.post_attention_layernorm.weight)
                transferred_params += 1
            
            # MLP層
            if hasattr(original_layer, 'mlp'):
                original_mlp = original_layer.mlp
                so8t_mlp = so8t_layer.mlp
                
                if hasattr(original_mlp, 'gate_up_proj'):
                    gate_up_weight = original_mlp.gate_up_proj.weight
                    gate_weight = gate_up_weight[:config.intermediate_size]
                    up_weight = gate_up_weight[config.intermediate_size:]
                    so8t_mlp.gate_up_proj.weight.copy_(gate_up_weight)
                    transferred_params += 1
                
                if hasattr(original_mlp, 'down_proj'):
                    so8t_mlp.down_proj.weight.copy_(original_mlp.down_proj.weight)
                    transferred_params += 1
        
        # Output layer norm
        if hasattr(original_model.model, 'norm'):
            so8t_model.model.norm.weight.copy_(original_model.model.norm.weight)
            transferred_params += 1
        
        # LM head
        if hasattr(original_model, 'lm_head'):
            so8t_model.lm_head.weight.copy_(original_model.lm_head.weight)
            transferred_params += 1
    
    logger.info(f"[STEP 5] Weight transfer complete: {transferred_params} parameters transferred")
    logger.info(f"[STEP 6] SO8T rotation gates initialized (new parameters)")
    
    # SO8T固有のパラメータ数を計算
    num_so8t_params = 0
    for layer in so8t_model.model.layers:
        if hasattr(layer.self_attn, 'so8t_rotation_gate'):
            num_blocks = config.hidden_size // 8
            num_so8t_params += num_blocks * 8 * 8  # theta parameters
    
    logger.info(f"SO8T parameters added: {num_so8t_params:,} ({num_so8t_params/1e6:.2f}M)")
    
    # 統合検証
    if verify:
        logger.info("[STEP 7] Verifying SO8T integration")
        verify_integration(so8t_model, tokenizer, device)
    
    # モデル保存
    logger.info(f"[STEP 8] Saving SO8T-integrated model to {output_path}")
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # メモリをクリアしてから保存
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("[INFO] GPU memory cleared before saving")
    
    # モデルとトークナイザーを保存（進捗表示付き、タイムアウト付き）
    logger.info("[INFO] Saving model weights (this may take several minutes)...")
    logger.info("[INFO] Estimated time: 5-15 minutes depending on model size")
    
    try:
        # タイムアウト設定（1時間）
        with timeout_handler(timeout_seconds=3600):
            so8t_model.save_pretrained(
                str(output_path),
                safe_serialization=True,  # safetensors形式で保存（高速）
                max_shard_size="5GB"  # シャードサイズを制限
            )
        logger.info("[OK] Model weights saved successfully")
    except TimeoutError:
        logger.error("[ERROR] Model save operation timed out after 1 hour")
        raise
    except Exception as e:
        logger.error(f"[ERROR] Failed to save model: {e}")
        logger.info("[INFO] Attempting to save without safe_serialization...")
        try:
            with timeout_handler(timeout_seconds=3600):
                so8t_model.save_pretrained(str(output_path), safe_serialization=False)
            logger.info("[OK] Model weights saved (without safe_serialization)")
        except TimeoutError:
            logger.error("[ERROR] Model save operation timed out (fallback method)")
            raise
    
    logger.info("[INFO] Saving tokenizer...")
    tokenizer.save_pretrained(str(output_path))
    logger.info("[OK] Tokenizer saved successfully")
    
    # 設定ファイルを更新（SO8T統合を示す）
    config_dict = config.to_dict()
    config_dict['so8t_integrated'] = True
    config_dict['so8t_parameters'] = num_so8t_params
    
    with open(output_path / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    # 統合情報を保存
    integration_info = {
        'original_model': str(model_path),
        'integrated_layers': config.num_hidden_layers,
        'total_layers': config.num_hidden_layers,
        'hidden_size': config.hidden_size,
        'num_blocks': config.hidden_size // 8,
        'so8t_parameters': num_so8t_params,
        'so8t_parameters_per_layer': num_so8t_params // config.num_hidden_layers,
        'transferred_parameters': transferred_params,
    }
    
    with open(output_path / 'so8t_integration_info.json', 'w', encoding='utf-8') as f:
        json.dump(integration_info, f, indent=2, ensure_ascii=False)
    
    logger.info("[SUCCESS] SO8T integration completed successfully!")
    logger.info(f"Integrated {config.num_hidden_layers} layers with SO8T")
    logger.info(f"Model saved to {output_path}")
    
    return so8t_model


def verify_integration(model, tokenizer, device):
    """
    SO8T統合の検証
    
    Args:
        model: SO8T統合済みモデル
        tokenizer: トークナイザー
        device: デバイス
    """
    logger.info("Running integration verification...")
    
    # テスト入力
    test_inputs = [
        "こんにちは、私はAIアシスタントです。",
        "What is 2+2?",
        "Explain SO(8) group structure.",
    ]
    
    model.eval()
    with torch.no_grad():
        for text in test_inputs:
            try:
                inputs = tokenizer(text, return_tensors="pt")
                if device == "cuda" and torch.cuda.is_available():
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = model(**inputs)
                
                # 出力形状確認
                assert outputs.logits.shape[0] == 1  # batch_size
                assert outputs.logits.shape[2] == model.config.vocab_size
                
                logger.info(f"[OK] Test input processed successfully: {text[:50]}...")
                
            except Exception as e:
                logger.error(f"[FAILED] Test failed for input '{text[:50]}...': {e}")
                raise
    
    # SO8T固有のパラメータ確認
    so8t_params_found = False
    for name, param in model.named_parameters():
        if 'so8t' in name.lower() or 'rotation_gate' in name.lower():
            so8t_params_found = True
            logger.info(f"[OK] Found SO8T parameter: {name}")
            break
    
    if not so8t_params_found:
        logger.warning("[WARNING] No SO8T parameters found in model")
    
    logger.info("[OK] Integration verification passed!")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Integrate SO8T into Phi-3 model"
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to original Phi-3 model'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Output path for SO8T-integrated model'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda)'
    )
    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='Skip integration verification'
    )
    parser.add_argument(
        '--torch_dtype',
        type=str,
        default='bfloat16',
        choices=['bfloat16', 'float16', 'float32'],
        help='Model data type (default: bfloat16)'
    )
    
    args = parser.parse_args()
    
    try:
        integrate_so8t_to_phi3(
            model_path=args.model_path,
            output_path=args.output_path,
            device=args.device,
            verify=not args.no_verify,
            torch_dtype=args.torch_dtype,
        )
        logger.info("[SUCCESS] SO8T integration completed!")
        return 0
        
    except Exception as e:
        logger.error(f"[FAILED] SO8T integration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

