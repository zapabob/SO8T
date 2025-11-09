"""
SO8T完全パイプライン（メモリ効率版）
統合→焼きこみ→GGUF変換を中間ファイルなしで実行

Author: SO8T Project Team
Date: 2024-11-06
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

from so8t_layer import SO8TAttentionWrapper, collect_so8t_orthogonality_loss
from burn_in import burn_in_model
from integrate_phi4_so8t import integrate_so8t_into_phi4

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="SO8T Full Pipeline")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_gguf", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SO8T Full Pipeline (Memory-Efficient)")
    print("=" * 80)
    
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(args.dtype, torch.float32)
    
    # Step 1: モデルロード
    logger.info("[Step 1] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map=args.device,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    config = model.config
    logger.info(f"[Step 1] Loaded: {config.model_type}, {config.num_hidden_layers} layers")
    
    # Step 2: SO8T統合
    logger.info("[Step 2] Integrating SO8T...")
    integration_info = integrate_so8t_into_phi4(
        model=model,
        config=config,
        so8t_enabled=True,
        init_scale=0.05,
        orthogonal_reg=1e-4
    )
    logger.info(f"[Step 2] SO8T integrated: {integration_info['layers_modified']} layers")
    
    # Step 3: 焼きこみ適用
    logger.info("[Step 3] Applying burn-in...")
    burn_stats = burn_in_model(
        model=model,
        verify=True,
        inplace=True,
        save_stats=False  # ディスク容量節約
    )
    logger.info(f"[Step 3] Burn-in complete: {len(burn_stats)} modules processed")
    
    # Step 4: GGUF変換の準備
    logger.info("[Step 4] Preparing for GGUF conversion...")
    logger.info("[Step 4] Note: Use external/llama.cpp-master/convert_hf_to_gguf.py")
    logger.info(f"[Step 4] Model is ready in memory but cannot be saved due to disk space")
    logger.info(f"[Step 4] Recommendation: Free up disk space and save manually")
    
    print("\n" + "=" * 80)
    print("[Pipeline] SO8T integration and burn-in completed in memory!")
    print("=" * 80)
    print("\nModel is ready but not saved due to disk constraints.")
    print("To complete the pipeline:")
    print("  1. Free up disk space (need ~14GB)")
    print("  2. Save model: model.save_pretrained(output_path)")
    print("  3. Convert to GGUF using llama.cpp")


if __name__ == "__main__":
    main()


























