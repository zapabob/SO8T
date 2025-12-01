#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO(8) Compatible LoRA Adapter Merger
NKAT理論に基づくSO(8)残差アダプターをベースモデルに永続的に統合

このスクリプトは以下の処理を行います：
1. ベースモデルをCPUオフロードで読み込み
2. SO(8)アダプターを読み込み
3. 重みをマージ (W_new = W_base + α · (W_up × R_SO8 × W_down))
4. マージ済みモデルをsafetensors形式で保存

ハードウェア制約: RTX 3060 (12GB VRAM) + 32GB RAM
CPUオフロードを積極的に活用し、VRAM不足を防ぐ
"""

import os
import torch
import gc
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import argparse

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_base_model(base_model_path: str, device_map: str = "cpu") -> torch.nn.Module:
    """
    ベースモデルをCPUオフロードで読み込み

    Args:
        base_model_path: ベースモデルのパス
        device_map: デバイス配置 ("cpu" または "auto")

    Returns:
        読み込まれたモデル
    """
    from transformers import AutoModelForCausalLM

    logger.info(f"Loading base model from {base_model_path} with device_map={device_map}")

    # CPUオフロード設定
    if device_map == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="cpu",  # 明示的にCPUに配置
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    else:
        # autoの場合もCPUオフロードを有効化
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory={0: "12GB", "cpu": "32GB"},  # VRAM 12GB, RAM 32GB
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            offload_folder="./offload"  # オフロード用一時フォルダ
        )

    logger.info(f"Base model loaded successfully. Model type: {type(model)}")
    return model


def load_tokenizer(base_model_path: str):
    """トークナイザーを読み込み"""
    from transformers import AutoTokenizer

    logger.info(f"Loading tokenizer from {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    return tokenizer


def load_and_merge_adapter(base_model: torch.nn.Module, adapter_path: str) -> torch.nn.Module:
    """
    SO(8)アダプターを読み込み、マージする

    Args:
        base_model: ベースモデル
        adapter_path: アダプターパス

    Returns:
        マージ済みモデル
    """
    from peft import PeftModel

    logger.info(f"Loading SO(8) adapter from {adapter_path}")

    # アダプター読み込み
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.float16,
        device_map="cpu"  # CPUで処理
    )

    # メモリ解放
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("Merging LoRA weights... This may take a while.")

    # マージ実行 (W_new = W_base + α · (W_up × R_SO8 × W_down))
    merged_model = model.merge_and_unload()

    # メモリ解放
    del model
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("SO(8) adapter merged successfully")
    return merged_model


def save_merged_model(
    model: torch.nn.Module,
    tokenizer,
    output_dir: str,
    max_shard_size: str = "2GB"
):
    """
    マージ済みモデルをsafetensors形式で保存

    Args:
        model: マージ済みモデル
        tokenizer: トークナイザー
        output_dir: 保存先ディレクトリ
        max_shard_size: シャード最大サイズ
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving merged model to {output_dir} with max_shard_size={max_shard_size}")

    # モデル保存 (safetensors形式)
    model.save_pretrained(
        output_path,
        safe_serialization=True,  # safetensors形式
        max_shard_size=max_shard_size
    )

    # トークナイザー保存
    tokenizer.save_pretrained(output_path)

    # 設定ファイル更新
    config_path = output_path / "config.json"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # マージ済みであることを示すメタデータ追加
        config["_merged_with_so8_adapter"] = True
        config["_merge_date"] = torch.datetime.datetime.now().isoformat()

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info(f"Merged model saved successfully to {output_dir}")


def validate_paths(base_model: str, adapter_path: str, output_dir: str):
    """パス検証"""
    # ベースモデル
    if not os.path.exists(base_model) and not base_model.startswith(("microsoft/", "meta-llama/", "Borea-")):
        raise FileNotFoundError(f"Base model path not found: {base_model}")

    # アダプターパス
    adapter_config = Path(adapter_path) / "adapter_config.json"
    if not adapter_config.exists():
        raise FileNotFoundError(f"Adapter config not found: {adapter_config}")

    # 出力ディレクトリ
    output_path = Path(output_dir)
    if output_path.exists() and list(output_path.glob("*")):
        logger.warning(f"Output directory {output_dir} is not empty. Files may be overwritten.")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="SO(8) Compatible LoRA Adapter Merger",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python scripts/utils/merge_so8_adapter.py \\
    --base_model models/Borea-Phi-3.5-mini-Instruct-Jp \\
    --adapter_path outputs/so8_adapter/final_adapter \\
    --output_dir outputs/merged_so8_model

Hardware constraints: RTX 3060 (12GB VRAM) + 32GB RAM
Uses CPU offloading to prevent VRAM overflow.
        """
    )

    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Path to base model (Hugging Face ID or local path)"
    )

    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to trained SO(8) adapter directory"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for merged model"
    )

    parser.add_argument(
        "--device_map",
        type=str,
        default="cpu",
        choices=["cpu", "auto"],
        help="Device mapping strategy (default: cpu for safety)"
    )

    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="2GB",
        help="Maximum shard size for safetensors (default: 2GB)"
    )

    args = parser.parse_args()

    try:
        logger.info("=" * 60)
        logger.info("SO(8) Compatible LoRA Adapter Merger Started")
        logger.info("=" * 60)

        # パス検証
        validate_paths(args.base_model, args.adapter_path, args.output_dir)

        # 初期メモリ状態確認
        if torch.cuda.is_available():
            logger.info(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB used")

        # 1. ベースモデル読み込み (CPUオフロード)
        logger.info("Step 1: Loading base model with CPU offloading...")
        base_model = load_base_model(args.base_model, args.device_map)

        # メモリチェック
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"After base model load - GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB used")

        # 2. トークナイザー読み込み
        logger.info("Step 2: Loading tokenizer...")
        tokenizer = load_tokenizer(args.base_model)

        # 3. SO(8)アダプター読み込みとマージ
        logger.info("Step 3: Loading and merging SO(8) adapter...")
        merged_model = load_and_merge_adapter(base_model, args.adapter_path)

        # メモリチェック
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"After merge - GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB used")

        # 4. マージ済みモデル保存
        logger.info("Step 4: Saving merged model...")
        save_merged_model(
            merged_model,
            tokenizer,
            args.output_dir,
            args.max_shard_size
        )

        # 最終メモリ解放
        del merged_model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("=" * 60)
        logger.info("SO(8) Adapter merge completed successfully!")
        logger.info(f"Merged model saved to: {args.output_dir}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Merge failed: {e}")
        # エラー時のメモリ解放
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise


if __name__ == "__main__":
    main()
