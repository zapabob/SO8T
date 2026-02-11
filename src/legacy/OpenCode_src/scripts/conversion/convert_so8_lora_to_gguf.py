#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO(8) LoRA to GGUF Converter
SO(8)アダプター学習済みモデルをGGUF形式に変換

このスクリプトは以下の処理を行います：
1. 標準LoRA形式のモデル読み込み
2. LoRA重みのマージ
3. GGUF形式への変換
"""

import os
import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import argparse
import subprocess
import sys

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_lora_model(lora_path: str):
    """LoRAモデルを読み込み"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    logger.info(f"Loading LoRA model from {lora_path}")

    # ベースモデルパスを取得（adapter_config.jsonから）
    config_path = Path(lora_path) / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {lora_path}")

    with open(config_path, 'r') as f:
        adapter_config = json.load(f)

    # adapter_configにbase_model_name_or_pathがない場合は手動指定
    if "base_model_name_or_path" not in adapter_config or not adapter_config["base_model_name_or_path"]:
        # デフォルトのPhi-3.5モデルを使用
        base_model_path = "microsoft/Phi-3.5-mini-instruct"
        logger.warning(f"Base model not specified in config, using default: {base_model_path}")
    else:
        base_model_path = adapter_config["base_model_name_or_path"]

    # ベースモデル読み込み
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="cpu",  # GGUF変換時はCPU使用
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )

    # LoRAアダプター読み込み
    if (Path(lora_path) / "adapter_model.safetensors").exists():
        model = PeftModel.from_pretrained(model, lora_path)
    elif (Path(lora_path) / "adapter_model.bin").exists():
        model = PeftModel.from_pretrained(model, lora_path)
    else:
        raise FileNotFoundError(f"No adapter model found in {lora_path}")

    # LoRAをマージ
    logger.info("Merging LoRA weights...")
    model = model.merge_and_unload()

    return model, tokenizer


def save_model_for_gguf(model, tokenizer, output_path: str):
    """GGUF変換用のモデル保存"""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model for GGUF conversion to {output_path}")

    # モデル保存
    model.save_pretrained(output_dir)

    # トークナイザー保存
    tokenizer.save_pretrained(output_dir)

    # config.jsonの修正（GGUF変換に適した設定）
    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)

        # GGUF変換に適した設定を追加/修正
        config_updates = {
            "torch_dtype": "float16",
            "use_cache": True,
            "low_cpu_mem_usage": True,
        }

        config.update(config_updates)

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    logger.info(f"Model saved for GGUF conversion: {output_path}")


def convert_to_gguf(model_path: str, output_path: str, quantization: str = "bf16"):
    """GGUF形式に変換"""
    logger.info(f"Converting to GGUF: {model_path} -> {output_path}")

    # llama.cpp convert_hf_to_gguf.pyを使用
    convert_script = Path(__file__).parent.parent / "external" / "llama.cpp-master" / "convert_hf_to_gguf.py"

    if not convert_script.exists():
        raise FileNotFoundError(f"convert_hf_to_gguf.py not found at {convert_script}")

    # 量子化タイプのマッピング
    quant_map = {
        "f16": "f16",
        "bf16": "bf16",
        "f32": "f32",
        "q8_0": "q8_0",
        "q4_k_m": "q4_k_m",
        "q4_0": "q4_0",
    }

    if quantization not in quant_map:
        raise ValueError(f"Unsupported quantization: {quantization}")

    gguf_type = quant_map[quantization]

    cmd = [
        sys.executable, str(convert_script),
        model_path,
        "--outfile", output_path,
        "--outtype", gguf_type
    ]

    logger.info(f"Running command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)

    if result.returncode == 0:
        logger.info(f"GGUF conversion completed: {output_path}")
        return True
    else:
        logger.error(f"GGUF conversion failed: {result.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="SO(8) LoRA to GGUF Converter")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA model directory")
    parser.add_argument("--output_path", type=str, required=True, help="Output GGUF file path")
    parser.add_argument("--quantization", type=str, default="bf16",
                       choices=["f16", "bf16", "f32", "q8_0", "q4_k_m", "q4_0"],
                       help="Quantization type for GGUF")

    args = parser.parse_args()

    try:
        # LoRAモデル読み込みとマージ
        logger.info("Loading and merging LoRA model...")
        model, tokenizer = load_lora_model(args.lora_path)

        # GGUF変換用の一時ディレクトリ
        temp_model_dir = Path(args.output_path).parent / "temp_model_for_gguf"
        temp_model_dir.mkdir(exist_ok=True)

        # マージ済みモデル保存
        save_model_for_gguf(model, tokenizer, str(temp_model_dir))

        # GGUF変換
        success = convert_to_gguf(str(temp_model_dir), args.output_path, args.quantization)

        if success:
            logger.info(f"SO(8) LoRA to GGUF conversion completed successfully!")
            logger.info(f"Output: {args.output_path}")
        else:
            logger.error("GGUF conversion failed")
            sys.exit(1)

        # 一時ディレクトリ削除
        import shutil
        shutil.rmtree(temp_model_dir, ignore_errors=True)
        logger.info("Cleaned up temporary files")

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
