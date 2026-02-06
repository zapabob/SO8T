#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS v2.5 LoRAアダプタをベースモデルとマージして完全なSafeTensorモデルを作成
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_full_safetensors():
    """LoRAアダプタをマージして完全なSafeTensorモデルを作成"""

    # モデルパス
    base_model_path = "AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp"  # 正しいベースモデル
    adapter_path = "models/aegis_v25_final"
    output_path = "models/aegis_v25_full_safetensors"

    try:
        logger.info("Loading base model and tokenizer...")

        # トークナイザーの読み込み
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)

        # ベースモデルの読み込み
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        logger.info("Loading LoRA adapter...")

        # LoRAアダプタの読み込み
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        logger.info("Merging LoRA adapter with base model...")

        # LoRAアダプタをマージ
        merged_model = model.merge_and_unload()

        logger.info(f"Saving full model to {output_path}...")

        # 出力ディレクトリの作成
        os.makedirs(output_path, exist_ok=True)

        # モデルの保存
        merged_model.save_pretrained(
            output_path,
            safe_serialization=True,
            max_shard_size="2GB"
        )

        # トークナイザーの保存
        tokenizer.save_pretrained(output_path)

        logger.info("[OK] Full SafeTensor model created successfully!")

        # モデルサイズの確認
        total_size = 0
        for file_path in os.listdir(output_path):
            if file_path.endswith('.safetensors'):
                file_size = os.path.getsize(os.path.join(output_path, file_path))
                total_size += file_size
                logger.info(f"Shard: {file_path}, Size: {file_size / (1024**3):.2f} GB")

        logger.info(f"Total model size: {total_size / (1024**3):.2f} GB")

        return output_path

    except Exception as e:
        logger.error(f"Failed to create full SafeTensor model: {e}")
        raise

if __name__ == "__main__":
    output_path = create_full_safetensors()
    print(f"Full SafeTensor model created at: {output_path}")