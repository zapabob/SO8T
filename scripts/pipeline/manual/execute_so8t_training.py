#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T学習実行スクリプト

評価済みデータセットを使用してSO8Tを再学習

Usage:
    python scripts/pipelines/execute_so8t_training.py --dataset D:\webdataset\processed\so8t_training\so8t_dataset_20251108_040442.jsonl
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings("ignore")

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "so8t-mmllm"))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import bitsandbytes as bnb
from tqdm import tqdm
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SO8TTrainingDataset(Dataset):
    """SO8T学習用データセット"""
    
    def __init__(self, data_path: Path, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        logger.info(f"Loading SO8T training dataset from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    self.samples.append(sample)
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Loaded {len(self.samples)} training samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        output = sample.get('output', '')
        
        # プロンプト構築
        if input_text:
            prompt = f"指示: {instruction}\n入力: {input_text}\n出力: {output}"
        else:
            prompt = f"指示: {instruction}\n出力: {output}"
        
        # トークナイズ
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0)  # 言語モデリング用
        }


def train_so8t_model(
    dataset_path: Path,
    base_model: str = "models/Borea-Phi-3.5-mini-Instruct-Jp",
    output_dir: Path = None,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    gradient_accumulation_steps: int = 4
) -> Path:
    """
    SO8Tモデル学習
    
    Args:
        dataset_path: 学習データセットパス
        base_model: ベースモデル名
        output_dir: 出力ディレクトリ
        batch_size: バッチサイズ
        learning_rate: 学習率
        num_epochs: エポック数
        gradient_accumulation_steps: 勾配累積ステップ数
    
    Returns:
        学習済みモデルのパス
    """
    logger.info("="*80)
    logger.info("SO8T Model Training")
    logger.info("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if output_dir is None:
        output_dir = Path(r"D:\webdataset\checkpoints\training") / datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # トークナイザー読み込み
    logger.info(f"Loading tokenizer from {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # データセット準備
    train_dataset = SO8TTrainingDataset(dataset_path, tokenizer)
    
    if len(train_dataset) == 0:
        logger.warning("[WARNING] No training samples available")
        return output_dir
    
    # モデル読み込み（8bit量子化）
    logger.info(f"Loading model from {base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True
    )
    
    # QLoRA設定
    logger.info("Setting up QLoRA...")
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    
    # 学習可能パラメータ表示
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # 学習設定
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        fp16=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_steps=10,
        report_to="none"  # MLflow/W&Bは別途設定
    )
    
    # データコレクター
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Trainer初期化
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )
    
    # 学習実行
    logger.info("Starting training...")
    trainer.train()
    
    # モデル保存
    logger.info("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"[OK] Model saved to {output_dir}")
    
    return output_dir


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Execute SO8T Training")
    parser.add_argument(
        '--dataset',
        type=Path,
        default=Path(r"D:\webdataset\processed\so8t_training\so8t_dataset_20251108_040442.jsonl"),
        help='Training dataset path'
    )
    parser.add_argument(
        '--base-model',
        type=str,
        default="models/Borea-Phi-3.5-mini-Instruct-Jp",
        help='Base model path'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output directory'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-5,
        help='Learning rate'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of epochs'
    )
    parser.add_argument(
        '--gradient-accumulation',
        type=int,
        default=4,
        help='Gradient accumulation steps'
    )
    
    args = parser.parse_args()
    
    # SO8T学習実行
    model_path = train_so8t_model(
        dataset_path=args.dataset,
        base_model=args.base_model,
        output_dir=args.output,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation
    )
    
    logger.info("="*80)
    logger.info("[COMPLETE] SO8T Training Finished")
    logger.info(f"Model saved to: {model_path}")
    logger.info("="*80)


if __name__ == '__main__':
    main()



