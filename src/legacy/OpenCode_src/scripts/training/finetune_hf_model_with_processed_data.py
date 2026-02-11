#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
収集・加工済みデータでHugging Faceモデルをfine-tuning

四値分類データでHugging Faceモデル（Qwen2.5/Phi-3.5）をfine-tuningし、
SO8Tで使用可能な形式で保存
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
import yaml
from tqdm import tqdm
from datasets import load_from_disk

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/finetune_hf_with_processed_data.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# チェックポイント設定
CHECKPOINT_INTERVAL = 300  # 5分間隔
MAX_CHECKPOINTS = 10


class ProcessedDataDataset(Dataset):
    """収集・加工済みデータセット"""
    
    def __init__(self, dataset_path: Path, tokenizer, max_length: int = 2048, split: str = "train"):
        """
        Args:
            dataset_path: Hugging Face Datasetディレクトリパス
            tokenizer: トークナイザー
            max_length: 最大シーケンス長
            split: データセット分割名
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"Loading dataset from {dataset_path} (split={split})...")
        
        # Hugging Face Dataset読み込み
        try:
            dataset_dict = load_from_disk(str(dataset_path))
            if split in dataset_dict:
                self.dataset = dataset_dict[split]
            else:
                # 分割がない場合はtrainを使用
                self.dataset = dataset_dict.get("train", list(dataset_dict.values())[0])
            
            logger.info(f"[OK] Loaded {len(self.dataset):,} samples")
        except Exception as e:
            logger.error(f"[ERROR] Failed to load dataset: {e}")
            raise
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # 既にトークナイズされている場合はそのまま返す
        if "input_ids" in sample:
            return {
                "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(sample["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(sample.get("labels", sample["input_ids"]), dtype=torch.long)
            }
        
        # トークナイズされていない場合はトークナイズ
        text = sample.get("text", "")
        if not text:
            # instruction + output形式
            instruction = sample.get("instruction", "")
            output = sample.get("output", "")
            text = f"{instruction}\n\n{output}" if instruction else output
        
        # トークナイズ
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": encoded["input_ids"].squeeze()
        }


class PowerFailureRecovery:
    """電源断リカバリーシステム"""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.last_checkpoint_time = time.time()
    
    def should_save_checkpoint(self) -> bool:
        """チェックポイント保存すべきか"""
        return time.time() - self.last_checkpoint_time >= CHECKPOINT_INTERVAL
    
    def save_checkpoint(self, trainer: Trainer, epoch: int, step: int):
        """チェックポイント保存"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}_step_{step}"
        trainer.save_model(str(checkpoint_path))
        self.last_checkpoint_time = time.time()
        logger.info(f"[CHECKPOINT] Saved to {checkpoint_path}")


class HFModelFinetuner:
    """Hugging FaceモデルFine-tuningクラス"""
    
    def __init__(
        self,
        base_model_name: str,
        dataset_path: Path,
        output_dir: Path,
        config: Optional[Dict] = None
    ):
        """
        Args:
            base_model_name: ベースモデル名
            dataset_path: データセットパス
            output_dir: 出力ディレクトリ
            config: 設定辞書
        """
        self.base_model_name = base_model_name
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        
        logger.info("="*80)
        logger.info("Hugging Face Model Finetuner Initialized")
        logger.info("="*80)
        logger.info(f"Base model: {base_model_name}")
        logger.info(f"Dataset: {dataset_path}")
        logger.info(f"Output: {output_dir}")
        
        # デバイス設定
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")
        
        # 電源断リカバリー
        self.recovery = PowerFailureRecovery(self.output_dir / "checkpoints")
    
    def load_model_and_tokenizer(self):
        """モデルとトークナイザーを読み込み"""
        logger.info(f"Loading model and tokenizer: {self.base_model_name}...")
        
        # 量子化設定（8bit）
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        
        # トークナイザー読み込み
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # モデル読み込み
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # LoRA準備
        model = prepare_model_for_kbit_training(model)
        
        # LoRA設定
        lora_config = LoraConfig(
            r=self.config.get("lora_r", 64),
            lora_alpha=self.config.get("lora_alpha", 128),
            target_modules=self.config.get("lora_target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            lora_dropout=self.config.get("lora_dropout", 0.05),
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # LoRA適用
        model = get_peft_model(model, lora_config)
        
        logger.info("[OK] Model and tokenizer loaded")
        
        return model, tokenizer
    
    def prepare_datasets(self, tokenizer):
        """データセット準備"""
        logger.info("Preparing datasets...")
        
        train_dataset = ProcessedDataDataset(
            self.dataset_path,
            tokenizer,
            max_length=self.config.get("max_seq_length", 2048),
            split="train"
        )
        
        val_dataset = ProcessedDataDataset(
            self.dataset_path,
            tokenizer,
            max_length=self.config.get("max_seq_length", 2048),
            split="val"
        )
        
        logger.info(f"[OK] Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")
        
        return train_dataset, val_dataset
    
    def train(self):
        """Fine-tuning実行"""
        logger.info("="*80)
        logger.info("Starting Fine-tuning")
        logger.info("="*80)
        
        # モデルとトークナイザー読み込み
        model, tokenizer = self.load_model_and_tokenizer()
        
        # データセット準備
        train_dataset, val_dataset = self.prepare_datasets(tokenizer)
        
        # データコレクター
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # トレーニング引数
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "checkpoints"),
            num_train_epochs=self.config.get("num_train_epochs", 3),
            per_device_train_batch_size=self.config.get("per_device_train_batch_size", 1),
            per_device_eval_batch_size=self.config.get("per_device_eval_batch_size", 1),
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 16),
            learning_rate=self.config.get("learning_rate", 2.0e-4),
            weight_decay=self.config.get("weight_decay", 0.01),
            warmup_ratio=self.config.get("warmup_ratio", 0.1),
            lr_scheduler_type=self.config.get("lr_scheduler_type", "cosine"),
            logging_steps=self.config.get("logging_steps", 10),
            save_steps=self.config.get("save_steps", 500),
            save_total_limit=self.config.get("save_total_limit", 5),
            evaluation_strategy="steps",
            eval_steps=self.config.get("eval_steps", 500),
            fp16=self.config.get("fp16", True),
            bf16=self.config.get("bf16", False),
            gradient_checkpointing=self.config.get("gradient_checkpointing", True),
            optim=self.config.get("optim", "paged_adamw_8bit"),
            report_to=[],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False
        )
        
        # トレーナー
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator
        )
        
        # シグナルハンドラー設定
        def signal_handler(signum, frame):
            logger.warning(f"Received signal {signum}, saving checkpoint...")
            self.recovery.save_checkpoint(trainer, 0, trainer.state.global_step)
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)
        
        # 学習実行
        logger.info("Starting training...")
        trainer.train()
        
        # 最終モデル保存
        final_model_dir = self.output_dir / "final_model"
        trainer.save_model(str(final_model_dir))
        tokenizer.save_pretrained(str(final_model_dir))
        
        logger.info(f"[OK] Final model saved to {final_model_dir}")
        
        return final_model_dir


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Fine-tune Hugging Face Model with Processed Data")
    parser.add_argument(
        '--base-model',
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help='Base model name'
    )
    parser.add_argument(
        '--dataset',
        type=Path,
        required=True,
        help='Hugging Face Dataset directory path'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory'
    )
    parser.add_argument(
        '--config',
        type=Path,
        help='Configuration file path (YAML)'
    )
    
    args = parser.parse_args()
    
    # 設定読み込み
    config = {}
    if args.config and args.config.exists():
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    
    # Fine-tuning実行
    finetuner = HFModelFinetuner(
        base_model_name=args.base_model,
        dataset_path=args.dataset,
        output_dir=args.output,
        config=config
    )
    
    final_model_dir = finetuner.train()
    
    logger.info("="*80)
    logger.info("[COMPLETE] Fine-tuning completed!")
    logger.info(f"Final model: {final_model_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    main()

