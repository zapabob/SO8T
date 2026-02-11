#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HODACHI-Borea-phi3.5-mini-instinct-jp Thinking SFT Trainer
/thinkingモデル化のためのSupervised Fine-Tuningスクリプト

このスクリプトは以下の機能を提供します：
1. HODACHI-Borea-phi3.5-mini-instinct-jpモデルの読み込み
2. Thinking SFTデータセットでのファインチューニング
3. /thinking機能の学習
4. RTX3060最適化
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import psutil

# SO8T imports
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

# CUDA設定
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SFTConfig:
    """SFT設定"""
    model_name: str = "HODACHI-Borea-phi3.5-mini-instinct-jp"
    dataset_path: str = "data/sft_thinking/hodachi_borea_phi35_thinking_sft_dataset.jsonl"
    output_dir: str = "outputs/hodachi_borea_phi35_thinking_sft"
    max_length: int = 2048
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    warmup_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 50
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "loss"
    greater_is_better: bool = False

class ThinkingSFTDataset(Dataset):
    """Thinking SFTデータセット"""

    def __init__(self, data_file: str, tokenizer, max_length: int = 2048):
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        self.load_data()
        logger.info(f"Loaded {len(self.data)} Thinking SFT examples")

    def load_data(self):
        """データ読み込み"""
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading Thinking SFT data"):
                if line.strip():
                    item = json.loads(line.strip())
                    self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # メッセージをチャット形式に変換
        messages = item['messages']

        # システムプロンプトがある場合は追加
        system_message = None
        conversation = []

        for msg in messages:
            if msg['role'] == 'system':
                system_message = msg['content']
            elif msg['role'] == 'user':
                conversation.append({"role": "user", "content": msg['content']})
            elif msg['role'] == 'assistant':
                conversation.append({"role": "assistant", "content": msg['content']})

        # システムメッセージを最初のユーザーメッセージに統合
        if system_message and conversation:
            conversation[0]['content'] = f"{system_message}\n\n{conversation[0]['content']}"

        # Phi-3.5のチャットフォーマットに変換
        formatted_text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )

        # トークナイズ
        tokenized = self.tokenizer(
            formatted_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'labels': tokenized['input_ids'].squeeze()
        }

class ThinkingSFTTrainer:
    """Thinking SFTトレーナー"""

    def __init__(self, config: SFTConfig):
        self.config = config

        # モデルとトークナイザーの初期化
        self._init_model_and_tokenizer()

        # データセットの初期化
        self._init_dataset()

        # トレーニング設定
        self._init_training_args()

        logger.info("Thinking SFT Trainer initialized")
        logger.info(f"Model: {config.model_name}")
        logger.info(f"Dataset: {len(self.dataset)} examples")
        logger.info(f"Output dir: {config.output_dir}")

    def _init_model_and_tokenizer(self):
        """モデルとトークナイザーの初期化"""
        logger.info(f"Loading model: {self.config.model_name}")

        # モデルパスの解決
        model_path = f"models/{self.config.model_name}"

        if not Path(model_path).exists():
            # Hugging Face Hubからダウンロードする場合
            model_path = self.config.model_name

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side="left"
            )

            # PADトークンの設定
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,  # RTX3060最適化
                device_map="auto",
                load_in_8bit=True  # メモリ効率化
            )

            logger.info("Model and tokenizer loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _init_dataset(self):
        """データセットの初期化"""
        logger.info("Initializing Thinking SFT dataset")

        self.dataset = ThinkingSFTDataset(
            self.config.dataset_path,
            self.tokenizer,
            self.config.max_length
        )

        # データセットを訓練・検証に分割（8:2）
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        logger.info(f"Train dataset: {len(self.train_dataset)} examples")
        logger.info(f"Validation dataset: {len(self.val_dataset)} examples")

    def _init_training_args(self):
        """トレーニング引数の初期化"""
        self.training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            evaluation_strategy=self.config.evaluation_strategy,
            save_strategy=self.config.save_strategy,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,

            # RTX3060最適化設定
            fp16=True,
            gradient_checkpointing=True,
            dataloader_num_workers=0,
            dataloader_pin_memory=True,

            # メモリ最適化
            remove_unused_columns=False,
            label_smoothing_factor=0.1,

            # レポート
            report_to="tensorboard",
            run_name=f"hodachi_borea_phi35_thinking_sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    def train(self):
        """Thinking SFTトレーニングの実行"""
        logger.info("🚀 Starting Thinking SFT Training")
        logger.info(f"📊 Training examples: {len(self.train_dataset)}")
        logger.info(f"📊 Validation examples: {len(self.val_dataset)}")
        logger.info(f"🎯 Max steps: {self.training_args.num_train_epochs * len(self.train_dataset) // (self.config.batch_size * self.config.gradient_accumulation_steps)}")

        start_time = time.time()

        # データコレーター
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # トレーナーの初期化
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
        )

        # トレーニング実行
        trainer.train()

        # 最終モデルの保存
        final_model_path = Path(self.config.output_dir) / "final_model"
        trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)

        total_time = time.time() - start_time
        logger.info("✅ Thinking SFT Training completed!"        logger.info(".2f"        logger.info(".2f"
        # 音声通知
        try:
            import subprocess
            subprocess.run([
                "powershell", "-ExecutionPolicy", "Bypass",
                "-File", "scripts\\utils\\play_audio_notification.ps1"
            ], check=True)
        except Exception as e:
            logger.warning(f"Audio notification failed: {e}")

        return final_model_path

def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="HODACHI-Borea-phi3.5-mini-instinct-jp Thinking SFT Trainer")
    parser.add_argument("--model_name", type=str, default="HODACHI-Borea-phi3.5-mini-instinct-jp",
                       help="Model name or path")
    parser.add_argument("--dataset_path", type=str,
                       default="data/sft_thinking/hodachi_borea_phi35_thinking_sft_dataset.jsonl",
                       help="Path to Thinking SFT dataset")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")

    args = parser.parse_args()

    # 設定の作成
    config = SFTConfig()
    config.model_name = args.model_name
    config.dataset_path = args.dataset_path
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.num_train_epochs = args.num_epochs

    if args.output_dir:
        config.output_dir = args.output_dir
    else:
        config.output_dir = f"outputs/{config.model_name.replace('/', '_')}_thinking_sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # トレーナーの初期化と実行
    trainer = ThinkingSFTTrainer(config)
    final_model_path = trainer.train()

    print(f"\n🎉 Thinking SFT completed!")
    print(f"📁 Final model saved to: {final_model_path}")
    print(f"🧠 Model is now capable of /thinking functionality")

if __name__ == "__main__":
    main()
