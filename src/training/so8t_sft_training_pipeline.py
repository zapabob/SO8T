#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO(8)T SFT Training Pipeline
SO(8)残差アダプター統合SFT学習パイプライン
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
import os
import time
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import math
from datetime import datetime
import logging
from tqdm import tqdm

# 自作モジュールのインポート
from so8_residual_adapter import SO8ThinkingModel, create_so8_adapter_config, SO8Config

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SFTDataset(Dataset):
    """SFT用データセット"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(data_path)

    def load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """データセット読み込み"""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    data.append(item)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # テキストを取得
        text = item.get('text', '')
        if isinstance(text, str) and text.startswith('{'):
            # JSON形式の場合
            try:
                text_data = json.loads(text)
                instruction = text_data.get('instruction', '')
                if 'messages' in text_data:
                    # 会話形式
                    conversation = ""
                    for msg in text_data['messages']:
                        role = msg.get('role', 'user')
                        content = msg.get('content', '')
                        conversation += f"{role}: {content}\n"
                    text = conversation
                else:
                    text = instruction
            except:
                pass

        # トークナイズ
        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'labels': tokenized['input_ids'].squeeze()
        }

class SO8TCheckpointCallback:
    """SO(8)Tチェックポイントコールバック"""

    def __init__(self, save_dir: str, save_steps: int = 180):  # 3分 = 180秒
        self.save_dir = Path(save_dir)
        self.save_steps = save_steps
        self.last_save_time = time.time()
        self.checkpoints = []
        self.max_checkpoints = 5

    def __call__(self, model, tokenizer, step):
        current_time = time.time()

        # 時間ベースのチェックポイント保存
        if current_time - self.last_save_time >= self.save_steps:
            checkpoint_path = self.save_dir / f"checkpoint_step_{step}"
            checkpoint_path.mkdir(exist_ok=True)

            # モデル保存
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)

            # チェックポイント管理（ローリングストック）
            self.checkpoints.append(checkpoint_path)
            if len(self.checkpoints) > self.max_checkpoints:
                # 最も古いチェックポイントを削除
                old_checkpoint = self.checkpoints.pop(0)
                if old_checkpoint.exists():
                    shutil.rmtree(old_checkpoint)

            self.last_save_time = current_time
            logger.info(f"チェックポイント保存: {checkpoint_path}")

class SO8TSFTTrainer:
    """SO(8)T SFTトレーナー"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # モデルとトークナイザーの初期化
        self.setup_model_and_tokenizer()

        # SO(8)アダプター設定
        self.so8_config = create_so8_adapter_config(self.model.config.hidden_size)

        # データセット準備
        self.setup_datasets()

        # チェックポイントコールバック
        self.checkpoint_callback = SO8TCheckpointCallback(
            save_dir=config.get('output_dir', './checkpoints'),
            save_steps=config.get('checkpoint_interval', 180)
        )

    def setup_model_and_tokenizer(self):
        """モデルとトークナイザーのセットアップ"""
        model_name = self.config.get('model_name', 'microsoft/phi-3.5-mini-instruct')

        logger.info(f"モデル読み込み: {model_name}")

        # HuggingFaceキャッシュをH:/from_D/webdatasetに設定
        cache_dir = Path("H:/from_D/webdataset/hf_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 4-bit量子化設定
        if self.config.get('use_4bit', True):
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                cache_dir=str(cache_dir)
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                cache_dir=str(cache_dir)
            )

        # LoRA設定
        if self.config.get('use_lora', True):
            lora_config = LoraConfig(
                r=self.config.get('lora_r', 16),
                lora_alpha=self.config.get('lora_alpha', 32),
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=self.config.get('lora_dropout', 0.05),
                bias="none",
                task_type="CAUSAL_LM"
            )

            if self.config.get('use_4bit', True):
                self.model = prepare_model_for_kbit_training(self.model)

            self.model = get_peft_model(self.model, lora_config)
            logger.info("LoRAアダプター適用完了")

    def setup_datasets(self):
        """データセットの準備"""
        train_dataset_path = self.config.get('train_dataset', 'data/train_sft_enhanced.jsonl')
        eval_dataset_path = self.config.get('eval_dataset', 'data/test_eval.jsonl')

        self.train_dataset = SFTDataset(train_dataset_path, self.tokenizer,
                                      max_length=self.config.get('max_length', 2048))
        self.eval_dataset = SFTDataset(eval_dataset_path, self.tokenizer,
                                     max_length=self.config.get('max_length', 2048))

        logger.info(f"トレーニングデータ数: {len(self.train_dataset)}")
        logger.info(f"評価データ数: {len(self.eval_dataset)}")

    def create_so8t_model(self):
        """SO(8)Tモデル作成"""
        logger.info("SO(8)Tモデル作成開始")

        # SO(8)Thinkingモデルでラップ
        so8t_model = SO8ThinkingModel(self.model, self.so8_config)

        # デバイス移動
        so8t_model = so8t_model.to(self.device)

        logger.info("SO(8)Tモデル作成完了")
        return so8t_model

    def train(self):
        """トレーニング実行"""
        logger.info("SO(8)T SFTトレーニング開始")

        # SO(8)Tモデル作成
        model = self.create_so8t_model()

        # トレーニング引数
        training_args = TrainingArguments(
            output_dir=self.config.get('output_dir', './checkpoints'),
            num_train_epochs=self.config.get('num_epochs', 3),
            per_device_train_batch_size=self.config.get('batch_size', 4),
            per_device_eval_batch_size=self.config.get('batch_size', 4),
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 2),
            learning_rate=self.config.get('learning_rate', 2e-5),
            weight_decay=self.config.get('weight_decay', 0.01),
            warmup_steps=self.config.get('warmup_steps', 100),
            logging_steps=self.config.get('logging_steps', 10),
            save_steps=self.config.get('save_steps', 500),
            evaluation_strategy="steps",
            eval_steps=self.config.get('eval_steps', 500),
            save_total_limit=self.config.get('save_total_limit', 3),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=not self.config.get('use_4bit', True),
            bf16=self.config.get('use_4bit', True),
            dataloader_pin_memory=False,
        )

        # データコレーター
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # カスタムコールバック
        class SO8TTrainingCallback:
            def __init__(self, so8t_model):
                self.so8t_model = so8t_model
                self.step_count = 0

            def __call__(self, state, control, model, tokenizer, **kwargs):
                # アルファゲートアニーリング
                if hasattr(self.so8t_model, 'so8_adapters'):
                    for adapter in self.so8t_model.so8_adapters.values():
                        if hasattr(adapter, 'anneal_alpha_gate'):
                            adapter.anneal_alpha_gate(self.step_count)

                self.step_count += 1

                # 定期チェックポイント保存
                if hasattr(self, 'checkpoint_callback'):
                    self.checkpoint_callback(model, tokenizer, self.step_count)

        training_callback = SO8TTrainingCallback(model)
        training_callback.checkpoint_callback = self.checkpoint_callback

        # トレーナー作成
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            callbacks=[training_callback]
        )

        # トレーニング実行
        logger.info("トレーニング開始...")
        start_time = time.time()

        trainer.train()

        training_time = time.time() - start_time
        logger.info(f"トレーニング時間: {training_time:.2f}秒")

        # 最終モデル保存
        final_model_path = Path(self.config.get('output_dir', './checkpoints')) / "final_model"
        final_model_path.mkdir(exist_ok=True)

        trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)

        logger.info(f"最終モデル保存完了: {final_model_path}")
        print(f"最終モデル保存完了: {final_model_path}")
        print(f"トレーニング時間: {training_time:.2f}秒")
        return trainer, training_time, final_model_path

def create_sft_config() -> Dict[str, Any]:
    """SFT設定を作成"""
    return {
        'model_name': 'AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp',
        'train_dataset': 'data/train_sft_enhanced.jsonl',
        'eval_dataset': 'data/test_eval.jsonl',
        'output_dir': './checkpoints/sft_so8t',
        'num_epochs': 3,
        'batch_size': 2,  # 小さめのバッチサイズ
        'gradient_accumulation_steps': 4,
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
        'warmup_steps': 100,
        'max_length': 2048,
        'logging_steps': 10,
        'save_steps': 500,
        'eval_steps': 500,
        'save_total_limit': 3,
        'use_4bit': True,
        'use_lora': True,
        'lora_r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.05,
        'checkpoint_interval': 180  # 3分
    }

def main():
    """メイン関数"""
    print("[START] SO(8)T SFT Training Pipeline")
    print("=" * 50)

    # 設定
    config = create_sft_config()

    # トレーナー作成
    trainer = SO8TSFTTrainer(config)

    # トレーニング実行
    trained_trainer, training_time = trainer.train()

    print("[OK] SFTトレーニング完了!")
    print(f"[STATS] トレーニング済みモデル: {config['output_dir']}/final_model")
    print(f"トレーニング時間: {training_time:.2f}秒")
    # 音声通知
    try:
        import subprocess
        subprocess.run([
            "powershell", "-ExecutionPolicy", "Bypass",
            "-File", "scripts\\utils\\play_audio_notification.ps1"
        ], check=True)
    except Exception as e:
        print(f"[WARNING] 音声通知失敗: {e}")

if __name__ == "__main__":
    main()
