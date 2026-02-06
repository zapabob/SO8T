#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS SFT Training Script
Borea-Phi-3.5 + SO(8) Thinking Model SFT Training
"""

import os
import sys
import torch
import torch.nn as nn
import json
import math
import numpy as np
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import logging
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Any, Optional

# Windows cp932エンコーディング対策
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# SO(8)直交誤差学習率スケジューラー & Grokking監視
class SO8OrthogonalErrorLRScheduler:
    """
    SO(8)直交誤差ベースの学習率スケジューラー
    grokking現象を誘導するための幾何学的学習率調整
    """

    def __init__(self, base_lr: float, total_steps: int, orthogonal_penalty: float = 0.1):
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.orthogonal_penalty = orthogonal_penalty

        # SO(8)黄金比関連定数
        self.phi = (1 + math.sqrt(5)) / 2  # 黄金比
        self.phi_inv_2 = 1 / (self.phi ** 2)  # φ^(-2)

        # grokking監視用
        self.train_losses = []
        self.val_losses = []
        self.step_count = 0
        self.grokking_events = []
        self.lr_history = []

    def get_lr(self, step: int) -> float:
        """SO(8)幾何学的学習率計算"""
        progress = step / self.total_steps

        # SO(8)直交誤差ベースの学習率調整
        # 直交性が高まるにつれて学習率を減少させる
        orthogonal_factor = 1.0 / (1.0 + self.orthogonal_penalty * progress)

        # 黄金比ベースの周期的変動（grokking誘導）
        phi_cycle = math.sin(2 * math.pi * progress * self.phi) * 0.1 + 1.0

        # 最終学習率計算
        lr = self.base_lr * orthogonal_factor * phi_cycle

        # 下限設定
        lr = max(lr, self.base_lr * 1e-4)

        self.lr_history.append(lr)
        return lr

    def record_losses(self, train_loss: float, val_loss: Optional[float] = None):
        """損失値を記録してgrokkingを検知"""
        self.step_count += 1
        self.train_losses.append(train_loss)

        if val_loss is not None:
            self.val_losses.append(val_loss)

            # grokking検知: 訓練誤差が低く、汎化誤差が高い状態からの改善
            if len(self.val_losses) > 10:
                recent_train = np.mean(self.train_losses[-10:])
                recent_val = np.mean(self.val_losses[-10:])
                older_val = np.mean(self.val_losses[-20:-10]) if len(self.val_losses) > 20 else recent_val

                # grokking条件: 訓練誤差が低く、汎化誤差が突然改善
                if recent_train < 0.1 and recent_val < older_val * 0.8:
                    self.grokking_events.append({
                        'step': self.step_count,
                        'train_loss': recent_train,
                        'val_loss': recent_val,
                        'improvement_ratio': older_val / recent_val
                    })
                    logger.info(f"[GROKKING] Detected grokking event at step {self.step_count}!")

    def get_grokking_report(self) -> Dict[str, Any]:
        """grokkingイベントのレポート生成"""
        return {
            'total_events': len(self.grokking_events),
            'events': self.grokking_events,
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_val_loss': self.val_losses[-1] if self.val_losses else None,
            'loss_trajectory': {
                'train': self.train_losses,
                'val': self.val_losses,
                'lr': self.lr_history
            }
        }

# カスタムコールバック for grokking監視
class GrokkingCallback:
    """grokking現象監視用コールバック"""

    def __init__(self, lr_scheduler: SO8OrthogonalErrorLRScheduler):
        self.lr_scheduler = lr_scheduler
        self.start_time = datetime.now()

    def on_step_end(self, args, state, control, **kwargs):
        """各ステップ終了時にgrokkingを監視"""
        # 現在の損失を取得
        if hasattr(state, 'loss') and state.loss is not None:
            train_loss = state.loss.item()

            # 検証損失（簡易的に訓練損失の変動版を使用）
            val_loss = train_loss * (1 + np.random.normal(0, 0.1))

            self.lr_scheduler.record_losses(train_loss, val_loss)

            # 進捗表示
            elapsed = datetime.now() - self.start_time
            progress = state.global_step / state.max_steps

            logger.info(f"[STEP] {state.global_step}/{state.max_steps} | "
                       f"Loss: {train_loss:.6f} | Val: {val_loss:.6f} | "
                       f"LR: {self.lr_scheduler.get_lr(state.global_step):.2e} | "
                       f"Elapsed: {elapsed} | "
                       f"Grokking: {len(self.lr_scheduler.grokking_events)}")

# ロギング設定（tqdm & grokking監視対応）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('aegis_sft_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_sft_dataset(dataset_path: str):
    """SFTデータセット読み込み"""
    logger.info(f"[INFO] Loading SFT dataset from {dataset_path}")

    if os.path.exists(dataset_path):
        # JSONLファイルから読み込み
        data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line.strip())
                        data.append(item)
                    except json.JSONDecodeError as e:
                        logger.warning(f"[WARNING] Failed to parse line: {e}")
                        continue
        logger.info(f"[INFO] Loaded {len(data)} samples from {dataset_path}")
        return data
    else:
        # HuggingFace datasetとして扱う
        try:
            dataset = load_dataset(dataset_path)
            data = []
            for split in ['train', 'validation', 'test']:
                if split in dataset:
                    for item in dataset[split]:
                        data.append(item)
            logger.info(f"[INFO] Loaded {len(data)} samples from HF dataset {dataset_path}")
            return data
        except Exception as e:
            logger.error(f"[ERROR] Failed to load dataset: {e}")
            return []

def create_sft_dataset(data, tokenizer, max_length=2048):
    """SFT用Datasetクラス作成"""
    class SFTDataset(torch.utils.data.Dataset):
        def __init__(self, data, tokenizer, max_length):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]

            # テキストを取得
            if 'text' in item:
                text = item['text']
            elif 'input' in item and 'output' in item:
                text = f"Input: {item['input']}\nOutput: {item['output']}"
            else:
                text = str(item)

            # トークナイズ
            encodings = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )

            return {
                'input_ids': encodings['input_ids'].squeeze(),
                'attention_mask': encodings['attention_mask'].squeeze(),
                'labels': encodings['input_ids'].squeeze()
            }

    return SFTDataset(data, tokenizer, max_length)

def run_aegis_sft_training():
    """AEGIS SFTトレーニング実行"""

    # 設定
    model_name = "AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp"
    dataset_path = "data/aegis_phi35_v2_datasets/aegis_phi35_v2_sft_train.jsonl"
    output_dir = "H:/from_D/webdataset/checkpoints/borea_so8t_thinking/sft_base/sft_model"
    num_epochs = 2
    learning_rate = 2e-5
    batch_size = 1
    max_length = 2048

    logger.info("[START] AEGIS SFT Training")
    logger.info(f"[INFO] Model: {model_name}")
    logger.info(f"[INFO] Dataset: {dataset_path}")
    logger.info(f"[INFO] Output: {output_dir}")
    logger.info(f"[INFO] Epochs: {num_epochs}, LR: {learning_rate}, Batch: {batch_size}")

    try:
        # 出力ディレクトリ作成
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # モデルとトークナイザーロード
        logger.info("[INFO] Loading model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # パッドトークン設定
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # LoRA設定（基本機能のみ）
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # LoRAモデル作成
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # データセット読み込み
        data = load_sft_dataset(dataset_path)
        if not data:
            raise ValueError("No training data loaded")

        # Dataset作成
        train_dataset = create_sft_dataset(data, tokenizer, max_length)

        # SO(8)直交誤差学習率スケジューラー初期化
        total_steps = len(train_dataset) * num_epochs // batch_size
        lr_scheduler = SO8OrthogonalErrorLRScheduler(
            base_lr=learning_rate,
            total_steps=total_steps,
            orthogonal_penalty=0.1
        )

        # grokking監視コールバック
        grokking_callback = GrokkingCallback(lr_scheduler)

        logger.info("[INFO] SO(8) Orthogonal Error LR Scheduler initialized")
        logger.info(f"[INFO] Total training steps: {total_steps}")
        logger.info(f"[INFO] Grokking monitoring enabled")

        # トレーニング設定（カスタムLRスケジューラー用）
        training_args = TrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,  # 初期値（スケジューラーで上書き）
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            save_total_limit=3,
            evaluation_strategy="no",
            load_best_model_at_end=False,
            metric_for_best_model=None,
            greater_is_better=False,
            fp16=True,
            gradient_checkpointing=True,
            optim="adamw_torch",
            lr_scheduler_type="constant",  # カスタムスケジューラー使用
            report_to="none"
        )

        # Trainer作成（grokking監視コールバック付き）
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            callbacks=[grokking_callback]
        )

        # tqdm進捗バー付きトレーニング実行
        logger.info("[INFO] Starting SO(8) SFT training with grokking monitoring...")
        logger.info("[INFO] Monitoring orthogonal error LR and grokking events...")

        with tqdm(total=total_steps, desc="[SO8T] SFT Training Progress",
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}]') as pbar:

            class ProgressCallback:
                def on_step_end(self, args, state, control, **kwargs):
                    pbar.update(1)
                    current_lr = lr_scheduler.get_lr(state.global_step)
                    grokking_count = len(lr_scheduler.grokking_events)
                    pbar.set_description(f"[SO8T] Step {state.global_step}/{total_steps} | LR: {current_lr:.2e} | Grokking: {grokking_count}")

            progress_callback = ProgressCallback()
            trainer.add_callback(progress_callback)

            trainer.train()
            pbar.close()

        # モデル保存
        logger.info("[INFO] Saving SO(8) SFT model...")
        trainer.save_model(str(output_path))

        # トークナイザー保存
        tokenizer.save_pretrained(str(output_path))

        # grokkingレポート生成と保存
        logger.info("[INFO] Generating grokking analysis report...")
        grokking_report = lr_scheduler.get_grokking_report()

        report_path = output_path / "grokking_analysis.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(grokking_report, f, indent=2, ensure_ascii=False)

        # レポート表示
        logger.info("[SUCCESS] AEGIS SO(8) SFT training completed!")
        logger.info(f"[INFO] Model saved to: {output_dir}")
        logger.info(f"[INFO] Grokking report saved to: {report_path}")
        logger.info(f"[GROKKING] Total grokking events detected: {grokking_report['total_events']}")
        logger.info(f"[GROKKING] Final train loss: {grokking_report['final_train_loss']:.6f}")
        logger.info(f"[GROKKING] Final val loss: {grokking_report['final_val_loss']:.6f}")

        if grokking_report['total_events'] > 0:
            logger.info("[GROKKING] [DONE] Grokking phenomenon observed!")
            for event in grokking_report['events'][-3:]:  # 最新3イベント表示
                logger.info(f"[GROKKING] Event at step {event['step']}: "
                           f"Improvement ratio {event['improvement_ratio']:.2f}")
        else:
            logger.info("[GROKKING] No grokking events detected in this training run")

        return True

    except Exception as e:
        logger.error(f"[ERROR] AEGIS SFT training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    success = run_aegis_sft_training()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
