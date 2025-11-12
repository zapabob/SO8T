#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四値分類学習スクリプト

ALLOW/ESCALATION/DENY/REFUSEの四値分類を学習

Usage:
    python scripts/train_four_class_classifier.py --config configs/train_four_class.yaml
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
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
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ラベルマッピング
LABEL_TO_ID = {"ALLOW": 0, "ESCALATION": 1, "DENY": 2, "REFUSE": 3}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.class_weights)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class FourClassDataset(Dataset):
    """四値分類データセット"""
    
    def __init__(self, data_path: Path, tokenizer, max_length: int = 2048):
        """
        Args:
            data_path: JSONLファイルパス
            tokenizer: トークナイザー
            max_length: 最大シーケンス長
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        logger.info(f"Loading dataset from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    text = sample.get("text", "")
                    label = sample.get("label", "ALLOW")
                    if text and label in LABEL_TO_ID:
                        self.samples.append({
                            "text": text,
                            "label": LABEL_TO_ID[label]
                        })
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Loaded {len(self.samples):,} samples")
        # ラベル分布
        label_counts = {}
        for s in self.samples:
            label = ID_TO_LABEL[s["label"]]
            label_counts[label] = label_counts.get(label, 0) + 1
        logger.info(f"Label distribution: {label_counts}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]
        label = sample["label"]
        
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
            "labels": torch.tensor(label, dtype=torch.long)
        }


class FourClassModel(nn.Module):
    """四値分類モデル（ベースモデル + 分類ヘッド）"""
    
    def __init__(self, base_model, num_classes: int = 4, hidden_size: int = 3072):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # ベースモデルの出力
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 最後の隠れ状態の平均プーリング
        hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]
        pooled = hidden_states.mean(dim=1)  # [batch, hidden_size]
        
        # 分類
        logits = self.classifier(pooled)  # [batch, num_classes]
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {
            "loss": loss,
            "logits": logits
        }


def compute_metrics(eval_pred):
    """評価メトリクス計算"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # 基本メトリクス
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_per_class = f1_score(labels, predictions, average=None)
    
    # 誤検知率（危険な要求を誤ってALLOWする率）
    # DENY/REFUSEが正解なのにALLOWと予測した場合
    false_positive = 0
    false_positive_total = 0
    for true_label, pred_label in zip(labels, predictions):
        if true_label in [2, 3]:  # DENY or REFUSE
            false_positive_total += 1
            if pred_label == 0:  # ALLOW
                false_positive += 1
    
    false_positive_rate = false_positive / false_positive_total if false_positive_total > 0 else 0.0
    
    # 混同行列
    cm = confusion_matrix(labels, predictions)
    
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_allow": f1_per_class[0],
        "f1_escalation": f1_per_class[1],
        "f1_deny": f1_per_class[2],
        "f1_refuse": f1_per_class[3],
        "false_positive_rate": false_positive_rate,
        "confusion_matrix": cm.tolist()
    }


class FourClassTrainer:
    """四値分類学習クラス"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: 設定ファイルパス
        """
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.trainer = None
        
        logger.info(f"Four Class Trainer initialized")
        logger.info(f"  Device: {self.device}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_model(self):
        """モデルとトークナイザーをセットアップ"""
        logger.info("Setting up model and tokenizer...")
        
        model_path = self.config['model']['base_model']
        logger.info(f"Loading model from: {model_path}")
        
        # トークナイザー
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 8bit量子化設定
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
            bnb_8bit_use_double_quant=True,
        )
        
        # ベースモデル読み込み
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        # 8bit学習準備
        base_model = prepare_model_for_kbit_training(base_model)
        
        # LoRA設定
        lora_config = LoraConfig(
            r=self.config['model']['lora']['r'],
            lora_alpha=self.config['model']['lora']['lora_alpha'],
            target_modules=self.config['model']['lora']['target_modules'],
            lora_dropout=self.config['model']['lora']['lora_dropout'],
            bias=self.config['model']['lora']['bias'],
            task_type=TaskType.CAUSAL_LM
        )
        
        base_model = get_peft_model(base_model, lora_config)
        
        # 分類モデル作成
        hidden_size = self.config['model'].get('hidden_size', 3072)
        self.model = FourClassModel(base_model, num_classes=4, hidden_size=hidden_size)
        
        logger.info("[OK] Model setup completed")
    
    def setup_datasets(self):
        """データセットをセットアップ"""
        logger.info("Setting up datasets...")
        
        # 訓練データ
        train_data_path = Path(self.config['data']['train_data'])
        if not train_data_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_data_path}")
        
        self.train_dataset = FourClassDataset(
            train_data_path,
            self.tokenizer,
            max_length=self.config['data']['max_seq_length']
        )
        
        # 検証データ
        val_data_path = Path(self.config['data']['val_data'])
        if val_data_path.exists():
            self.val_dataset = FourClassDataset(
                val_data_path,
                self.tokenizer,
                max_length=self.config['data']['max_seq_length']
            )
        else:
            logger.warning(f"Validation data not found: {val_data_path}")
            self.val_dataset = None
        
        logger.info(f"[OK] Datasets setup completed")
        logger.info(f"  Train samples: {len(self.train_dataset):,}")
        if self.val_dataset:
            logger.info(f"  Val samples: {len(self.val_dataset):,}")
    
    def train(self):
        """学習実行"""
        logger.info("="*80)
        logger.info("Four Class Classification Training")
        logger.info("="*80)
        
        # モデル・データセットセットアップ
        self.setup_model()
        self.setup_datasets()
        
        # 訓練引数
        training_args = TrainingArguments(
            output_dir=self.config['training']['output_dir'],
            num_train_epochs=self.config['training']['num_train_epochs'],
            per_device_train_batch_size=self.config['training']['per_device_train_batch_size'],
            per_device_eval_batch_size=self.config['training'].get('per_device_eval_batch_size', 2),
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            learning_rate=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            lr_scheduler_type=self.config['training']['lr_scheduler_type'],
            warmup_ratio=self.config['training']['warmup_ratio'],
            logging_steps=self.config['training']['logging_steps'],
            save_steps=self.config['training']['save_steps'],
            save_total_limit=self.config['training'].get('save_total_limit', 5),
            evaluation_strategy=self.config['training'].get('evaluation_strategy', 'steps'),
            eval_steps=self.config['training'].get('eval_steps', 500),
            bf16=self.config['training'].get('bf16', True),
            gradient_checkpointing=self.config['training'].get('gradient_checkpointing', True),
            optim=self.config['training'].get('optim', 'paged_adamw_8bit'),
            load_best_model_at_end=self.config['training'].get('load_best_model_at_end', True),
            metric_for_best_model=self.config['training'].get('metric_for_best_model', 'f1_macro'),
            greater_is_better=True,
            report_to=self.config['training'].get('report_to', [])
        )
        
        # Trainer作成
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=compute_metrics
        )
        
        # 学習実行
        logger.info("Starting training...")
        start_time = time.time()
        
        try:
            self.trainer.train()
            
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time/3600:.2f} hours")
            
            # 最終評価
            if self.val_dataset:
                logger.info("Running final evaluation...")
                eval_results = self.trainer.evaluate()
                logger.info("Final evaluation results:")
                for key, value in eval_results.items():
                    logger.info(f"  {key}: {value}")
            
            # 最終モデル保存
            output_model_dir = Path(self.config['training']['output_dir']) / "final_model"
            output_model_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(str(output_model_dir))
            self.tokenizer.save_pretrained(str(output_model_dir))
            
            logger.info(f"[OK] Final model saved to {output_model_dir}")
            
        except KeyboardInterrupt:
            logger.warning("[WARNING] Training interrupted by user")
        except Exception as e:
            logger.error(f"[ERROR] Training failed: {e}")
            raise


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Four Class Classification Training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_four_class.yaml",
        help="Configuration file path"
    )
    
    args = parser.parse_args()
    
    # 学習実行
    trainer = FourClassTrainer(config_path=args.config)
    
    try:
        trainer.train()
        return 0
        
    except Exception as e:
        logger.error(f"[FAILED] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import time
    sys.exit(main())

