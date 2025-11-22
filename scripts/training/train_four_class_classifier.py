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
import signal
import atexit
import time
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

# 電源断復旧用のセッション管理クラス
class TrainingSessionManager:
    """トレーニングセッションの管理と自動復旧"""

    def __init__(self, output_dir: str, session_id: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.session_file = self.output_dir / "training_session.json"
        self.checkpoint_interval = 300  # 5分ごとにセッション保存
        self.last_save_time = time.time()

        # セッションID生成
        if session_id is None:
            import uuid
            session_id = str(uuid.uuid4())[:8]

        self.session_id = session_id
        self.session_data = {
            "session_id": self.session_id,
            "start_time": time.time(),
            "last_update": time.time(),
            "current_step": 0,
            "total_steps": 0,
            "status": "initializing",
            "config_path": None,
            "resume_from_checkpoint": None
        }

        # シグナルハンドラー設定
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        if hasattr(signal, 'SIGBREAK'):  # Windows
            signal.signal(signal.SIGBREAK, self._signal_handler)

        # 終了時処理
        atexit.register(self._save_session)

        # 既存セッションの読み込み
        self._load_session()

    def _signal_handler(self, signum, frame):
        """シグナルハンドラー"""
        logger.warning(f"Signal {signum} received. Saving session and exiting...")
        self._save_session()
        sys.exit(1)

    def _load_session(self):
        """セッション情報の読み込み"""
        if self.session_file.exists():
            try:
                with open(self.session_file, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)

                # セッションデータ更新
                self.session_data.update(loaded_data)
                self.session_id = loaded_data.get('session_id', self.session_id)

                logger.info(f"[SESSION] Loaded session: {self.session_id}")
                logger.info(f"[SESSION] Previous status: {loaded_data.get('status', 'unknown')}")
                logger.info(f"[SESSION] Progress: {loaded_data.get('current_step', 0)}/{loaded_data.get('total_steps', 0)}")

            except Exception as e:
                logger.warning(f"Failed to load session file: {e}")

    def _save_session(self):
        """セッション情報の保存"""
        try:
            self.session_data["last_update"] = time.time()
            self.session_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.session_file, 'w', encoding='utf-8') as f:
                json.dump(self.session_data, f, indent=2, ensure_ascii=False)

            logger.debug(f"[SESSION] Session saved: {self.session_id}")

        except Exception as e:
            logger.error(f"Failed to save session: {e}")

    def update_progress(self, current_step: int, total_steps: int, status: str = "running"):
        """進捗更新"""
        self.session_data["current_step"] = current_step
        self.session_data["total_steps"] = total_steps
        self.session_data["status"] = status

        # 定期保存
        current_time = time.time()
        if current_time - self.last_save_time >= self.checkpoint_interval:
            self._save_session()
            self.last_save_time = current_time

    def set_config(self, config_path: str, resume_from_checkpoint: Optional[str] = None):
        """設定情報の保存"""
        self.session_data["config_path"] = config_path
        self.session_data["resume_from_checkpoint"] = resume_from_checkpoint
        self._save_session()

    def finish_session(self):
        """セッション完了"""
        self.session_data["status"] = "completed"
        self.session_data["end_time"] = time.time()
        self._save_session()
        logger.info(f"[SESSION] Session completed: {self.session_id}")

    def get_session_info(self) -> Dict[str, Any]:
        """セッション情報取得"""
        return self.session_data.copy()

# トレーニング進捗更新用のコールバッククラス
class ProgressCallback:
    """トレーニング進捗をセッション管理に更新するコールバック"""

    def __init__(self, session_manager: TrainingSessionManager, total_steps: int):
        self.session_manager = session_manager
        self.total_steps = total_steps

    def __call__(self, step: int):
        """各ステップで呼ばれる"""
        self.session_manager.update_progress(step, self.total_steps, "running")

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

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing on the base model"""
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing on the base model"""
        if hasattr(self.base_model, 'gradient_checkpointing_disable'):
            self.base_model.gradient_checkpointing_disable()
    
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
    
    def __init__(self, config_path: str, resume_from_checkpoint: Optional[str] = None):
        """
        Args:
            config_path: 設定ファイルパス
            resume_from_checkpoint: 再開するチェックポイントのパス
        """
        self.config = self._load_config(config_path)
        self.resume_from_checkpoint = resume_from_checkpoint
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        # 自動チェックポイント検出
        if self.resume_from_checkpoint is None:
            self.resume_from_checkpoint = self._find_latest_checkpoint()

        # セッション管理初期化
        output_dir = self.config['training']['output_dir']
        self.session_manager = TrainingSessionManager(output_dir)
        self.session_manager.set_config(config_path, self.resume_from_checkpoint)

    def _find_latest_checkpoint(self) -> Optional[str]:
        """最新のチェックポイントを検出"""
        output_dir = Path(self.config['training']['output_dir'])
        if not output_dir.exists():
            return None

        # checkpoint-* ディレクトリを検索
        checkpoint_dirs = list(output_dir.glob("checkpoint-*"))
        if not checkpoint_dirs:
            return None

        # ステップ番号でソートして最新のものを選択
        checkpoint_dirs.sort(key=lambda x: int(x.name.split('-')[1]) if len(x.name.split('-')) > 1 and x.name.split('-')[1].isdigit() else 0)

        latest_checkpoint = checkpoint_dirs[-1]
        logger.info(f"[CHECKPOINT] Found latest checkpoint: {latest_checkpoint}")

        return str(latest_checkpoint)
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

        # チェックポイント再開情報
        if self.resume_from_checkpoint:
            logger.info(f"[RESUME] Resuming training from checkpoint: {self.resume_from_checkpoint}")
        else:
            logger.info("[RESUME] Starting new training session")

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
            eval_strategy=self.config['training'].get('evaluation_strategy', 'steps'),
            eval_steps=self.config['training'].get('eval_steps', 500),
            bf16=self.config['training'].get('bf16', True),
            gradient_checkpointing=self.config['training'].get('gradient_checkpointing', True),
            optim=self.config['training'].get('optim', 'paged_adamw_8bit'),
            load_best_model_at_end=self.config['training'].get('load_best_model_at_end', True),
            metric_for_best_model=self.config['training'].get('metric_for_best_model', 'f1_macro'),
            greater_is_better=True,
            resume_from_checkpoint=self.resume_from_checkpoint,
            report_to=self.config['training'].get('report_to', []),

        # 総ステップ数の計算
        total_steps = int((len(self.train_dataset) / (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)) * training_args.num_train_epochs)
        self.total_steps = total_steps

        # 進捗コールバック設定
        self.progress_callback = ProgressCallback(self.session_manager, total_steps)
        )
        
        # Trainer作成
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[self.progress_callback]
        )
        
        # 学習実行
        logger.info("Starting training...")
        start_time = time.time()
        
        try:
            self.trainer.train()
            
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time/3600:.2f} hours")

            # セッション完了マーク
            self.session_manager.finish_session()

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
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    
    args = parser.parse_args()
    
    # 学習実行
    trainer = FourClassTrainer(
        config_path=args.config,
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    
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

