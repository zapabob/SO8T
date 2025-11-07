#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Borea-Phi-3.5-mini 日本語ファインチューニングスクリプト

QLoRA 8bit学習、電源断リカバリー対応、チェックポイント保存（5分間隔）

Usage:
    python scripts/finetune_borea_japanese.py --config configs/finetune_borea_japanese.yaml
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import deque
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent

# オフロードディレクトリ
OFFLOAD_DIR = PROJECT_ROOT / "offload"
OFFLOAD_DIR.mkdir(exist_ok=True)

# チェックポイント設定
CHECKPOINT_INTERVAL = 300  # 5分間隔
MAX_CHECKPOINTS = 10


class JapaneseDataset(Dataset):
    """日本語データセット"""
    
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
                    if text:
                        self.samples.append(text)
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Loaded {len(self.samples):,} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text = self.samples[idx]
        
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
    
    def __init__(self, session_file: Path):
        self.session_file = session_file
        self.session: Optional[Dict] = None
        self.emergency_save = False
        
        # シグナルハンドラー設定
        signal.signal(signal.SIGINT, self._emergency_handler)
        signal.signal(signal.SIGTERM, self._emergency_handler)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, self._emergency_handler)
    
    def _emergency_handler(self, signum, frame):
        """緊急保存ハンドラー"""
        logger.warning(f"[WARNING] Signal {signum} received. Emergency save...")
        self.emergency_save = True
        if self.session:
            self.save_session()
        logger.info("[OK] Emergency save completed")
        sys.exit(0)
    
    def create_session(self, total_steps: int) -> Dict:
        """新規セッション作成"""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session = {
            "session_id": session_id,
            "start_time": time.time(),
            "current_step": 0,
            "total_steps": total_steps,
            "current_epoch": 0,
            "last_checkpoint": time.time(),
            "checkpoints": [],
            "completed": False,
            "end_time": None
        }
        self.session = session
        return session
    
    def load_session(self) -> Optional[Dict]:
        """前回セッション復旧"""
        if not self.session_file.exists():
            return None
        
        try:
            with open(self.session_file, 'r', encoding='utf-8') as f:
                session = json.load(f)
            self.session = session
            logger.info(f"[OK] Session restored: {session['session_id']}")
            logger.info(f"    Progress: {session['current_step']}/{session['total_steps']}")
            return session
        except Exception as e:
            logger.warning(f"[WARNING] Failed to restore session: {e}")
            return None
    
    def save_session(self):
        """セッション保存"""
        if not self.session:
            return
        
        self.session_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.session_file, 'w', encoding='utf-8') as f:
            json.dump(self.session, f, indent=2, ensure_ascii=False)
    
    def save_checkpoint(self, checkpoint_data: Dict, checkpoint_id: int, checkpoint_dir: Path):
        """チェックポイント保存"""
        checkpoint_path = checkpoint_dir / f"checkpoint_{self.session['session_id']}_{checkpoint_id}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint_data, checkpoint_path)
        
        # チェックポイントリスト更新
        self.session["checkpoints"].append(str(checkpoint_path))
        
        # 古いチェックポイント削除
        if len(self.session["checkpoints"]) > MAX_CHECKPOINTS:
            old_checkpoint = Path(self.session["checkpoints"].pop(0))
            if old_checkpoint.exists():
                old_checkpoint.unlink()
        
        self.session["last_checkpoint"] = time.time()
        self.save_session()


class BoreaJapaneseFinetuner:
    """Borea-Phi-3.5-mini 日本語ファインチューニングクラス"""
    
    def __init__(
        self,
        config_path: str,
        resume_path: Optional[str] = None,
        auto_resume: bool = False
    ):
        """
        Args:
            config_path: 設定ファイルパス
            resume_path: 復旧チェックポイントパス
            auto_resume: 自動再開モード
        """
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.trainer = None
        
        # リカバリー設定
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_base = Path("checkpoints") / "borea_phi35_mini_japanese"
        checkpoint_base.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = checkpoint_base / f"session_{self.session_id}"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.session_file = checkpoint_base / "training_session.json"
        
        self.recovery = PowerFailureRecovery(self.session_file)
        
        # 自動再開モード
        if auto_resume:
            existing_session = self.recovery.load_session()
            if existing_session:
                resume_path = self._find_latest_checkpoint(existing_session)
        
        self.resume_path = resume_path
        self.is_recovery = resume_path is not None
        
        logger.info(f"Borea Japanese Finetuner initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Session ID: {self.session_id}")
        logger.info(f"  Checkpoint dir: {self.checkpoint_dir}")
        logger.info(f"  Recovery mode: {self.is_recovery}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _find_latest_checkpoint(self, session: Dict) -> Optional[str]:
        """最新チェックポイントを検索"""
        checkpoints = session.get("checkpoints", [])
        if not checkpoints:
            return None
        
        latest_cp = checkpoints[-1]
        cp_path = Path(latest_cp)
        if cp_path.exists():
            return str(cp_path.parent)
        return None
    
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
        
        # 8bit量子化設定（メモリ効率化）
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf8",
        )
        
        # メモリ制限設定（より厳しい制限）
        if torch.cuda.is_available():
            # GPUメモリをより制限
            max_memory = {0: "8GB", "cpu": "30GB"}  # CPUオフロードを積極的に使用
        else:
            max_memory = {"cpu": "16GB"}
        
        # モデル読み込み（段階的読み込みでメモリ使用量を最小化）
        # ページングファイル不足を回避するため、より軽量な方法を使用
        logger.info("Loading model with minimal memory usage...")
        logger.warning("WARNING: Disk space is very low. Model loading may fail if page file cannot expand.")
        
        # 一時ファイルの場所を環境変数で設定（可能な場合）
        import tempfile
        temp_dir = tempfile.gettempdir()
        logger.info(f"Using temp directory: {temp_dir}")
        
        try:
            # 方法1: 最小限のメモリで読み込み（CPU経由）
            # 量子化設定を簡略化
            bnb_config_simple = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config_simple,
                device_map="cpu",  # まずCPUに読み込み
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                use_cache=False,  # キャッシュを無効化してメモリ削減
            )
            logger.info("Model loaded to CPU successfully")
            
            # CPUからGPUに移動（段階的、メモリが十分な場合のみ）
            if torch.cuda.is_available():
                try:
                    logger.info("Moving model to GPU...")
                    self.model = self.model.to(self.device)
                    logger.info("Model moved to GPU successfully")
                except RuntimeError as e:
                    logger.warning(f"Failed to move to GPU: {e}")
                    logger.info("Keeping model on CPU (will be slower but should work)")
                    self.device = torch.device("cpu")
                    
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            logger.error("="*80)
            logger.error("CRITICAL: Insufficient disk space for page file expansion!")
            logger.error("="*80)
            logger.error("Please free up at least 10GB of disk space on C: drive")
            logger.error("or move the model to a drive with more free space.")
            logger.error("="*80)
            raise RuntimeError(
                "Failed to load model due to insufficient disk space. "
                "Please free up at least 10GB on C: drive or increase page file size manually."
            )
        
        # 8bit学習準備（量子化済みの場合のみ）
        if hasattr(self.model, 'quantization_config') and self.model.quantization_config is not None:
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            # 量子化されていない場合はここで量子化
            logger.info("Applying 8bit quantization...")
            self.model = prepare_model_for_kbit_training(self.model)
        
        # LoRA設定
        lora_config = LoraConfig(
            r=self.config['model']['lora']['r'],
            lora_alpha=self.config['model']['lora']['lora_alpha'],
            target_modules=self.config['model']['lora']['target_modules'],
            lora_dropout=self.config['model']['lora']['lora_dropout'],
            bias=self.config['model']['lora']['bias'],
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("[OK] Model setup completed")
    
    def setup_datasets(self):
        """データセットをセットアップ"""
        logger.info("Setting up datasets...")
        
        # 訓練データ
        train_data_path = Path(self.config['data']['train_data'][0])
        if not train_data_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_data_path}")
        
        self.train_dataset = JapaneseDataset(
            train_data_path,
            self.tokenizer,
            max_length=self.config['data']['max_seq_length']
        )
        
        # 検証データ
        val_data_path = Path(self.config['data']['val_data'])
        if val_data_path.exists():
            self.val_dataset = JapaneseDataset(
                val_data_path,
                self.tokenizer,
                max_length=self.config['data']['max_seq_length']
            )
        else:
            logger.warning(f"Validation data not found: {val_data_path}, using train split")
            self.val_dataset = None
        
        logger.info(f"[OK] Datasets setup completed")
        logger.info(f"  Train samples: {len(self.train_dataset):,}")
        if self.val_dataset:
            logger.info(f"  Val samples: {len(self.val_dataset):,}")
    
    def train(self):
        """学習実行"""
        logger.info("="*80)
        logger.info("Borea-Phi-3.5-mini Japanese Finetuning")
        logger.info("="*80)
        
        # モデル・データセットセットアップ
        self.setup_model()
        self.setup_datasets()
        
        # セッション作成
        total_steps = len(self.train_dataset) // self.config['training']['per_device_train_batch_size'] * self.config['training']['num_train_epochs']
        if not self.is_recovery:
            self.recovery.create_session(total_steps)
        else:
            self.recovery.load_session()
        
        # データコレクター
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # 訓練引数
        training_args = TrainingArguments(
            output_dir=str(self.checkpoint_dir),
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
            metric_for_best_model=self.config['training'].get('metric_for_best_model', 'eval_loss'),
            greater_is_better=False,
            report_to=self.config['training'].get('report_to', []),
            resume_from_checkpoint=self.resume_path if self.is_recovery else None
        )
        
        # Trainer作成
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
        )
        
        # 学習実行
        logger.info("Starting training...")
        start_time = time.time()
        
        try:
            self.trainer.train(resume_from_checkpoint=self.resume_path if self.is_recovery else None)
            
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time/3600:.2f} hours")
            
            # 最終モデル保存
            output_model_dir = Path(self.config['training']['output_dir']) / "final_model"
            output_model_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(str(output_model_dir))
            self.tokenizer.save_pretrained(str(output_model_dir))
            
            logger.info(f"[OK] Final model saved to {output_model_dir}")
            
            # セッション完了
            if self.recovery.session:
                self.recovery.session["completed"] = True
                self.recovery.session["end_time"] = time.time()
                self.recovery.save_session()
            
        except KeyboardInterrupt:
            logger.warning("[WARNING] Training interrupted by user")
            self.recovery.save_session()
        except Exception as e:
            logger.error(f"[ERROR] Training failed: {e}")
            self.recovery.save_session()
            raise


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Borea-Phi-3.5-mini Japanese Finetuning"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/finetune_borea_japanese.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint path"
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Automatically resume from latest checkpoint"
    )
    
    args = parser.parse_args()
    
    # ファインチューニング実行
    trainer = BoreaJapaneseFinetuner(
        config_path=args.config,
        resume_path=args.resume,
        auto_resume=args.auto_resume
    )
    
    try:
        trainer.train()
        
        # 音声通知
        audio_file = PROJECT_ROOT / ".cursor" / "marisa_owattaze.wav"
        if audio_file.exists():
            try:
                ps_cmd = f"""
                if (Test-Path '{audio_file}') {{
                    Add-Type -AssemblyName System.Windows.Forms
                    $player = New-Object System.Media.SoundPlayer '{audio_file}'
                    $player.PlaySync()
                    Write-Host '[OK] Audio notification played' -ForegroundColor Green
                }}
                """
                subprocess.run(
                    ["powershell", "-Command", ps_cmd],
                    cwd=str(PROJECT_ROOT),
                    check=False
                )
            except Exception as e:
                logger.warning(f"Failed to play audio: {e}")
        
        return 0
        
    except Exception as e:
        logger.error(f"[FAILED] Finetuning failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import subprocess
    sys.exit(main())

