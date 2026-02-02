#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Borea-Phi-3.5-mini-Instruct-JpをSO8Tで再学習

収集・加工済みデータを使用してBorea-Phi-3.5-mini-Instruct-JpモデルをSO8Tで再学習
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
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
    TrainerCallback,
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
import psutil
import gc
from contextlib import contextmanager
import time

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/retrain_borea_phi35_with_so8t.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# チェックポイント設定
CHECKPOINT_INTERVAL = 300  # 5分間隔
MAX_CHECKPOINTS = 10


class SO8TTrainingDataset(Dataset):
    """SO8T学習用データセット"""
    
    def __init__(
        self,
        data_path: Path,
        tokenizer,
        max_length: int = 2048,
        use_quadruple_thinking: bool = True
    ):
        """
        Args:
            data_path: JSONLファイルパス（four_class_*.jsonl形式）
            tokenizer: トークナイザー
            max_length: 最大シーケンス長
            use_quadruple_thinking: 四重推論形式を使用するか
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_quadruple_thinking = use_quadruple_thinking
        self.samples = []
        
        logger.info(f"Loading SO8T training dataset from {data_path}...")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                try:
                    sample = json.loads(line.strip())
                    
                    # テキスト構築
                    instruction = sample.get("instruction", "")
                    input_text = sample.get("input", "")
                    output = sample.get("output", "")
                    
                    # 四重推論形式を使用する場合
                    if self.use_quadruple_thinking and output:
                        # outputに四重推論形式が含まれている場合はそのまま使用
                        text = f"{instruction}\n\n{input_text}\n\n{output}" if instruction else f"{input_text}\n\n{output}"
                    else:
                        # 通常形式
                        if instruction and input_text:
                            text = f"{instruction}\n\n{input_text}\n\n{output}"
                        elif instruction:
                            text = f"{instruction}\n\n{output}"
                        else:
                            text = f"{input_text}\n\n{output}"
                    
                    if text.strip():
                        self.samples.append({
                            "text": text,
                            "four_class_label": sample.get("four_class_label", "ALLOW"),
                            "domain_label": sample.get("domain_label", "general"),
                            "safety_label": sample.get("safety_label", "ALLOW"),
                            "thinking_format": sample.get("thinking_format", "quadruple")
                        })
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_no}: JSON decode error: {e}")
                    continue
        
        logger.info(f"[OK] Loaded {len(self.samples):,} training samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]
        
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
        self.checkpoint_history = deque(maxlen=MAX_CHECKPOINTS)
    
    def should_save_checkpoint(self) -> bool:
        """チェックポイント保存すべきか"""
        return time.time() - self.last_checkpoint_time >= CHECKPOINT_INTERVAL
    
    def save_checkpoint(self, trainer: Trainer, epoch: int, step: int):
        """チェックポイント保存"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}_step_{step}"
        trainer.save_model(str(checkpoint_path))
        
        # チェックポイント履歴に追加
        self.checkpoint_history.append(checkpoint_path)
        
        # 古いチェックポイントを削除
        if len(self.checkpoint_history) > MAX_CHECKPOINTS:
            old_checkpoint = self.checkpoint_history.popleft()
            if old_checkpoint.exists():
                import shutil
                shutil.rmtree(old_checkpoint)
                logger.info(f"[CLEANUP] Removed old checkpoint: {old_checkpoint}")
        
        self.last_checkpoint_time = time.time()
        logger.info(f"[CHECKPOINT] Saved to {checkpoint_path}")


class PerformanceProfiler:
    """パフォーマンスプロファイラー"""
    
    def __init__(self):
        self.metrics = {
            "memory_usage": [],
            "gpu_memory_usage": [],
            "training_speed": [],
            "checkpoints": []
        }
        self.start_time = None
    
    @contextmanager
    def profile_step(self, step_name: str):
        """ステップのプロファイリング"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        else:
            start_gpu_memory = 0
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            if torch.cuda.is_available():
                end_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            else:
                end_gpu_memory = 0
                peak_gpu_memory = 0
            
            elapsed = end_time - start_time
            
            self.metrics["memory_usage"].append({
                "step": step_name,
                "start_mb": start_memory,
                "end_mb": end_memory,
                "delta_mb": end_memory - start_memory,
                "timestamp": datetime.now().isoformat()
            })
            
            if torch.cuda.is_available():
                self.metrics["gpu_memory_usage"].append({
                    "step": step_name,
                    "start_mb": start_gpu_memory,
                    "end_mb": end_gpu_memory,
                    "peak_mb": peak_gpu_memory,
                    "timestamp": datetime.now().isoformat()
                })
            
            logger.info(f"[PROFILE] {step_name}: {elapsed:.2f}s, Memory: {end_memory:.1f}MB, GPU: {end_gpu_memory:.1f}MB")
    
    def record_training_speed(self, samples_per_second: float, tokens_per_second: float):
        """学習速度を記録"""
        self.metrics["training_speed"].append({
            "samples_per_second": samples_per_second,
            "tokens_per_second": tokens_per_second,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_summary(self) -> Dict:
        """サマリー取得"""
        summary = {
            "peak_memory_mb": max([m["end_mb"] for m in self.metrics["memory_usage"]], default=0),
            "peak_gpu_memory_mb": max([m["peak_mb"] for m in self.metrics["gpu_memory_usage"]], default=0) if self.metrics["gpu_memory_usage"] else 0,
            "avg_training_speed": {
                "samples_per_second": np.mean([m["samples_per_second"] for m in self.metrics["training_speed"]]) if self.metrics["training_speed"] else 0,
                "tokens_per_second": np.mean([m["tokens_per_second"] for m in self.metrics["training_speed"]]) if self.metrics["training_speed"] else 0
            }
        }
        return summary


class SO8TRetrainer:
    """SO8T再学習クラス"""
    
    def __init__(
        self,
        base_model_path: Path,
        dataset_path: Path,
        output_dir: Path,
        config: Optional[Dict] = None
    ):
        """
        Args:
            base_model_path: Borea-Phi-3.5-mini-Instruct-Jpモデルパス
            dataset_path: 学習データセットパス
            output_dir: 出力ディレクトリ
            config: 設定辞書
        """
        self.base_model_path = Path(base_model_path)
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        
        logger.info("="*80)
        logger.info("SO8T Retrainer Initialized")
        logger.info("="*80)
        logger.info(f"Base model: {base_model_path}")
        logger.info(f"Dataset: {dataset_path}")
        logger.info(f"Output: {output_dir}")
        
        # デバイス設定
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")
        
        # 電源断リカバリー
        self.recovery = PowerFailureRecovery(self.output_dir / "checkpoints")
        
        # パフォーマンスプロファイラー
        self.profiler = PerformanceProfiler()
    
    def load_model_and_tokenizer(self):
        """モデルとトークナイザーを読み込み"""
        with self.profiler.profile_step("model_loading"):
            logger.info(f"Loading model and tokenizer from {self.base_model_path}...")
            
            # 量子化設定（8bit）
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            
            # トークナイザー読み込み
            tokenizer = AutoTokenizer.from_pretrained(
                str(self.base_model_path),
                trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # モデル読み込み
            model = AutoModelForCausalLM.from_pretrained(
                str(self.base_model_path),
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
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
            
            # 学習可能パラメータ表示
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
            
            logger.info("[OK] Model and tokenizer loaded")
            return model, tokenizer
    
    def prepare_datasets(self, tokenizer):
        """データセット準備"""
        with self.profiler.profile_step("dataset_preparation"):
            logger.info("Preparing datasets...")
            
            # データセット分割（簡易版：80/10/10）
            all_samples = []
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        all_samples.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
            
            import random
            random.seed(42)
            random.shuffle(all_samples)
            
            total = len(all_samples)
            
            # データが少ない場合の処理
            if total == 0:
                raise ValueError(f"No valid samples found in {self.dataset_path}")
            elif total == 1:
                # 1行しかない場合、訓練データとして使用
                logger.warning(f"Only 1 sample found. Using it as training data.")
                train_size = 1
                val_size = 0
            else:
                train_size = int(total * 0.8)
                val_size = int(total * 0.1)
                # 訓練データが0件にならないように調整
                if train_size == 0:
                    train_size = 1
                    val_size = 0
            
            # 一時ファイルに分割保存
            temp_dir = self.output_dir / "temp_splits"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            train_path = temp_dir / "train.jsonl"
            val_path = temp_dir / "val.jsonl"
            test_path = temp_dir / "test.jsonl"
            
            with open(train_path, 'w', encoding='utf-8') as f:
                for sample in all_samples[:train_size]:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            if val_size > 0:
                with open(val_path, 'w', encoding='utf-8') as f:
                    for sample in all_samples[train_size:train_size + val_size]:
                        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            else:
                # 検証データがない場合、空ファイルを作成
                val_path.write_text('', encoding='utf-8')
            
            with open(test_path, 'w', encoding='utf-8') as f:
                for sample in all_samples[train_size + val_size:]:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            # データセット作成
            train_dataset = SO8TTrainingDataset(
                train_path,
                tokenizer,
                max_length=self.config.get("max_seq_length", 2048),
                use_quadruple_thinking=self.config.get("use_quadruple_thinking", True)
            )
            
            val_dataset = SO8TTrainingDataset(
                val_path,
                tokenizer,
                max_length=self.config.get("max_seq_length", 2048),
                use_quadruple_thinking=self.config.get("use_quadruple_thinking", True)
            )
            
            logger.info(f"[OK] Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")
            
            return train_dataset, val_dataset
    
    def train(self):
        """再学習実行"""
        logger.info("="*80)
        logger.info("Starting SO8T Retraining")
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
        
        # 検証データがあるかチェック
        has_val_data = len(val_dataset) > 0
        
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
            eval_strategy="steps" if has_val_data else "no",
            eval_steps=self.config.get("eval_steps", 500) if has_val_data else None,
            fp16=self.config.get("fp16", True),
            bf16=self.config.get("bf16", False),
            gradient_checkpointing=self.config.get("gradient_checkpointing", True),
            optim=self.config.get("optim", "paged_adamw_8bit"),
            report_to=[],
            load_best_model_at_end=has_val_data,
            metric_for_best_model="eval_loss" if has_val_data else None,
            greater_is_better=False
        )
        
        # カスタムコールバック（パフォーマンス計測用）
        class PerformanceCallback(TrainerCallback):
            def __init__(self, profiler):
                super().__init__()
                self.profiler = profiler
                self.step_start_time = None
            
            def on_init_end(self, args, state, control, **kwargs):
                """Trainer初期化完了時のコールバック"""
                # 必要に応じて初期化処理を追加可能
                pass
            
            def on_step_begin(self, args, state, control, **kwargs):
                self.step_start_time = time.time()
            
            def on_step_end(self, args, state, control, **kwargs):
                if self.step_start_time:
                    elapsed = time.time() - self.step_start_time
                    samples_per_second = args.per_device_train_batch_size * args.gradient_accumulation_steps / elapsed if elapsed > 0 else 0
                    # 簡易的なトークン数推定（平均2048トークンと仮定）
                    tokens_per_second = samples_per_second * 2048
                    self.profiler.record_training_speed(samples_per_second, tokens_per_second)
        
        # トレーナー
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset if has_val_data else None,
            data_collator=data_collator,
            callbacks=[PerformanceCallback(self.profiler)]
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
        with self.profiler.profile_step("training"):
            trainer.train()
        
        # 最終モデル保存
        with self.profiler.profile_step("model_saving"):
            final_model_dir = self.output_dir / "final_model"
            trainer.save_model(str(final_model_dir))
            tokenizer.save_pretrained(str(final_model_dir))
            
            # メタデータ保存
            metadata = {
                "base_model": str(self.base_model_path),
                "dataset_path": str(self.dataset_path),
                "config": self.config,
                "timestamp": datetime.now().isoformat(),
                "performance": self.profiler.get_summary()
            }
            
            metadata_path = final_model_dir / "so8t_retraining_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # パフォーマンスレポート保存
            performance_report_path = self.output_dir / "performance_report.json"
            with open(performance_report_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "metrics": self.profiler.metrics,
                    "summary": self.profiler.get_summary()
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[OK] Final model saved to {final_model_dir}")
            logger.info(f"[OK] Performance report saved to {performance_report_path}")
        
        return final_model_dir


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Retrain Borea-Phi-3.5-mini-Instruct-Jp with SO8T")
    parser.add_argument(
        '--base-model',
        type=Path,
        default=Path("models/Borea-Phi-3.5-mini-Instruct-Jp"),
        help='Base model path (models/Borea-Phi-3.5-mini-Instruct-Jp)'
    )
    parser.add_argument(
        '--dataset',
        type=Path,
        required=True,
        help='Training dataset path (JSONL)'
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
    
    # 再学習実行
    retrainer = SO8TRetrainer(
        base_model_path=args.base_model,
        dataset_path=args.dataset,
        output_dir=args.output,
        config=config
    )
    
    final_model_dir = retrainer.train()
    
    logger.info("="*80)
    logger.info("[COMPLETE] SO8T Retraining completed!")
    logger.info(f"Final model: {final_model_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    main()

