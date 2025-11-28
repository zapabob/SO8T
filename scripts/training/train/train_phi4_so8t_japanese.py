#!/usr/bin/env python3
"""
Phi-4 + SO8T 日本語ファインチューニングスクリプト
QLoRA 8bit + PET正規化 + 電源断リカバリー + 3分チェックポイント
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from so8t_core import SO8TRotationGate, PETRegularizer, PETSchedule

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    3分間隔チェックポイント管理（5個ストック）
    """
    
    def __init__(self, checkpoint_dir: Path, max_checkpoints: int = 5, interval_seconds: int = 180):
        """
        Args:
            checkpoint_dir: チェックポイント保存ディレクトリ
            max_checkpoints: 最大保持チェックポイント数
            interval_seconds: チェックポイント間隔（秒）
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.interval_seconds = interval_seconds
        self.last_save_time = time.time()
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def should_save(self) -> bool:
        """チェックポイント保存すべきか判定"""
        return (time.time() - self.last_save_time) >= self.interval_seconds
    
    def save_checkpoint(self, model, tokenizer, optimizer, global_step: int, loss: float):
        """チェックポイント保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{global_step}_{timestamp}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # モデル・トークナイザー保存
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        
        # オプティマイザ状態保存
        torch.save({
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
            'loss': loss,
        }, checkpoint_path / "optimizer.pt")
        
        logger.info(f"[CHECKPOINT] Saved to {checkpoint_path} (step={global_step}, loss={loss:.4f})")
        
        self.last_save_time = time.time()
        
        # 古いチェックポイント削除
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """古いチェックポイントを削除（最新5個保持）"""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*"), key=lambda p: p.stat().st_mtime)
        
        if len(checkpoints) > self.max_checkpoints:
            for old_checkpoint in checkpoints[:-self.max_checkpoints]:
                import shutil
                shutil.rmtree(old_checkpoint)
                logger.info(f"[CLEANUP] Removed old checkpoint: {old_checkpoint.name}")
    
    def get_latest_checkpoint(self) -> Path:
        """最新のチェックポイントを取得"""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*"), key=lambda p: p.stat().st_mtime)
        if checkpoints:
            return checkpoints[-1]
        return None


def add_so8t_to_model(model, hidden_size: int, num_layers: int, device: str):
    """
    モデルにSO8T回転ゲートを追加
    
    Args:
        model: Phi-4モデル
        hidden_size: 隠れ層サイズ
        num_layers: レイヤー数
        device: デバイス
    """
    logger.info(f"[SO8T] Adding SO8T rotation gates to {num_layers} layers...")
    
    for layer_idx in tqdm(range(num_layers), desc="Adding SO8T gates"):
        layer = model.model.layers[layer_idx]
        
        if hasattr(layer, 'self_attn'):
            attn = layer.self_attn
            
            # SO8T回転ゲートを追加
            so8t_gate = SO8TRotationGate(
                hidden_size=hidden_size,
                use_cayley=True,
                orthogonal_regularization=1e-3,
            )
            
            # デバイスに移動
            so8t_gate = so8t_gate.to(device=device, dtype=torch.bfloat16)
            
            attn.so8t_gate = so8t_gate
    
    logger.info("[SO8T] SO8T integration complete")


class SO8TTrainer(Trainer):
    """
    SO8T + PET統合トレーナー
    """
    
    def __init__(self, *args, pet_regularizer=None, checkpoint_manager=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pet_regularizer = pet_regularizer
        self.checkpoint_manager = checkpoint_manager
        
        # シグナルハンドラー登録
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """シグナルハンドラー（電源断対応）"""
        logger.warning(f"\n[SIGNAL] Received signal {signum}, saving checkpoint...")
        if self.checkpoint_manager:
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                tokenizer=self.tokenizer,
                optimizer=self.optimizer,
                global_step=self.state.global_step,
                loss=self.state.log_history[-1].get('loss', 0.0) if self.state.log_history else 0.0,
            )
        logger.info("[EXIT] Emergency checkpoint saved")
        sys.exit(0)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        損失計算（SO8T直交性 + PET正規化を追加）
        """
        # 元の損失計算
        outputs = model(**inputs)
        loss = outputs.loss
        
        # SO8T直交性正則化
        so8t_loss = torch.tensor(0.0, device=loss.device)
        for layer in model.model.layers:
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'so8t_gate'):
                so8t_loss = so8t_loss + layer.self_attn.so8t_gate.get_orthogonality_loss()
        
        # PET正規化
        pet_loss = torch.tensor(0.0, device=loss.device)
        if self.pet_regularizer:
            # 訓練進捗を計算
            progress = self.state.global_step / self.state.max_steps if self.state.max_steps > 0 else 0.0
            
            # hidden_statesに対してPET正規化を適用
            # Note: outputs.hidden_statesが利用可能な場合
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                last_hidden_state = outputs.hidden_states[-1]
                pet_loss = self.pet_regularizer(last_hidden_state, progress)
        
        # 総損失
        total_loss = loss + so8t_loss + pet_loss
        
        # ログ
        if self.state.global_step % 10 == 0:
            logger.info(
                f"[LOSS] step={self.state.global_step}, "
                f"lm_loss={loss.item():.4f}, "
                f"so8t_loss={so8t_loss.item():.4f}, "
                f"pet_loss={pet_loss.item():.4f}, "
                f"total={total_loss.item():.4f}"
            )
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def training_step(self, model, inputs):
        """
        訓練ステップ（チェックポイント管理追加）
        """
        loss = super().training_step(model, inputs)
        
        # 3分間隔チェックポイント
        if self.checkpoint_manager and self.checkpoint_manager.should_save():
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                tokenizer=self.tokenizer,
                optimizer=self.optimizer,
                global_step=self.state.global_step,
                loss=loss.item() if isinstance(loss, torch.Tensor) else loss,
            )
        
        return loss


def main():
    parser = argparse.ArgumentParser(description="Train Phi-4 + SO8T Japanese")
    parser.add_argument("--model_path", type=str, default="Phi-4-mini-instruct", help="Phi-4 model path")
    parser.add_argument("--data_path", type=str, default="data/phi4_japanese_synthetic.jsonl", help="Training data path")
    parser.add_argument("--output_dir", type=str, default="checkpoints/phi4_so8t_japanese", help="Output directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/phi4_so8t_japanese_checkpoints", help="Checkpoint directory")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("Phi-4 + SO8T Japanese Fine-tuning")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Batch size: {args.batch_size} x {args.gradient_accumulation_steps}")
    logger.info(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    logger.info("=" * 70)
    
    # デバイス
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[DEVICE] Using {device}")
    
    # BitsAndBytes設定（8bit量子化）
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
        bnb_8bit_use_double_quant=True,
    )
    
    logger.info("[STEP 1] Loading model and tokenizer...")
    
    # モデル・トークナイザー読み込み
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
    )
    
    logger.info(f"Model loaded: {config.model_type}, hidden_size={config.hidden_size}, layers={config.num_hidden_layers}")
    
    # SO8T統合
    logger.info("[STEP 2] Integrating SO8T...")
    add_so8t_to_model(model, config.hidden_size, config.num_hidden_layers, device)
    
    # LoRA設定
    logger.info("[STEP 3] Preparing LoRA...")
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # PET正規化
    pet_schedule = PETSchedule(
        phase_boundaries=(0.3, 0.7),
        lambdas=(0.01, 0.05, 0.1),
    )
    pet_regularizer = PETRegularizer(schedule=pet_schedule)
    
    # チェックポイント管理
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=Path(args.checkpoint_dir),
        max_checkpoints=5,
        interval_seconds=180,  # 3分
    )
    
    # データセット読み込み
    logger.info("[STEP 4] Loading dataset...")
    dataset = load_dataset('json', data_files=args.data_path, split='train')
    
    def tokenize_function(examples):
        # instruction + input + output を結合
        texts = []
        for i in range(len(examples['instruction'])):
            text = f"{examples['instruction'][i]}\n{examples['output'][i]}"
            texts.append(text)
        
        return tokenizer(
            texts,
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    
    logger.info(f"Dataset size: {len(tokenized_dataset)}")
    
    # 訓練設定
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
    )
    
    # トレーナー
    trainer = SO8TTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        pet_regularizer=pet_regularizer,
        checkpoint_manager=checkpoint_manager,
    )
    
    # 訓練実行
    logger.info("[STEP 5] Starting training...")
    
    if args.resume_from_checkpoint:
        logger.info(f"[RESUME] Resuming from {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()
    
    # 最終保存
    logger.info("[STEP 6] Saving final model...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("[SUCCESS] Training completed!")
    logger.info(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()

