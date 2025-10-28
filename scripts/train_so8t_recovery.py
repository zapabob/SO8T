#!/usr/bin/env python3
"""
SO8T Qwen2.5-7B-Instruct 電源断リカバリー学習スクリプト

電源断からの自動復旧機能付きSO8T学習システム
- 5分間隔での自動チェックポイント保存
- 緊急保存機能 (Ctrl+C対応)
- バックアップローテーション (最大10個)
- セッション管理と完全なセッション追跡

Usage:
    python train_so8t_recovery.py
    python train_so8t_recovery.py --resume checkpoints/so8t_qwen2.5-7b_session_20251027_201432
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
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType
import yaml
from tqdm import tqdm
import numpy as np
import pickle

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

from models.so8t_model import SO8TModel, SO8TModelConfig
from training.so8t_dataset_loader import SO8TDataset, collate_so8t_batch
from training.losses import SafetyAwareLoss, SafetyMetrics


class RecoverySO8TTrainer:
    """電源断リカバリー機能付きSO8T Safe Agent Trainer"""
    
    def __init__(self, config_path: str = "configs/training_config.yaml", resume_path: Optional[str] = None):
        """Initialize SO8T trainer with recovery capabilities."""
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.trainer = None
        
        # リカバリー設定
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = Path("checkpoints") / f"so8t_qwen2.5-7b_session_{self.session_id}"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # バックアップローテーション設定
        self.max_backups = 10
        self.backup_interval = 300  # 5分間隔
        
        # 学習状態
        self.current_epoch = 0
        self.current_step = 0
        self.best_safety_score = 0.0
        self.training_history = []
        
        # 復旧フラグ
        self.resume_path = resume_path
        self.is_recovery = resume_path is not None
        
        # シグナルハンドラー設定
        self._setup_signal_handlers()
        
        # 自動保存スレッド
        self.auto_save_thread = None
        self.stop_auto_save = False
        
        logger.info(f"SO8T Recovery Trainer initialized on device: {self.device}")
        logger.info(f"Session ID: {self.session_id}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating emergency save...")
            self._emergency_save()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination
        if hasattr(signal, 'SIGBREAK'):  # Windows
            signal.signal(signal.SIGBREAK, signal_handler)
    
    def _emergency_save(self):
        """Emergency save when interrupted."""
        try:
            logger.info("Performing emergency save...")
            self._save_checkpoint("emergency", is_emergency=True)
            logger.info("Emergency save completed")
        except Exception as e:
            logger.error(f"Emergency save failed: {e}")
    
    def _auto_save_worker(self):
        """Auto-save worker thread."""
        while not self.stop_auto_save:
            time.sleep(self.backup_interval)
            if not self.stop_auto_save:
                try:
                    self._save_checkpoint("auto", is_emergency=False)
                    logger.info(f"Auto-save completed at step {self.current_step}")
                except Exception as e:
                    logger.error(f"Auto-save failed: {e}")
    
    def _save_checkpoint(self, checkpoint_type: str, is_emergency: bool = False):
        """Save training checkpoint."""
        try:
            checkpoint_name = f"{checkpoint_type}_{self.current_epoch}_{self.current_step}"
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pt"
            
            # モデル状態保存
            checkpoint_data = {
                'epoch': self.current_epoch,
                'step': self.current_step,
                'model_state_dict': self.model.state_dict() if self.model else None,
                'optimizer_state_dict': getattr(self, 'task_optimizer', None),
                'safety_optimizer_state_dict': getattr(self, 'safety_optimizer', None),
                'best_safety_score': self.best_safety_score,
                'training_history': self.training_history,
                'config': self.config,
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'checkpoint_type': checkpoint_type,
                'is_emergency': is_emergency
            }
            
            # PyTorch形式で保存
            torch.save(checkpoint_data, checkpoint_path)
            
            # JSON形式でも保存（軽量）
            json_path = self.checkpoint_dir / f"{checkpoint_name}.json"
            json_data = {
                'epoch': self.current_epoch,
                'step': self.current_step,
                'best_safety_score': self.best_safety_score,
                'training_history': self.training_history,
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'checkpoint_type': checkpoint_type,
                'is_emergency': is_emergency
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            # バックアップローテーション
            self._rotate_backups()
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Checkpoint save failed: {e}")
            raise
    
    def _rotate_backups(self):
        """Rotate backup checkpoints to prevent disk space issues."""
        try:
            # 既存のチェックポイントを取得
            checkpoints = list(self.checkpoint_dir.glob("*.pt"))
            checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # 最大数を超える場合は古いものを削除
            if len(checkpoints) > self.max_backups:
                for old_checkpoint in checkpoints[self.max_backups:]:
                    old_checkpoint.unlink()
                    # 対応するJSONファイルも削除
                    json_file = old_checkpoint.with_suffix('.json')
                    if json_file.exists():
                        json_file.unlink()
                    logger.info(f"Removed old checkpoint: {old_checkpoint.name}")
                    
        except Exception as e:
            logger.error(f"Backup rotation failed: {e}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        try:
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            
            checkpoint_data = torch.load(checkpoint_path, map_location=self.device)
            
            # 学習状態を復元
            self.current_epoch = checkpoint_data.get('epoch', 0)
            self.current_step = checkpoint_data.get('step', 0)
            self.best_safety_score = checkpoint_data.get('best_safety_score', 0.0)
            self.training_history = checkpoint_data.get('training_history', [])
            
            logger.info(f"Checkpoint loaded - Epoch: {self.current_epoch}, Step: {self.current_step}")
            logger.info(f"Best safety score: {self.best_safety_score:.4f}")
            
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Checkpoint loading failed: {e}")
            raise
    
    def _setup_model(self):
        """Setup SO8T model with Qwen2.5-7B-Instruct base."""
        logger.info("Setting up SO8T model...")
        
        # Load tokenizer
        model_path = self.config['model']['base_model_name']
        logger.info(f"Loading tokenizer from: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side='right'
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create SO8T model config
        so8t_config = SO8TModelConfig(
            base_model_name=model_path,
            task_head_hidden_size=self.config['model']['task_head_hidden_size'],
            safety_head_hidden_size=self.config['model']['safety_head_hidden_size'],
            safety_num_classes=self.config['model']['safety_num_classes'],
            rationale_max_length=self.config['model']['rationale_max_length'],
            pet_lambda=self.config['model']['pet_lambda'],
            safety_threshold=self.config['model']['safety_threshold'],
            vocab_size=self.config['model']['vocab_size']
        )
        
        # Create SO8T model (base model will be loaded automatically)
        logger.info(f"Creating SO8T model with base: {model_path}")
        self.model = SO8TModel(so8t_config)
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'gradient_checkpointing_enable'):
            self.model.base_model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled for memory efficiency")
        
        # Move model to device (heads are already moved in SO8TModel.__init__)
        # Only move the entire model if needed
        if hasattr(self.model, 'base_model'):
            # Base model is already on device via device_map="auto"
            # Just ensure heads are on the correct device
            device = next(self.model.base_model.parameters()).device
            self.model.task_head_a.to(device)
            self.model.safety_head_b.to(device)
        
        # 復旧時はモデル状態を復元
        if self.is_recovery and self.resume_path:
            checkpoint_data = self._load_checkpoint(self.resume_path)
            if checkpoint_data.get('model_state_dict'):
                self.model.load_state_dict(checkpoint_data['model_state_dict'])
                logger.info("Model state restored from checkpoint")
        
        # Setup QLoRA (RTX3060対応)
        self._setup_qlora()
        
        logger.info("SO8T model setup completed")
    
    def _setup_qlora(self):
        """Setup QLoRA configuration for RTX3060."""
        logger.info("Setting up QLoRA for RTX3060...")
        
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            # QLoRA configuration
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config['qlora']['r'],
                lora_alpha=self.config['qlora']['lora_alpha'],
                lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none",
            )
            
            # Apply QLoRA to base model only (SO8T heads are separate)
            self.model.base_model = get_peft_model(self.model.base_model, lora_config)
            
            # Print trainable parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
            
        except Exception as e:
            logger.warning(f"QLoRA setup failed: {e}")
            logger.info("Training will proceed with full model fine-tuning")
    
    def _setup_lora(self):
        """Setup LoRA configuration (legacy)."""
        logger.info("Setting up LoRA...")
        
        # Qwen2Modelはprepare_inputs_for_generationメソッドがないため、
        # 一旦LoRAを無効化して学習を進める
        logger.warning("LoRA temporarily disabled due to Qwen2Model compatibility issues")
        logger.info("Training will proceed with full model fine-tuning")
        
        # 将来的にLoRAを有効化する場合は、以下のコードを使用
        # lora_config = LoraConfig(
        #     r=self.config['qlora']['r'],
        #     lora_alpha=self.config['qlora']['lora_alpha'],
        #     target_modules=self.config['qlora']['target_modules'],
        #     lora_dropout=self.config['qlora']['lora_dropout'],
        #     bias=self.config['qlora']['bias'],
        #     task_type=TaskType.CAUSAL_LM,
        # )
        # 
        # # Apply LoRA to base model only (not to SO8T heads)
        # if hasattr(self.model, 'base_model'):
        #     self.model.base_model = get_peft_model(self.model.base_model, lora_config)
        # else:
        #     logger.warning("Base model not found, skipping LoRA setup")
        
        logger.info("LoRA setup completed (disabled)")
    
    def _setup_datasets(self):
        """Setup training and validation datasets."""
        logger.info("Setting up datasets...")
        
        # Training dataset
        self.train_dataset = SO8TDataset(
            data_path=self.config['data']['train_data_path'],
            tokenizer=self.tokenizer,
            max_length=self.config['data']['max_length']
        )
        
        # Validation dataset (using same data for now)
        self.val_dataset = SO8TDataset(
            data_path=self.config['data']['val_data_path'],
            tokenizer=self.tokenizer,
            max_length=self.config['data']['max_length']
        )
        
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.val_dataset)}")
    
    def _custom_training_loop(self):
        """Custom training loop for SO8T with safety-aware loss and recovery."""
        logger.info("Starting custom SO8T training loop with recovery...")
        
        # Setup optimizers with explicit float conversion
        learning_rate = float(self.config['training']['learning_rate'])
        safety_learning_rate = float(self.config['training']['safety_learning_rate'])
        weight_decay = float(self.config['training']['weight_decay'])
        
        self.task_optimizer = torch.optim.AdamW(
            self.model.task_head_a.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.safety_optimizer = torch.optim.AdamW(
            self.model.safety_head_b.parameters(),
            lr=safety_learning_rate,
            weight_decay=weight_decay
        )
        
        # 復旧時はオプティマイザー状態を復元
        if self.is_recovery and self.resume_path:
            checkpoint_data = self._load_checkpoint(self.resume_path)
            if checkpoint_data.get('optimizer_state_dict'):
                self.task_optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            if checkpoint_data.get('safety_optimizer_state_dict'):
                self.safety_optimizer.load_state_dict(checkpoint_data['safety_optimizer_state_dict'])
            logger.info("Optimizer states restored from checkpoint")
        
        # Setup loss function
        loss_fn = SafetyAwareLoss(
            task_weight=self.config['loss']['task_weight'],
            safety_weight=self.config['loss']['safety_weight'],
            rationale_weight=self.config['loss']['rationale_weight'],
            pet_weight=self.config['loss']['pet_weight'],
            safety_penalty_weight=self.config['loss']['safety_penalty_weight'],
            escalate_reward_weight=self.config['loss']['escalate_reward_weight']
        )
        
        # Setup metrics
        metrics_fn = SafetyMetrics()
        
        # Training loop
        num_epochs = self.config['training']['num_epochs']
        batch_size = self.config['training']['batch_size']
        
        # Setup mixed precision training for memory efficiency
        scaler = torch.cuda.amp.GradScaler()
        logger.info("Mixed precision training enabled for memory efficiency")
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_so8t_batch
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_so8t_batch
        )
        
        # 自動保存スレッド開始
        self.auto_save_thread = threading.Thread(target=self._auto_save_worker, daemon=True)
        self.auto_save_thread.start()
        
        try:
            for epoch in range(self.current_epoch, num_epochs):
                self.current_epoch = epoch
                logger.info(f"Epoch {epoch + 1}/{num_epochs}")
                
                # Training phase
                self.model.train()
                train_loss = 0.0
                train_metrics = {"ALLOW": 0, "REFUSE": 0, "ESCALATE": 0}
                
                progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
                
                for batch_idx, batch in enumerate(progress_bar):
                    self.current_step += 1
                    
                    # Move batch to device
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    # Forward pass with mixed precision
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['task_labels'],
                            safety_labels=batch['safety_labels'],
                            rationale_labels=batch['rationale_labels'],
                            return_dict=True
                        )
                        
                        # Compute loss
                        losses = loss_fn(
                            task_logits=outputs['task_logits'],
                            safety_logits=outputs['safety_logits'],
                            rationale_logits=outputs['rationale_logits'],
                            task_labels=batch['task_labels'],
                            safety_labels=batch['safety_labels'],
                            rationale_labels=batch['rationale_labels'],
                        hidden_states=outputs.get('hidden_states'),
                        epoch=epoch,
                        total_epochs=num_epochs
                    )
                    
                    # Backward pass with mixed precision
                    self.task_optimizer.zero_grad()
                    self.safety_optimizer.zero_grad()
                    
                    # Scale loss for mixed precision
                    scaler.scale(losses['total_loss']).backward()
                    
                    # Update optimizers with scaling
                    scaler.step(self.task_optimizer)
                    scaler.step(self.safety_optimizer)
                    scaler.update()
                    
                    # Update metrics
                    train_loss += losses['total_loss'].item()
                    
                    # Clear cache to free memory
                    if batch_idx % 5 == 0:  # Every 5 batches
                        torch.cuda.empty_cache()
                    
            # GPU memory monitoring
            gpu_memory_reserved = torch.cuda.memory_reserved(0) / 1024**3  # GiB
            gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GiB
            effective_batch_size = self.config['training']['batch_size'] * self.config['training']['gradient_accumulation_steps']
            
            # Update progress bar with SO8T group structure info
            progress_bar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'safety_loss': f"{losses['safety_loss'].item():.4f}",
                'pet_loss': f"{losses['pet_loss'].item():.4f}",
                'step': self.current_step,
                'SO8T': 'active',
                'GPU_mem': f"{gpu_memory_allocated:.1f}GB",
                'eff_batch': effective_batch_size
            })
            
            # Log memory usage for model card
            if self.current_step % self.config['training']['logging_steps'] == 0:
                logger.info(f"Step {self.current_step}: GPU Memory - Reserved: {gpu_memory_reserved:.2f}GB, Allocated: {gpu_memory_allocated:.2f}GB")
                logger.info(f"Effective batch size: {effective_batch_size}, Quantization: 8bit QLoRA")
                
                # Validation phase
                self.model.eval()
                val_loss = 0.0
                val_metrics = {"ALLOW": 0, "REFUSE": 0, "ESCALATE": 0}
                
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}"):
                        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                for k, v in batch.items()}
                        
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['task_labels'],
                            safety_labels=batch['safety_labels'],
                            rationale_labels=batch['rationale_labels'],
                            return_dict=True
                        )
                        
                        losses = loss_fn(
                            task_logits=outputs['task_logits'],
                            safety_logits=outputs['safety_logits'],
                            rationale_logits=outputs['rationale_logits'],
                            task_labels=batch['task_labels'],
                            safety_labels=batch['safety_labels'],
                            rationale_labels=batch['rationale_labels'],
                            hidden_states=outputs.get('hidden_states'),
                            epoch=epoch,
                            total_epochs=num_epochs
                        )
                        
                        val_loss += losses['total_loss'].item()
                        
                        # Compute safety metrics
                        safety_metrics = metrics_fn.compute_metrics(
                            outputs['safety_logits'],
                            batch['safety_labels']
                        )
                        
                        # Update decision counts
                        predictions = torch.argmax(outputs['safety_logits'], dim=-1)
                        for pred in predictions:
                            decision = self.model.safety_labels[pred.item()]
                            val_metrics[decision] += 1
                
                # Calculate average metrics
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                
                # Calculate safety score
                total_val_samples = sum(val_metrics.values())
                safety_score = val_metrics['REFUSE'] / total_val_samples if total_val_samples > 0 else 0.0
                
                # 学習履歴を記録
                epoch_data = {
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'safety_score': safety_score,
                    'val_metrics': val_metrics,
                    'timestamp': datetime.now().isoformat()
                }
                self.training_history.append(epoch_data)
                
                logger.info(f"Epoch {epoch + 1} Results:")
                logger.info(f"  Train Loss: {avg_train_loss:.4f}")
                logger.info(f"  Val Loss: {avg_val_loss:.4f}")
                logger.info(f"  Safety Score: {safety_score:.4f}")
                logger.info(f"  Val Decisions: {val_metrics}")
                
                # Save best model
                if safety_score > self.best_safety_score:
                    self.best_safety_score = safety_score
                    self._save_checkpoint("best_model", is_emergency=False)
                    logger.info(f"New best model saved with safety score: {safety_score:.4f}")
                
                # エポック終了時のチェックポイント保存
                self._save_checkpoint("epoch_end", is_emergency=False)
                
        finally:
            # 自動保存スレッド停止
            self.stop_auto_save = True
            if self.auto_save_thread:
                self.auto_save_thread.join(timeout=5)
    
    def train(self):
        """Start training process with recovery capabilities."""
        logger.info("Starting SO8T training with recovery...")
        
        try:
            # Start training
            start_time = time.time()
            
            # Custom training loop for SO8T
            self._custom_training_loop()
            
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Save final model
            self._save_checkpoint("final", is_emergency=False)
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            # 緊急保存を試行
            self._emergency_save()
            raise
    
    def run_training(self):
        """Run complete training pipeline with recovery."""
        logger.info("Starting SO8T Qwen2.5-7B-Instruct training pipeline with recovery...")
        
        try:
            # Setup components
            self._setup_model()
            self._setup_datasets()
            
            # Start training
            self.train()
            
            logger.info("SO8T training pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="SO8T Qwen2.5-7B-Instruct Recovery Training")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                       help="Path to training configuration file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = RecoverySO8TTrainer(args.config, args.resume)
    
    # Run training
    trainer.run_training()


if __name__ == "__main__":
    main()
