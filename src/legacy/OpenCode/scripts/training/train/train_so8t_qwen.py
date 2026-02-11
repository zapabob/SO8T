#!/usr/bin/env python3
"""
SO8T Qwen2.5-7B-Instruct 再学習・蒸留スクリプト

Qwen2.5-7B-Instructモデルを使用してSO8T Safe Agentを再学習・蒸留します。
RTX3060級GPUでの効率的な学習を実現します。

Usage:
    python train_so8t_qwen.py
    python train_so8t_qwen.py --config configs/training_config.yaml
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
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

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

from models.so8t_model import SO8TModel, SO8TModelConfig
from training.so8t_dataset_loader import SO8TDataset, collate_so8t_batch
from training.losses import SafetyAwareLoss, SafetyMetrics


class SO8TTrainer:
    """SO8T Safe Agent Trainer for Qwen2.5-7B-Instruct"""
    
    def __init__(self, config_path: str = "configs/training_config.yaml"):
        """Initialize SO8T trainer."""
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.trainer = None
        
        logger.info(f"SO8T Trainer initialized on device: {self.device}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
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
        self.model.to(self.device)
        
        # Setup LoRA
        self._setup_lora()
        
        logger.info("SO8T model setup completed")
    
    def _setup_lora(self):
        """Setup LoRA configuration."""
        logger.info("Setting up LoRA...")
        
        lora_config = LoraConfig(
            r=self.config['qlora']['r'],
            lora_alpha=self.config['qlora']['lora_alpha'],
            target_modules=self.config['qlora']['target_modules'],
            lora_dropout=self.config['qlora']['lora_dropout'],
            bias=self.config['qlora']['bias'],
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA to base model only (not to SO8T heads)
        if hasattr(self.model, 'base_model'):
            self.model.base_model = get_peft_model(self.model.base_model, lora_config)
        else:
            logger.warning("Base model not found, skipping LoRA setup")
        
        logger.info("LoRA setup completed")
    
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
    
    def _setup_training_args(self):
        """Setup training arguments."""
        logger.info("Setting up training arguments...")
        
        output_dir = self.config['output']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config['training']['num_epochs'],
            per_device_train_batch_size=self.config['training']['batch_size'],
            per_device_eval_batch_size=self.config['training']['batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            learning_rate=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            max_grad_norm=self.config['training']['max_grad_norm'],
            warmup_steps=self.config['training']['warmup_steps'],
            lr_scheduler_type=self.config['training']['lr_scheduler_type'],
            save_steps=self.config['training']['save_steps'],
            eval_steps=self.config['training']['eval_steps'],
            logging_steps=self.config['training']['logging_steps'],
            save_total_limit=self.config['training']['save_total_limit'],
            load_best_model_at_end=self.config['training']['load_best_model_at_end'],
            metric_for_best_model=self.config['training']['metric_for_best_model'],
            greater_is_better=self.config['training']['greater_is_better'],
            fp16=self.config['output']['fp16'],
            bf16=self.config['output']['bf16'],
            dataloader_num_workers=self.config['output']['dataloader_num_workers'],
            remove_unused_columns=self.config['output']['remove_unused_columns'],
            report_to=self.config['output']['report_to'],
            logging_dir=self.config['output']['logging_dir'],
            seed=self.config['output']['seed'],
        )
    
    def _setup_trainer(self):
        """Setup trainer with custom data collator and compute metrics."""
        logger.info("Setting up trainer...")
        
        # Custom data collator
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors="pt"
        )
        
        # Custom compute metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            # Implement custom metrics computation
            return {}
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        logger.info("Trainer setup completed")
    
    def train(self):
        """Start training process."""
        logger.info("Starting SO8T training...")
        
        try:
            # Start training
            start_time = time.time()
            
            # Custom training loop for SO8T
            self._custom_training_loop()
            
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Save final model
            self._save_model()
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _custom_training_loop(self):
        """Custom training loop for SO8T with safety-aware loss."""
        logger.info("Starting custom SO8T training loop...")
        
        # Setup optimizers
        task_optimizer = torch.optim.AdamW(
            self.model.task_head_a.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        safety_optimizer = torch.optim.AdamW(
            self.model.safety_head_b.parameters(),
            lr=self.config['training']['safety_learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
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
        
        best_safety_score = 0.0
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_metrics = {"ALLOW": 0, "REFUSE": 0, "ESCALATE": 0}
            
            progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
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
                
                # Backward pass
                task_optimizer.zero_grad()
                safety_optimizer.zero_grad()
                
                losses['total_loss'].backward()
                
                task_optimizer.step()
                safety_optimizer.step()
                
                # Update metrics
                train_loss += losses['total_loss'].item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{losses['total_loss'].item():.4f}",
                    'safety_loss': f"{losses['safety_loss'].item():.4f}",
                    'pet_loss': f"{losses['pet_loss'].item():.4f}"
                })
            
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
            
            logger.info(f"Epoch {epoch + 1} Results:")
            logger.info(f"  Train Loss: {avg_train_loss:.4f}")
            logger.info(f"  Val Loss: {avg_val_loss:.4f}")
            logger.info(f"  Safety Score: {safety_score:.4f}")
            logger.info(f"  Val Decisions: {val_metrics}")
            
            # Save best model
            if safety_score > best_safety_score:
                best_safety_score = safety_score
                self._save_model(f"best_model_epoch_{epoch + 1}")
                logger.info(f"New best model saved with safety score: {safety_score:.4f}")
    
    def _save_model(self, suffix: str = "final"):
        """Save the trained model."""
        logger.info(f"Saving model: {suffix}")
        
        output_dir = Path(self.config['output']['output_dir'])
        model_dir = output_dir / f"so8t_qwen2.5-7b_{suffix}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), model_dir / "pytorch_model.bin")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(model_dir)
        
        # Save config
        with open(model_dir / "config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Model saved to: {model_dir}")
    
    def run_training(self):
        """Run complete training pipeline."""
        logger.info("Starting SO8T Qwen2.5-7B-Instruct training pipeline...")
        
        try:
            # Setup components
            self._setup_model()
            self._setup_datasets()
            self._setup_training_args()
            self._setup_trainer()
            
            # Start training
            self.train()
            
            logger.info("SO8T training pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="SO8T Qwen2.5-7B-Instruct Training")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                       help="Path to training configuration file")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = SO8TTrainer(args.config)
    
    # Run training
    trainer.run_training()


if __name__ == "__main__":
    main()
