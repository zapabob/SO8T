"""
SO8T QLoRA Training Script

This script implements QLoRA (Quantized Low-Rank Adaptation) training for the SO8T Safe Agent.
It includes safety-first training with PET regularization, dual optimizers, and safety KPI monitoring.

Key features:
- QLoRA for efficient 7B model fine-tuning on RTX3060
- Safety-first loss function with penalties and rewards
- PET regularization for temporal consistency
- Dual optimizers for task and safety heads
- Safety KPI-based early stopping
- Comprehensive logging and checkpointing
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)
from bitsandbytes import BitsAndBytesConfig
import wandb
from tqdm import tqdm
import numpy as np

# Import our modules
from models.so8t_model import SO8TModel, SO8TModelConfig
from training.so8t_dataset_loader import create_train_val_dataloaders, analyze_dataset
from training.losses import SafetyAwareLoss, SafetyMetrics, create_loss_function

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SO8TTrainer:
    """
    SO8T Safe Agent Trainer with QLoRA and safety-first training.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the SO8T trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.loss_fn = None
        self.metrics_fn = None
        self.optimizer = None
        self.scheduler = None
        
        # Training state
        self.current_epoch = 0
        self.best_safety_score = 0.0
        self.training_history = []
        
        # Setup logging
        self._setup_logging()
        
        # Initialize wandb if enabled
        if config.get("use_wandb", False):
            self._setup_wandb()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config["output_dir"]) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler(log_dir / "training.log")
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        wandb.init(
            project=self.config.get("wandb_project", "so8t-safe-agent"),
            name=self.config.get("run_name", "so8t-training"),
            config=self.config
        )
    
    def _load_tokenizer(self):
        """Load and configure tokenizer."""
        logger.info(f"Loading tokenizer from {self.config['base_model_name']}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["base_model_name"],
            trust_remote_code=True
        )
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Tokenizer loaded. Vocab size: {len(self.tokenizer)}")
    
    def _load_model(self):
        """Load and configure the SO8T model with QLoRA."""
        logger.info(f"Loading base model from {self.config['base_model_name']}")
        
        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # Load base model
        base_model = AutoModel.from_pretrained(
            self.config["base_model_name"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Prepare model for k-bit training
        base_model = prepare_model_for_kbit_training(base_model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.get("lora_r", 16),
            lora_alpha=self.config.get("lora_alpha", 32),
            target_modules=self.config.get("lora_target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
            lora_dropout=self.config.get("lora_dropout", 0.1),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA to base model
        self.model = get_peft_model(base_model, lora_config)
        
        # Add SO8T heads
        self._add_so8t_heads()
        
        # Print trainable parameters
        self._print_trainable_parameters()
        
        logger.info("Model loaded and configured with QLoRA")
    
    def _add_so8t_heads(self):
        """Add SO8T dual heads to the model."""
        # Get hidden size from base model
        hidden_size = self.model.config.hidden_size
        
        # Create SO8T heads
        self.model.task_head_a = nn.Linear(hidden_size, self.config.get("task_head_hidden_size", 4096))
        self.model.safety_head_b = nn.Linear(hidden_size, self.config.get("safety_head_hidden_size", 2048))
        
        # Move heads to device
        self.model.task_head_a = self.model.task_head_a.to(self.device)
        self.model.safety_head_b = self.model.safety_head_b.to(self.device)
        
        # Add to trainable parameters
        self.model.trainable_parameters.extend([
            self.model.task_head_a.weight,
            self.model.task_head_a.bias,
            self.model.safety_head_b.weight,
            self.model.safety_head_b.bias
        ])
    
    def _print_trainable_parameters(self):
        """Print information about trainable parameters."""
        trainable_params = 0
        all_param = 0
        
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        logger.info(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_param:,} || "
            f"Trainable%: {100 * trainable_params / all_param:.2f}"
        )
    
    def _setup_loss_and_metrics(self):
        """Setup loss function and metrics."""
        self.loss_fn = create_loss_function(self.config.get("loss_config", {}))
        self.metrics_fn = SafetyMetrics()
        
        logger.info("Loss function and metrics initialized")
    
    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler."""
        # Separate optimizers for task and safety heads
        task_params = list(self.model.task_head_a.parameters())
        safety_params = list(self.model.safety_head_b.parameters())
        
        self.task_optimizer = torch.optim.AdamW(
            task_params,
            lr=self.config.get("task_learning_rate", 1e-4),
            weight_decay=self.config.get("weight_decay", 0.01)
        )
        
        self.safety_optimizer = torch.optim.AdamW(
            safety_params,
            lr=self.config.get("safety_learning_rate", 5e-5),
            weight_decay=self.config.get("weight_decay", 0.01)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.task_optimizer,
            T_max=self.config.get("num_epochs", 10)
        )
        
        logger.info("Optimizers and scheduler initialized")
    
    def _load_data(self):
        """Load training and validation data."""
        logger.info("Loading training data...")
        
        # Analyze dataset
        train_stats = analyze_dataset(self.config["train_data_path"])
        val_stats = analyze_dataset(self.config["val_data_path"])
        
        logger.info(f"Training data: {train_stats['total_samples']} samples")
        logger.info(f"Validation data: {val_stats['total_samples']} samples")
        
        # Create dataloaders
        self.train_dataloader, self.val_dataloader = create_train_val_dataloaders(
            train_data_path=self.config["train_data_path"],
            val_data_path=self.config["val_data_path"],
            tokenizer=self.tokenizer,
            batch_size=self.config.get("batch_size", 4),
            max_length=self.config.get("max_length", 2048),
            num_workers=self.config.get("num_workers", 0),
            include_rationale=self.config.get("include_rationale", True)
        )
        
        logger.info("Data loaded successfully")
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        task_loss = 0.0
        safety_loss = 0.0
        pet_loss = 0.0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch+1}/{self.config['num_epochs']}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["task_labels"],
                safety_labels=batch["safety_labels"],
                rationale_labels=batch.get("rationale_labels"),
                return_dict=True
            )
            
            # Compute loss
            loss_dict = self.loss_fn(
                task_logits=outputs["task_logits"],
                safety_logits=outputs["safety_logits"],
                rationale_logits=outputs.get("rationale_logits"),
                task_labels=batch["task_labels"],
                safety_labels=batch["safety_labels"],
                rationale_labels=batch.get("rationale_labels"),
                hidden_states=outputs["hidden_states"],
                epoch=epoch,
                total_epochs=self.config["num_epochs"]
            )
            
            # Backward pass
            self.task_optimizer.zero_grad()
            self.safety_optimizer.zero_grad()
            
            loss_dict["total_loss"].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get("max_grad_norm", 1.0)
            )
            
            # Update parameters
            self.task_optimizer.step()
            self.safety_optimizer.step()
            
            # Update learning rate
            self.scheduler.step()
            
            # Accumulate losses
            total_loss += loss_dict["total_loss"].item()
            task_loss += loss_dict["task_loss"].item()
            safety_loss += loss_dict["safety_loss"].item()
            pet_loss += loss_dict["pet_loss"].item()
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss_dict['total_loss'].item():.4f}",
                "task": f"{loss_dict['task_loss'].item():.4f}",
                "safety": f"{loss_dict['safety_loss'].item():.4f}",
                "pet": f"{loss_dict['pet_loss'].item():.4f}"
            })
            
            # Log to wandb
            if self.config.get("use_wandb", False):
                wandb.log({
                    "train/loss": loss_dict["total_loss"].item(),
                    "train/task_loss": loss_dict["task_loss"].item(),
                    "train/safety_loss": loss_dict["safety_loss"].item(),
                    "train/pet_loss": loss_dict["pet_loss"].item(),
                    "train/learning_rate": self.scheduler.get_last_lr()[0]
                })
        
        # Calculate average losses
        num_batches = len(self.train_dataloader)
        avg_losses = {
            "total_loss": total_loss / num_batches,
            "task_loss": task_loss / num_batches,
            "safety_loss": safety_loss / num_batches,
            "pet_loss": pet_loss / num_batches
        }
        
        return avg_losses
    
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        all_safety_logits = []
        all_safety_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation", leave=False):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["task_labels"],
                    safety_labels=batch["safety_labels"],
                    rationale_labels=batch.get("rationale_labels"),
                    return_dict=True
                )
                
                # Compute loss
                loss_dict = self.loss_fn(
                    task_logits=outputs["task_logits"],
                    safety_logits=outputs["safety_logits"],
                    rationale_logits=outputs.get("rationale_logits"),
                    task_labels=batch["task_labels"],
                    safety_labels=batch["safety_labels"],
                    rationale_labels=batch.get("rationale_labels"),
                    hidden_states=outputs["hidden_states"],
                    epoch=epoch,
                    total_epochs=self.config["num_epochs"]
                )
                
                total_loss += loss_dict["total_loss"].item()
                
                # Collect predictions for metrics
                all_safety_logits.append(outputs["safety_logits"])
                all_safety_labels.append(batch["safety_labels"])
        
        # Calculate average loss
        num_batches = len(self.val_dataloader)
        avg_loss = total_loss / num_batches
        
        # Calculate safety metrics
        all_safety_logits = torch.cat(all_safety_logits, dim=0)
        all_safety_labels = torch.cat(all_safety_labels, dim=0)
        
        safety_metrics = self.metrics_fn.compute_metrics(
            all_safety_logits,
            all_safety_labels,
            self.config.get("safety_threshold", 0.8)
        )
        
        # Combine metrics
        metrics = {
            "val_loss": avg_loss,
            **safety_metrics
        }
        
        return metrics
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config["output_dir"]) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "task_optimizer_state_dict": self.task_optimizer.state_dict(),
            "safety_optimizer_state_dict": self.safety_optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config
        }
        
        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved at epoch {epoch}")
        
        # Save LoRA adapter
        adapter_path = checkpoint_dir / f"adapter_epoch_{epoch}"
        self.model.save_pretrained(adapter_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _should_early_stop(self, metrics: Dict[str, float]) -> bool:
        """Check if training should stop early based on safety metrics."""
        safety_score = metrics.get("safety_score", 0.0)
        
        # Early stopping if safety score is very low
        if safety_score < 0.3:
            logger.warning(f"Safety score too low: {safety_score:.4f}")
            return True
        
        # Early stopping if no improvement for too long
        if len(self.training_history) > 5:
            recent_scores = [h["val_metrics"]["safety_score"] for h in self.training_history[-5:]]
            if max(recent_scores) - min(recent_scores) < 0.01:
                logger.warning("No improvement in safety score for 5 epochs")
                return True
        
        return False
    
    def train(self):
        """Main training loop."""
        logger.info("Starting SO8T training...")
        
        # Initialize components
        self._load_tokenizer()
        self._load_model()
        self._setup_loss_and_metrics()
        self._setup_optimizer_and_scheduler()
        self._load_data()
        
        # Training loop
        for epoch in range(self.config["num_epochs"]):
            logger.info(f"Epoch {epoch+1}/{self.config['num_epochs']}")
            
            # Train epoch
            train_metrics = self._train_epoch(epoch)
            
            # Validate epoch
            val_metrics = self._validate_epoch(epoch)
            
            # Log metrics
            logger.info(f"Train Loss: {train_metrics['total_loss']:.4f}")
            logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
            logger.info(f"Safety Score: {val_metrics['safety_score']:.4f}")
            logger.info(f"Refuse Recall: {val_metrics['refuse_recall']:.4f}")
            logger.info(f"Escalate Precision: {val_metrics['escalate_precision']:.4f}")
            
            # Save training history
            self.training_history.append({
                "epoch": epoch,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics
            })
            
            # Check if this is the best model
            is_best = val_metrics["safety_score"] > self.best_safety_score
            if is_best:
                self.best_safety_score = val_metrics["safety_score"]
            
            # Save checkpoint
            self._save_checkpoint(epoch, val_metrics, is_best)
            
            # Log to wandb
            if self.config.get("use_wandb", False):
                wandb.log({
                    "epoch": epoch,
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **{f"val/{k}": v for k, v in val_metrics.items()}
                })
            
            # Check for early stopping
            if self._should_early_stop(val_metrics):
                logger.warning("Early stopping triggered")
                break
        
        # Save final model
        self._save_checkpoint(epoch, val_metrics, is_best)
        
        # Save training history
        history_path = Path(self.config["output_dir"]) / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info("Training completed!")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train SO8T Safe Agent with QLoRA")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Base model name")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        config = {}
    
    # Override with command line arguments
    config.update({
        "train_data_path": args.train_data,
        "val_data_path": args.val_data,
        "output_dir": args.output_dir,
        "base_model_name": args.base_model,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "task_learning_rate": args.learning_rate,
        "use_wandb": args.use_wandb
    })
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = SO8TTrainer(config)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()