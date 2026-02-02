#!/usr/bin/env python3
"""
NKAT-SO8T Adapter Training for RTX 3060 (12GB VRAM)
Optimized for 12-hour completion with theoretical phase transition behavior.

Key optimizations:
- Frozen base model (only NKAT adapters trainable)
- Micro-batch training (batch_size=1, grad_accumulation=32)
- BF16 precision with gradient checkpointing
- Annealing scheduler for Alpha Gate natural emergence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import argparse
from contextlib import nullcontext
import gc
import psutil
import os
import sys

# Project imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    default_data_collator,
    get_linear_schedule_with_warmup
)
from src.layers.nkat_wrapper import NKAT_Wrapper

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnnealingScheduler:
    """
    Annealing scheduler for Alpha Gate emergence.

    Implements the theoretical requirement:
    - λ(t) starts at 0.0 (focus on language preservation)
    - Gradually increases to 0.1 (geometric reasoning constraint)
    - Allows natural phase transition of Alpha Gates
    """

    def __init__(self, warmup_steps: int, max_lambda: float = 0.1):
        self.warmup_steps = warmup_steps
        self.max_lambda = max_lambda
        self.current_step = 0

    def step(self) -> float:
        """Get current annealing weight λ(t)."""
        if self.current_step < self.warmup_steps:
            # Linear warmup from 0 to max_lambda
            progress = self.current_step / self.warmup_steps
            lambda_t = self.max_lambda * progress
        else:
            # Maintain maximum weight
            lambda_t = self.max_lambda

        self.current_step += 1
        return lambda_t

    def get_lambda(self) -> float:
        """Get current lambda without advancing step."""
        if self.current_step < self.warmup_steps:
            progress = self.current_step / self.warmup_steps
            return self.max_lambda * progress
        return self.max_lambda


class TrialityLoss(nn.Module):
    """
    Triality Loss: Encourages SO(8) geometric consistency.

    Implements orthogonality and rotation constraints for theoretical correctness.
    """

    def __init__(self, lambda_orthogonality: float = 1.0):
        super().__init__()
        self.lambda_orthogonality = lambda_orthogonality

    def forward(self, adapter: nn.Module) -> torch.Tensor:
        """Compute triality loss for SO(8) constraints."""
        loss = 0.0

        # Orthogonality loss for rotation matrices
        for param_name, param in adapter.named_parameters():
            if 'so8_raw' in param_name:
                # param shape: (num_blocks, 8, 8)
                A = 0.5 * (param - param.transpose(1, 2))  # Skew-symmetric
                R = torch.matrix_exp(A)  # Rotation matrix

                # Orthogonality: R^T @ R = I
                orthogonality_error = torch.norm(
                    torch.matmul(R.transpose(1, 2), R) - torch.eye(8, device=R.device),
                    dim=(1, 2)
                ).mean()

                loss += self.lambda_orthogonality * orthogonality_error

        return loss


class NKATSO8TTrainer:
    """
    Optimized trainer for NKAT-SO8T adapter on RTX 3060.

    Key optimizations:
    - Frozen base model training
    - Gradient accumulation for effective batch size
    - BF16 mixed precision
    - Gradient checkpointing
    - Memory-efficient data loading
    """

    def __init__(
        self,
        model_path: str,
        output_dir: str,
        train_data_path: str,
        max_steps: int = 10000,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 32,
        learning_rate: float = 1e-4,
        warmup_steps: int = 1000,
        save_steps: int = 500,
        logging_steps: int = 100,
        max_grad_norm: float = 1.0,
        annealing_warmup: int = 2000,
    ):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_steps = max_steps
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.effective_batch_size = batch_size * gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.max_grad_norm = max_grad_norm

        # Annealing scheduler for geometric reasoning emergence
        self.annealing_scheduler = AnnealingScheduler(annealing_warmup)

        # Loss components
        self.lm_criterion = nn.CrossEntropyLoss()
        self.triality_criterion = TrialityLoss()

        # Setup device and precision
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        self.use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

        logger.info(f"Using device: {self.device}")
        logger.info(f"BF16 precision: {self.use_bf16}")
        logger.info(f"Effective batch size: {self.effective_batch_size}")

        # Initialize model and tokenizer
        self._setup_model_and_tokenizer()

        # Setup data
        self._setup_data(train_data_path)

        # Setup optimizer and scheduler
        self._setup_optimizer_and_scheduler()

        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        self.start_time = time.time()

    def _setup_model_and_tokenizer(self):
        """Initialize model with NKAT-SO8T wrapper and tokenizer."""
        logger.info("Loading model and tokenizer...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.use_bf16 else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        # Wrap with NKAT-SO8T adapter
        self.model = NKAT_Wrapper(base_model)

        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model.base_model, 'gradient_checkpointing_enable'):
            self.model.base_model.gradient_checkpointing_enable()

        # Move to device
        self.model.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(".1f")

        # Log gate initializations
        initial_gates = self.model.get_gate_values()
        initial_activations = self.model.get_gate_activations()
        logger.info(f"Initial Alpha Gates: {initial_gates[:3]}...")  # Show first 3
        logger.info(f"Initial Gate Activations: {initial_activations[:3]}...")

    def _setup_data(self, train_data_path: str):
        """Setup training data with memory-efficient loading."""
        logger.info("Setting up training data...")

        # Load dataset
        with open(train_data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]

        # Create dataset
        self.dataset = TextDataset(data, self.tokenizer)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=default_data_collator,
            num_workers=0,  # Memory efficient
            pin_memory=False
        )

        logger.info(f"Dataset size: {len(self.dataset)}")
        logger.info(f"Steps per epoch: {len(self.dataloader)}")

    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler."""
        # Only optimize NKAT adapter parameters
        trainable_params = self.model.get_trainable_parameters()

        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        # Learning rate scheduler with warmup
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.max_steps
        )

    def _save_checkpoint(self, step: int, loss: float):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / f"checkpoint-step-{step}"
        checkpoint_path.mkdir(exist_ok=True)

        # Save model state
        torch.save({
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'loss': loss,
            'gate_values': self.model.get_gate_values(),
            'gate_activations': self.model.get_gate_activations(),
        }, checkpoint_path / "checkpoint.pt")

        # Save tokenizer and config
        self.tokenizer.save_pretrained(checkpoint_path)
        self.model.base_model.config.save_pretrained(checkpoint_path)

        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Keep only last 3 checkpoints for disk efficiency
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """Keep only the 3 most recent checkpoints."""
        checkpoints = sorted([
            d for d in self.output_dir.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-step-")
        ], key=lambda x: int(x.name.split("-")[-1]))

        if len(checkpoints) > 3:
            for old_ckpt in checkpoints[:-3]:
                import shutil
                shutil.rmtree(old_ckpt)
                logger.info(f"Removed old checkpoint: {old_ckpt}")

    def _log_progress(self, step: int, loss: float, learning_rate: float, lambda_t: float):
        """Log training progress and gate monitoring."""
        elapsed_time = time.time() - self.start_time
        steps_per_sec = step / elapsed_time if elapsed_time > 0 else 0

        # Monitor Alpha Gates for phase transition
        gate_values = self.model.get_gate_values()
        gate_activations = self.model.get_gate_activations()

        # Calculate phase transition metrics
        avg_activation = sum(gate_activations) / len(gate_activations)
        active_gates = sum(1 for act in gate_activations if act > 0.1)  # Gates that opened

        logger.info(
            f"Step {step}/{self.max_steps} | "
            ".4f"
            ".2e"            ".4f"            ".3f"
        )

        # Log gate status (show first 3 layers)
        logger.info(
            f"Gate Status | α: {gate_values[:3]} | σ(α): {gate_activations[:3]} | "
            ".3f"
        )

        # Memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            logger.info(".2f")

    def train(self):
        """Main training loop with RTX 3060 optimizations."""
        logger.info("Starting NKAT-SO8T training...")
        logger.info("=" * 60)

        self.model.train()
        accumulated_loss = 0.0
        step_loss = 0.0

        data_iter = iter(self.dataloader)

        for step in range(self.max_steps):
            try:
                # Get next batch (with cycling for multiple epochs)
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataloader)
                    batch = next(data_iter)

                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Get current annealing weight
                lambda_t = self.annealing_scheduler.step()

                # Forward pass with mixed precision
                autocast_context = torch.cuda.amp.autocast(dtype=torch.bfloat16) if self.use_bf16 else nullcontext()

                with autocast_context:
                    outputs, gate_values = self.model(**batch)

                    # Language modeling loss
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = batch['labels'][..., 1:].contiguous()
                    lm_loss = self.lm_criterion(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )

                    # Triality loss (geometric constraints)
                    triality_loss = self.triality_criterion(self.model.nkat_adapters[0])  # Sample from first layer

                    # Combined loss with annealing
                    total_loss = lm_loss + lambda_t * triality_loss

                # Backward pass with gradient accumulation
                if self.use_bf16:
                    self.scaler.scale(total_loss / self.gradient_accumulation_steps).backward()
                else:
                    (total_loss / self.gradient_accumulation_steps).backward()

                step_loss += total_loss.item()

                # Gradient accumulation step
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping and optimization
                    if self.use_bf16:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.get_trainable_parameters(), self.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.get_trainable_parameters(), self.max_grad_norm)
                        self.optimizer.step()

                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    accumulated_loss = step_loss / self.gradient_accumulation_steps

                    # Logging
                    if self.global_step % self.logging_steps == 0:
                        current_lr = self.lr_scheduler.get_last_lr()[0]
                        self._log_progress(self.global_step, accumulated_loss, current_lr, lambda_t)

                    # Checkpointing
                    if self.global_step % self.save_steps == 0 and self.global_step > 0:
                        self._save_checkpoint(self.global_step, accumulated_loss)

                    step_loss = 0.0
                    self.global_step += 1

            except Exception as e:
                logger.error(f"Error at step {step}: {e}")
                # Save emergency checkpoint
                self._save_checkpoint(step, accumulated_loss if 'accumulated_loss' in locals() else float('inf'))
                raise

        # Final checkpoint
        self._save_checkpoint(self.max_steps, accumulated_loss)

        # Training completion
        total_time = time.time() - self.start_time
        logger.info("=" * 60)
        logger.info("Training completed!")
        logger.info(f"Training completed in {total_time:.2f} seconds")
        logger.info(f"Final best loss: {self.best_loss:.4f}")
        logger.info(f"Checkpoints saved to: {self.output_dir}")

        # Final gate analysis
        final_gates = self.model.get_gate_values()
        final_activations = self.model.get_gate_activations()
        logger.info("Final Alpha Gate Analysis:")
        logger.info(f"  Average activation: {sum(final_activations)/len(final_activations):.4f}")
        logger.info(f"  Active gates (>0.1): {sum(1 for act in final_activations if act > 0.1)}/{len(final_activations)}")
        logger.info(f"  Max activation: {max(final_activations):.4f}")
        logger.info(f"  Min activation: {min(final_activations):.4f}")


class TextDataset(Dataset):
    """Memory-efficient text dataset for training."""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('text', item.get('input', ''))

        # Tokenize
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
            'labels': tokenized['input_ids'].squeeze()  # For causal LM
        }


def main():
    parser = argparse.ArgumentParser(description="Train NKAT-SO8T Adapter")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to base model")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for checkpoints")
    parser.add_argument("--train-data", type=str, required=True,
                       help="Path to training data (JSONL)")
    parser.add_argument("--max-steps", type=int, default=10000,
                       help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Micro batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=32,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=1000,
                       help="LR scheduler warmup steps")
    parser.add_argument("--annealing-warmup", type=int, default=2000,
                       help="Annealing warmup steps")
    parser.add_argument("--save-steps", type=int, default=500,
                       help="Checkpoint save frequency")
    parser.add_argument("--logging-steps", type=int, default=100,
                       help="Logging frequency")

    args = parser.parse_args()

    # Create trainer
    trainer = NKATSO8TTrainer(
        model_path=args.model_path,
        output_dir=args.output_dir,
        train_data_path=args.train_data,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        annealing_warmup=args.annealing_warmup,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()









