#!/usr/bin/env python3
"""
SO8T/thinking NKAT-SO8T Adapter Training Script (RTX 3060 Optimized)
Enhanced with tqdm progress bars and detailed logging for monitoring
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed
)
import numpy as np
from tqdm import tqdm
import psutil
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("GPUtil not available, GPU monitoring disabled")

# Set console encoding to UTF-8 for Windows compatibility
if os.name == 'nt':
    import codecs
    try:
        codecs.register(lambda name: codecs.lookup('utf-8') if name == 'cp65001' else None)
    except:
        pass

# Local imports
sys.path.append('src')
from layers.nkat_wrapper import NKAT_Wrapper
# SO8T components will be implemented inline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_nkat_so8t.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ProgressMonitor:
    """Enhanced progress monitoring with tqdm and system metrics"""

    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.step_times = []

        # Setup tqdm progress bar (simple format for Windows compatibility)
        self.pbar = tqdm(
            total=total_steps,
            desc="Training Progress",
            unit="step",
            ncols=100,
            ascii=True,
            bar_format='{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | {elapsed} | {rate_fmt} | {postfix}'
        )

    def update(self, step: int, loss: float = None, learning_rate: float = None):
        """Update progress bar with current metrics"""
        self.current_step = step
        elapsed_time = time.time() - self.start_time

        # Calculate metrics
        if loss is not None:
            self.step_times.append(time.time())
            if len(self.step_times) > 1:
                avg_step_time = (self.step_times[-1] - self.step_times[0]) / (len(self.step_times) - 1)
                eta_seconds = (self.total_steps - self.current_step) * avg_step_time
                eta_str = f"{eta_seconds/3600:.1f}h" if eta_seconds > 3600 else f"{eta_seconds/60:.1f}m"
            else:
                eta_str = "?"

        # Get system metrics
        gpu_info = self._get_gpu_info()
        memory_info = self._get_memory_info()

        # Update progress bar
        postfix = {
            'loss': '.4f' if loss is not None else 'N/A',
            'lr': '.2e' if learning_rate is not None else 'N/A',
            'gpu': f"{gpu_info['utilization']}%",
            'mem': f"{gpu_info['memory_used']}MB",
            'temp': f"{gpu_info['temperature']}°C",
            'cpu': f"{memory_info['cpu_percent']:.0f}%",
            'eta': eta_str
        }

        self.pbar.set_postfix(postfix)
        self.pbar.update(1)

        # Log detailed information every 50 steps
        if step % 50 == 0:
            logger.info(f"Step {step}/{self.total_steps} | Loss: {loss:.4f} | LR: {learning_rate:.2e} | "
                       f"GPU: {gpu_info['utilization']}% | Memory: {gpu_info['memory_used']}MB | "
                       f"Temperature: {gpu_info['temperature']}°C | ETA: {eta_str}")

    def _get_gpu_info(self) -> Dict:
        """Get GPU information"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return {
                    'utilization': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'temperature': gpu.temperature
                }
        except:
            pass
        return {'utilization': 0, 'memory_used': 0, 'memory_total': 0, 'temperature': 0}

    def _get_memory_info(self) -> Dict:
        """Get system memory information"""
        memory = psutil.virtual_memory()
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': memory.percent,
            'memory_used': memory.used / 1024 / 1024,  # MB
            'memory_total': memory.total / 1024 / 1024   # MB
        }

    def close(self):
        """Close progress bar"""
        self.pbar.close()
        total_time = time.time() - self.start_time
        logger.info(f"Progress monitoring completed. Total time: {total_time:.2f}s")
class EnhancedTrainer(Trainer):
    """Enhanced Trainer with progress monitoring"""

    def __init__(self, progress_monitor: ProgressMonitor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_monitor = progress_monitor

    def training_step(self, model, inputs):
        """Override training step to update progress"""
        result = super().training_step(model, inputs)

        # Update progress monitor
        current_step = self.state.global_step
        loss = result['loss'].item() if isinstance(result, dict) and 'loss' in result else None
        lr = self.optimizer.param_groups[0]['lr'] if self.optimizer else None

        self.progress_monitor.update(current_step, loss, lr)

        return result

class NKATSO8TTrainer:
    def __init__(
        self,
        model_path: str,
        output_dir: str,
        train_data_path: str,
        max_steps: int = 500,
        batch_size: int = 1,
        gradient_accumulation: int = 8,
        learning_rate: float = 2e-5,
        save_steps: int = 100,
        logging_steps: int = 25,
        warmup_steps: int = 50,
        annealing_warmup: int = 100,
        alpha_init: float = -5.0,
        seed: int = 42
    ):
        self.model_path = model_path
        self.output_dir = output_dir
        self.train_data_path = train_data_path
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.gradient_accumulation = gradient_accumulation
        self.learning_rate = learning_rate
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.warmup_steps = warmup_steps
        self.annealing_warmup = annealing_warmup
        self.alpha_init = alpha_init
        self.seed = seed

        set_seed(seed)

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.setup_model_and_tokenizer()
        self.setup_data()
        self.setup_training()

        logger.info("NKAT-SO8T Trainer initialized successfully")
        logger.info(f"Model: {model_path}")
        logger.info(f"Output dir: {output_dir}")
        logger.info(f"Max steps: {max_steps}")
        logger.info(f"Effective batch size: {batch_size * gradient_accumulation}")

    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer with NKAT-SO8T adapter"""
        logger.info("Loading model and tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        # Freeze base model parameters
        logger.info("Freezing base model parameters...")
        for name, param in self.model.named_parameters():
            if 'model.layers' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        # Add NKAT-SO8T adapter
        self.nkat_adapter = NKAT_Wrapper(
            hidden_size=self.model.config.hidden_size,
            alpha_init=self.alpha_init
        )

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(".2f")

    def setup_data(self):
        """Setup training data"""
        logger.info("Loading training data...")

        with open(self.train_data_path, 'r', encoding='utf-8') as f:
            self.train_data = [json.loads(line) for line in f]

        logger.info(f"Loaded {len(self.train_data)} training samples")

        # Create dataset
        self.dataset = NKATSO8TDataset(self.train_data, self.tokenizer)

    def setup_training(self):
        """Setup training arguments and components"""
        effective_batch_size = self.batch_size * self.gradient_accumulation

        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=1,
            max_steps=self.max_steps,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            save_total_limit=3,
            evaluation_strategy="no",
            fp16=False,
            bf16=True,
            gradient_checkpointing=True,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="none",
            load_best_model_at_end=False,
        )

        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            max_length=2048,
        )

    def train_step_optimized(self, batch):
        """Optimized training step with progress monitoring"""
        try:
            # Move batch to device
            batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**batch, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]  # Last layer

            # NKAT-SO8T processing
            nkat_output = self.nkat_adapter(hidden_states, hidden_states)

            # Compute loss (LM loss + Triality loss with annealing)
            step = getattr(self.trainer.state, 'global_step', 0)
            annealing_weight = min(1.0, step / self.annealing_warmup) * 0.1

            # Language modeling loss
            lm_logits = self.model.lm_head(nkat_output)
            lm_loss = nn.functional.cross_entropy(
                lm_logits.view(-1, lm_logits.size(-1)),
                batch['labels'].view(-1),
                ignore_index=-100
            )

            # Triality loss (placeholder)
            triality_loss = torch.tensor(0.0, device=self.model.device, dtype=torch.bfloat16)

            total_loss = lm_loss + annealing_weight * triality_loss

            # Backward pass
            total_loss.backward()

            return {
                'loss': total_loss.item(),
                'lm_loss': lm_loss.item(),
                'triality_loss': triality_loss.item(),
                'annealing_weight': annealing_weight,
                'alpha_gate': self.nkat_adapter.get_alpha_gate_value().item()
            }

        except Exception as e:
            logger.error(f"Error in training step: {e}")
            raise

    def monitor_system_resources(self) -> Dict:
        """Monitor system resources"""
        try:
            gpu = GPUtil.getGPUs()[0]
            return {
                'gpu_util': gpu.load * 100,
                'gpu_memory_used': gpu.memoryUsed,
                'gpu_memory_total': gpu.memoryTotal,
                'gpu_temp': gpu.temperature,
                'cpu_percent': psutil.cpu_percent(),
                'ram_used': psutil.virtual_memory().used / (1024**3),  # GB
                'ram_total': psutil.virtual_memory().total / (1024**3),  # GB
            }
        except:
            return {
                'gpu_util': 0,
                'gpu_memory_used': 0,
                'gpu_memory_total': 0,
                'gpu_temp': 0,
                'cpu_percent': 0,
                'ram_used': 0,
                'ram_total': 0,
            }

    def train(self):
        """Main training loop with enhanced tqdm and detailed logging"""
        logger.info("Starting NKAT-SO8T training with enhanced monitoring...")

        # Create data loader
        train_dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad] +
            list(self.nkat_adapter.parameters()),
            lr=self.learning_rate
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=self.warmup_steps
        )

        # Enhanced progress monitor
        progress_monitor = ProgressMonitor(self.max_steps)

        try:
            self.model.train()
            step = 0

            while step < self.max_steps:
                for batch in train_dataloader:
                    if step >= self.max_steps:
                        break

                    step_start = time.time()

                    # Training step
                    metrics = self.train_step_optimized(batch)

                    # Gradient accumulation and optimization
                    if (step + 1) % self.gradient_accumulation == 0:
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad] +
                            list(self.nkat_adapter.parameters()),
                            max_norm=1.0
                        )

                        optimizer.step()
                        optimizer.zero_grad()

                        # Update scheduler
                        if step < self.warmup_steps:
                            scheduler.step()

                    # Update progress bar
                    elapsed = time.time() - step_start
                    step_times.append(elapsed)

                    # System resources
                    resources = self.monitor_system_resources()

                    # Update progress monitor with enhanced metrics
                    current_lr = optimizer.param_groups[0]['lr']
                    progress_monitor.update(
                        step + 1,  # +1 because step starts from 0
                        loss=metrics['loss'],
                        learning_rate=current_lr
                    )

                    # Detailed logging (reduced frequency since progress monitor handles most updates)
                    if step % self.logging_steps == 0:
                        avg_step_time = np.mean(step_times[-10:]) if len(step_times) >= 10 else np.mean(step_times)
                        eta_seconds = avg_step_time * (self.max_steps - step)

                        logger.info(
                            f"Step {step}/{self.max_steps} | "
                            f"Loss: {metrics['loss']:.4f} | "
                            f"LM Loss: {metrics['lm_loss']:.4f} | "
                            f"Alpha Gate: {metrics['alpha_gate']:.4f} | "
                            f"Annealing Weight: {metrics['annealing_weight']:.4f} | "
                            f"GPU: {resources['gpu_util']:.1f}% | "
                            f"GPU Mem: {resources['gpu_memory_used']:.0f}MB | "
                            f"GPU Temp: {resources['gpu_temp']:.0f}°C | "
                            f"Step Time: {elapsed:.2f}s | "
                            f"ETA: {eta_seconds/3600:.1f}h"
                        )

                    # Save checkpoint
                    if step % self.save_steps == 0 and step > 0:
                        self.save_checkpoint(step)
                        logger.info(f"Checkpoint saved at step {step}")

                    step += 1

            # Final save
            self.save_checkpoint(step)
            logger.info("Training completed successfully!")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            progress_monitor.close()

            # Final statistics
            total_time = time.time() - start_time
            logger.info(f"Total training time: {total_time/3600:.2f} hours")
            logger.info(f"Average step time: {np.mean(step_times):.2f}s")
            logger.info(f"Final GPU temp: {self.monitor_system_resources()['gpu_temp']:.0f}°C")

    def save_checkpoint(self, step: int):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.output_dir) / f"checkpoint-step-{step}"
        checkpoint_dir.mkdir(exist_ok=True)

        # Save model and adapter
        torch.save({
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'nkat_adapter_state_dict': self.nkat_adapter.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'alpha_gate_value': self.nkat_adapter.get_alpha_gate_value().item(),
        }, checkpoint_dir / "checkpoint.pt")

        # Save config
        config = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'training_args': vars(self.training_args),
        }

        with open(checkpoint_dir / "training_config.json", 'w') as f:
            json.dump(config, f, indent=2, default=str)

        logger.info(f"Checkpoint saved to {checkpoint_dir}")


class NKATSO8TDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Format conversation
        conversation = item.get('conversation', item.get('messages', []))
        if isinstance(conversation, list):
            text = self.tokenizer.apply_chat_template(conversation, tokenize=False)
        else:
            text = str(conversation)

        # Tokenize
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=2048,
            padding=False,
            return_tensors=None
        )

        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': tokenized['input_ids'].copy(),  # For causal LM
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train NKAT-SO8T Adapter")
    parser.add_argument("--model-path", required=True, help="Path to base model")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--train-data", required=True, help="Training data path")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, default=1, help="Per device batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--save-steps", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--logging-steps", type=int, default=25, help="Log every N steps")
    parser.add_argument("--warmup-steps", type=int, default=50, help="Warmup steps")
    parser.add_argument("--annealing-warmup", type=int, default=100, help="Annealing warmup steps")
    parser.add_argument("--alpha-init", type=float, default=-5.0, help="Alpha gate initial value")

    args = parser.parse_args()

    logger.info("Starting NKAT-SO8T training with enhanced monitoring...")
    logger.info(f"Arguments: {vars(args)}")

    trainer = NKATSO8TTrainer(**vars(args))
    trainer.train()

    logger.info("Training completed! Playing notification...")
    # Play notification sound
    try:
        import winsound
        winsound.Beep(800, 500)
    except:
        pass


if __name__ == "__main__":
    main()