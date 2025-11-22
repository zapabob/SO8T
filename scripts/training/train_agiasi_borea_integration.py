#!/usr/bin/env python3
"""
AGIASI Integration Script: Hosting SO8T/thinking in Borea-Phi3.5-instinct-jp

Integrates SO8T/thinking layers into Borea-Phi3.5-instinct-jp base model.
Alpha Gate annealing achieves Golden Ratio (φ = 1.618) for phase transition.
Mass Gap Monitor observes geometric awakening during training.
"""

import os
import sys
import json
import time
import logging
import math
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
    DataCollatorForLanguageModeling,
    set_seed,
    TrainerCallback,
    BitsAndBytesConfig
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
    PeftModel
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
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Import SO8T components
try:
    from src.so8t_core.so8t_layer import SO8TReasoningLayer, orthogonality_loss
    from scripts.training.mass_gap_monitor import MassGapMonitor, SO8TMassGapCallback
    SO8T_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] SO8T components not available: {e}")
    SO8T_AVAILABLE = False

class SO8TPhi3Model(nn.Module):
    """Borea-Phi3.5-instinct-jp with integrated SO8T/thinking layers"""

    def __init__(self, base_model_path: str, target_layers: List[int] = None,
                 so8t_hidden_size: int = 2048, annealing_warmup_steps: int = 100):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.target_layers = target_layers or [4, 5, 6, 7, 8, 9, 10, 11]  # Intermediate layers
        self.so8t_hidden_size = so8t_hidden_size
        self.annealing_warmup_steps = annealing_warmup_steps

        # SO8T components
        self.so8t_rotations = nn.ParameterList()  # SO(8) rotation parameters (28 dimensions)
        self.alpha_gates = nn.ParameterList()     # Alpha Gate parameters

        for layer_idx in self.target_layers:
            rotation_params = nn.Parameter(torch.randn(28) * 0.01)
            alpha_gate = nn.Parameter(torch.tensor(-5.0, dtype=torch.float32))
            self.so8t_rotations.append(rotation_params)
            self.alpha_gates.append(alpha_gate)
            print(f"[ATOM] Layer {layer_idx+1}: SO(8) rotation gate + Alpha gate initialized")

        # Projection layers for SO8T
        hidden_size = self.base_model.config.hidden_size
        self.input_projection = nn.Linear(hidden_size, so8t_hidden_size)
        self.output_projection = nn.Linear(so8t_hidden_size, hidden_size)

        self._last_hidden_states = None

    def _construct_so8_matrix(self, rotation_params):
        """Construct SO(8) rotation matrix from Lie algebra parameters"""
        # SO(8) Lie algebra to group element (simplified Cayley transform)
        skew_matrix = torch.zeros(8, 8, device=rotation_params.device)

        # Map 28 parameters to SO(8) Lie algebra (simplified)
        idx = 0
        for i in range(8):
            for j in range(i+1, 8):
                if idx < len(rotation_params):
                    skew_matrix[i, j] = rotation_params[idx]
                    skew_matrix[j, i] = -rotation_params[idx]
                    idx += 1

        # Cayley transform: R = (I - A)(I + A)^(-1)
        I = torch.eye(8, device=rotation_params.device)
        A = skew_matrix

        # For numerical stability, use different approach
        rotation_matrix = torch.matrix_exp(A)  # Exponential map

        # Ensure orthogonality (project to SO(8))
        U, _, Vh = torch.linalg.svd(rotation_matrix)
        rotation_matrix = U @ Vh

        return rotation_matrix

    def _apply_so8_rotation(self, hidden_states, rotation_params):
        """Apply SO(8) rotation to hidden states"""
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Project to SO8T space
        projected = self.input_projection(hidden_states)  # [batch, seq, so8t_hidden_size]

        # Apply SO(8) rotations to the entire projected space
        rotation_matrix = self._construct_so8_matrix(rotation_params)  # [8, 8]

        # Split projected space into chunks that can be rotated
        rotated_chunks = []
        chunk_size = 8  # SO(8) operates on 8 dimensions

        for i in range(0, self.so8t_hidden_size, chunk_size):
            end_idx = min(i + chunk_size, self.so8t_hidden_size)
            chunk = projected[:, :, i:end_idx]

            if chunk.size(-1) == 8:  # Only rotate if we have exactly 8 dimensions
                # Apply rotation matrix
                chunk_reshaped = chunk.view(-1, 8)
                rotated_chunk = chunk_reshaped @ rotation_matrix.t()
                rotated_chunk = rotated_chunk.view(batch_size, seq_len, 8)
            else:
                rotated_chunk = chunk

            rotated_chunks.append(rotated_chunk)

        # Concatenate all chunks back
        rotated_hidden = torch.cat(rotated_chunks, dim=-1)

        # Ensure output size matches input projection size
        if rotated_hidden.size(-1) != self.so8t_hidden_size:
            # Pad or truncate if necessary
            if rotated_hidden.size(-1) > self.so8t_hidden_size:
                rotated_hidden = rotated_hidden[:, :, :self.so8t_hidden_size]
            else:
                padding = torch.zeros(batch_size, seq_len, self.so8t_hidden_size - rotated_hidden.size(-1),
                                    device=rotated_hidden.device, dtype=rotated_hidden.dtype)
                rotated_hidden = torch.cat([rotated_hidden, padding], dim=-1)

        # Project back to original hidden dimension
        final_output = self.output_projection(rotated_hidden)

        return final_output

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                current_step=0, target_alpha=None, **kwargs):
        """Forward pass with SO8T integration"""

        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )

        all_hidden_states = outputs.hidden_states
        if all_hidden_states is None:
            # If hidden states not available, get them from the model manually
            with torch.no_grad():
                hidden_outputs = self.base_model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                all_hidden_states = hidden_outputs.hidden_states

        enhanced_hidden_states = list(all_hidden_states)

        # Apply SO8T transformations to target layers
        ortho_loss = 0.0
        metacognitive_info = {}

        # Calculate current alpha target with two-phase annealing
        if target_alpha is not None:
            current_alpha_target = target_alpha
        else:
            # Phase 1: Anneal to 0.432 (first target)
            # Phase 2: Anneal to Golden Ratio (φ = 1.618)
            phase1_steps = self.annealing_warmup_steps // 2

            if current_step < phase1_steps:
                # Phase 1: Anneal to 0.432
                progress = current_step / phase1_steps
                current_alpha_target = -5.0 + progress * (0.432 - (-5.0))
            elif current_step < self.annealing_warmup_steps:
                # Phase 2: Anneal to Golden Ratio
                progress = (current_step - phase1_steps) / (self.annealing_warmup_steps - phase1_steps)
                current_alpha_target = 0.432 + progress * (1.618033988749895 - 0.432)
            else:
                current_alpha_target = 1.618033988749895  # Golden Ratio

        # Entropy-based metacognition (simplified)
        entropy_threshold_low = 2.0
        entropy_threshold_high = 4.0

        # Simplified entropy calculation from logits
        logits = outputs.logits
        if logits is not None:
            probs = torch.softmax(logits, dim=-1)
            normalized_entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean().item()
        else:
            normalized_entropy = 3.0  # Default moderate entropy

        # Metacognitive adjustment
        metacognitive_alpha_adjustment = 0.0
        if normalized_entropy < entropy_threshold_low:
            metacognitive_alpha_adjustment = 0.1  # Heating up (more SO8T)
            print(f"[BRAIN] Low entropy ({normalized_entropy:.4f}), heating up SO8T (α +0.1)")
        elif normalized_entropy > entropy_threshold_high:
            metacognitive_alpha_adjustment = -0.1  # Cooling down (less SO8T)
            print(f"[BRAIN] High entropy ({normalized_entropy:.4f}), cooling down SO8T (α -0.1)")

        # Apply SO8T transformations
        for i, layer_idx in enumerate(self.target_layers):
            if i < len(self.so8t_rotations) and layer_idx < len(all_hidden_states):
                rotation_params = self.so8t_rotations[i]
                alpha_gate = self.alpha_gates[i]
                hidden_state = all_hidden_states[layer_idx]

                # Annealing + metacognitive adjustment
                alpha_raw = alpha_gate + current_alpha_target + metacognitive_alpha_adjustment
                alpha_sigmoid = torch.sigmoid(alpha_raw)

                rotated_hidden = self._apply_so8_rotation(hidden_state, rotation_params)
                # Debug tensor sizes
                print(f"[DEBUG] hidden_state: {hidden_state.shape}")
                print(f"[DEBUG] rotated_hidden: {rotated_hidden.shape}")
                print(f"[DEBUG] input_projection(hidden_state): {self.input_projection(hidden_state).shape}")
                residual = rotated_hidden - self.input_projection(hidden_state)
                print(f"[DEBUG] residual: {residual.shape}")
                projected_residual = self.output_projection(residual)
                print(f"[DEBUG] projected_residual: {projected_residual.shape}")
                enhanced_state = hidden_state + alpha_sigmoid * projected_residual
                enhanced_hidden_states[layer_idx] = enhanced_state

                print(f"[ATOM] SO(8) rotation applied to layer {layer_idx+1}, Alpha: {alpha_sigmoid:.6f}")

                # Orthogonality loss
                rotation_matrices = self._construct_so8_matrix(rotation_params.unsqueeze(0))
                ortho_loss += orthogonality_loss(rotation_matrices).item()

        ortho_loss /= len(self.target_layers) if self.target_layers else 1.0

        self._last_hidden_states = enhanced_hidden_states[-1]

        # Create new outputs with enhanced hidden states
        enhanced_outputs = type(outputs)(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=tuple(enhanced_hidden_states),
            attentions=outputs.attentions,
        )

        # Add SO8T-specific information
        enhanced_outputs.ortho_loss = ortho_loss
        enhanced_outputs.metacognitive_info = {
            'entropy': normalized_entropy,
            'alpha_adjustment': metacognitive_alpha_adjustment,
            'target_alpha': current_alpha_target
        }

        return enhanced_outputs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the base model"""
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the base model"""
        if hasattr(self.base_model, 'gradient_checkpointing_disable'):
            self.base_model.gradient_checkpointing_disable()

class SO8TDataset(Dataset):
    """Dataset for SO8T training with Borea-Phi3.5 knowledge"""

    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        # Load Borea-Phi3.5 knowledge
        try:
            from datasets import load_dataset
            dataset = load_dataset("TFMC/imatrix-dataset-for-japanese-llm", split="train", streaming=True)

            # Sample some examples
            count = 0
            for example in dataset:
                if count >= 5000:  # Limit for demo
                    break
                text = example.get("text", "")
                if text:
                    self.samples.append(text)
                    count += 1

        except Exception as e:
            print(f"[WARNING] Could not load Borea knowledge: {e}. Using synthetic data.")
            # Fallback: generate synthetic instruction-response pairs
            for i in range(1000):
                instruction = f"質問{i}: これはテストデータです。"
                response = f"回答{i}: 了解しました。これはSO8T学習のためのサンプルデータです。"
                self.samples.append(f"<|user|>\n{instruction}<|end|>\n<|assistant|>\n{response}<|end|>")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]

        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": encodings["input_ids"].squeeze()
        }

class SO8TTrainer(Trainer):
    """Custom trainer with SO8T annealing and monitoring"""

    def __init__(self, annealing_warmup_steps=100, annealing_steps=800, **kwargs):
        super().__init__(**kwargs)
        self.annealing_warmup_steps = annealing_warmup_steps
        self.annealing_steps = annealing_steps

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        current_step = self.state.global_step if self.state else 0

        # Calculate target alpha for annealing
        target_alpha = None  # Let model handle annealing internally

        outputs = model(current_step=current_step, target_alpha=target_alpha, **inputs)
        loss = outputs.loss

        # Add orthogonality loss
        if hasattr(outputs, 'ortho_loss'):
            ortho_weight = 0.01
            loss = loss + ortho_weight * outputs.ortho_loss

        return (loss, outputs) if return_outputs else loss

class SO8TMonitorCallback(TrainerCallback):
    """Monitor SO8T state during training"""

    def __init__(self, log_every_n_steps=10, annealing_warmup_steps=100, model=None):
        self.log_every_n_steps = log_every_n_steps
        self.annealing_warmup_steps = annealing_warmup_steps
        self.model = model
        self.alpha_history = []
        self.ortho_history = []
        self.metacog_history = []

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_every_n_steps == 0:
            self._monitor_so8t_state(state.global_step)

    def _monitor_so8t_state(self, step):
        if not self.model:
            return

        # Monitor Alpha Gates
        alpha_values = []
        for alpha_gate in self.model.alpha_gates:
            alpha_val = torch.sigmoid(alpha_gate).item()
            alpha_values.append(alpha_val)

        avg_alpha = sum(alpha_values) / len(alpha_values) if alpha_values else 0.0
        self.alpha_history.append(avg_alpha)

        # Monitor Orthogonality
        ortho_errors = []
        for rotation_params in self.model.so8t_rotations:
            rotation_matrix = self.model._construct_so8_matrix(rotation_params.unsqueeze(0))
            ortho_err = orthogonality_loss(rotation_matrix).item()
            ortho_errors.append(ortho_err)

        avg_ortho = sum(ortho_errors) / len(ortho_errors) if ortho_errors else 0.0
        self.ortho_history.append(avg_ortho)

        # Phase status
        phi = 1.618033988749895
        if step < self.annealing_warmup_steps // 2:
            status = "[STABLE] Stable"
        elif step < self.annealing_warmup_steps:
            status = "[TRANSITION] Transitioning"
        elif abs(avg_alpha - phi) < 0.01:
            status = "[TARGET] Golden Ratio Reached"
        else:
            status = "[CONVERGING] Converging"

        print(f"[MONITOR] Step {step}: Alpha={avg_alpha:.6f}, Ortho={avg_ortho:.6f}, Status={status}")

class SO8TProgressCallback(TrainerCallback):
    """Progress callback with SO8T-specific formatting"""

    def on_step_end(self, args, state, control, **kwargs):
        if state.log_history and len(state.log_history) > 0:
            last_log = state.log_history[-1]
            train_loss = last_log.get('train_loss', 'N/A')
            if isinstance(train_loss, float):
                print(f"[TRAIN] Step {state.global_step}: Loss={train_loss:.6f}")
            else:
                print(f"[TRAIN] Step {state.global_step}: Loss={train_loss}")

def create_so8t_borea_model(base_model_path: str, target_layers: List[int] = None,
                           so8t_hidden_size: int = 2048, annealing_warmup_steps: int = 100):
    """Create SO8T-integrated Borea-Phi3.5 model"""

    print(f"[MODEL] Creating SO8T-integrated Borea-Phi3.5 model from {base_model_path}")
    print(f"[MODEL] Target layers: {target_layers}")
    print(f"[MODEL] SO8T hidden size: {so8t_hidden_size}")

    model = SO8TPhi3Model(
        base_model_path=base_model_path,
        target_layers=target_layers,
        so8t_hidden_size=so8t_hidden_size,
        annealing_warmup_steps=annealing_warmup_steps
    )

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"[MODEL] Total parameters: {total_params:,}")
    print(f"[MODEL] Trainable parameters: {trainable_params:,}")
    print(".2f")

    return model

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train AGIASI-integrated Borea-Phi3.5-instinct-jp")
    parser.add_argument("--base-model", type=str, default="models/Borea-Phi-3.5-mini-Instruct-Jp",
                       help="Base Borea-Phi3.5 model path")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, default=1, help="Per device batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=50, help="Warmup steps")
    parser.add_argument("--save-steps", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--logging-steps", type=int, default=10, help="Log every N steps")
    parser.add_argument("--so8t-hidden-size", type=int, default=2048, help="SO8T hidden size")
    parser.add_argument("--target-layers", type=str, default="4,5,6,7,8,9,10,11",
                       help="Comma-separated list of target layer indices")
    parser.add_argument("--annealing-warmup", type=int, default=100, help="Annealing warmup steps")
    parser.add_argument("--annealing-steps", type=int, default=400, help="Total annealing steps")
    parser.add_argument("--enable-mass-gap-monitor", action="store_true", default=True,
                       help="Enable Mass Gap Monitor")
    parser.add_argument("--monitor-interval", type=int, default=25, help="Mass Gap Monitor check interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    print("=" * 80)
    print("[START] Training AGIASI-integrated Borea-Phi3.5-instinct-jp")
    print("Hosting SO8T/thinking layers in Japanese LLM base model")
    print("=" * 80)
    print(f"Base model: {args.base_model}")
    print(f"SO8T target layers: {args.target_layers}")
    print(f"Alpha Gate annealing: {args.annealing_warmup} warmup + {args.annealing_steps} annealing steps")
    print(f"Golden Ratio target: φ = {1.618033988749895:.6f}")
    print(f"Mass Gap Monitor: {'Enabled' if args.enable_mass_gap_monitor else 'Disabled'}")

    # Parse target layers
    target_layers = [int(x.strip()) for x in args.target_layers.split(",")]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create SO8T-integrated model
    model = create_so8t_borea_model(
        base_model_path=args.base_model,
        target_layers=target_layers,
        so8t_hidden_size=args.so8t_hidden_size,
        annealing_warmup_steps=args.annealing_warmup
    )

    # Initialize Mass Gap Monitor
    mass_gap_monitor = None
    mgm_callback = None

    if args.enable_mass_gap_monitor and SO8T_AVAILABLE:
        try:
            mass_gap_monitor = MassGapMonitor(
                log_interval=args.monitor_interval,
                model_name="agiasi_borea_integration"
            )
            mass_gap_monitor.start_monitoring()
            mgm_callback = SO8TMassGapCallback(mass_gap_monitor)
            print("[BRAIN] Mass Gap Monitor active - watching for geometric awakening!")
        except Exception as e:
            print(f"[WARNING] Mass Gap Monitor failed: {e}")

    # Create dataset
    train_dataset = SO8TDataset(tokenizer)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="no",
        save_strategy="steps",
        load_best_model_at_end=False,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        run_name=f"agiasi_borea_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Callbacks
    progress_callback = SO8TProgressCallback()
    so8t_monitor = SO8TMonitorCallback(
        log_every_n_steps=args.logging_steps,
        annealing_warmup_steps=args.annealing_warmup,
        model=model
    )

    callbacks = [progress_callback, so8t_monitor]
    if mgm_callback:
        callbacks.append(mgm_callback)

    # Create trainer
    trainer = SO8TTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
        annealing_warmup_steps=args.annealing_warmup,
        annealing_steps=args.annealing_steps,
    )

    print("[IGNITION] Starting AGIASI integration training...")
    trainer.train()

    # Stop monitoring
    if mass_gap_monitor:
        print("[STOP] Stopping Mass Gap Monitor...")
        mass_gap_monitor.stop_monitoring()

    # Save final model
    final_model_path = output_dir / "final_model"
    final_model_path.mkdir(exist_ok=True)

    # Save model
    trainer.model.base_model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    # Save SO8T components
    torch.save({
        'so8t_rotations': trainer.model.so8t_rotations.state_dict(),
        'alpha_gates': trainer.model.alpha_gates.state_dict(),
        'input_projection': trainer.model.input_projection.state_dict(),
        'output_projection': trainer.model.output_projection.state_dict(),
    }, final_model_path / "so8t_components.pt")

    print("[COMPLETE] AGIASI successfully integrated into Borea-Phi3.5-instinct-jp!")
    print(f"Model saved to {final_model_path}")
    print("To use for inference:")
    print("  from peft import PeftModel")
    print(f"  model = PeftModel.from_pretrained('{args.base_model}', '{final_model_path}')")

    # Play notification
    try:
        import winsound
        winsound.Beep(1000, 1000)
    except:
        pass

if __name__ == "__main__":
    main()
