#!/usr/bin/env python3
"""
SO8T/thinking QLoRA Fine-tuning Script

Fine-tunes Borea-Phi3.5-instinct-jp with SO8T/thinking layers using QLoRA.
Base model is frozen, only SO8T adapters and LoRA parameters are trainable.
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
if os.name == 'nt':
    import codecs
    try:
        codecs.register(lambda name: codecs.lookup('utf-8') if name == 'cp65001' else None)
    except:
        pass

# Local imports
sys.path.append('src')
sys.path.append('src/so8t_core')
try:
    from so8t_core.so8t_layer import SO8TReasoningLayer, orthogonality_loss, triality_consistency_loss
    from transformers import TrainerCallback
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Import Mass Gap Monitor
    try:
        from scripts.training.mass_gap_monitor import MassGapMonitor, SO8TMassGapCallback
    except ImportError:
        # Fallback import
        import sys
        sys.path.append('scripts/training')
        from mass_gap_monitor import MassGapMonitor, SO8TMassGapCallback

except ImportError:
    # Direct import fallback
    import importlib.util
    spec = importlib.util.spec_from_file_location("so8t_layer", "src/so8t_core/so8t_layer.py")
    so8t_layer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(so8t_layer)
    SO8TReasoningLayer = so8t_layer.SO8TReasoningLayer
    orthogonality_loss = so8t_layer.orthogonality_loss
    triality_consistency_loss = so8t_layer.triality_consistency_loss

class SO8TMonitorCallback(TrainerCallback):
    """Callback for monitoring SO8T/thinking model internals during training."""

    def __init__(self, log_every_n_steps=25, annealing_warmup_steps=100, model=None):
        self.log_every_n_steps = log_every_n_steps
        self.annealing_warmup_steps = annealing_warmup_steps
        self.model = model

        # Monitoring data storage
        self.alpha_gate_values = []
        self.orthogonality_losses = []
        self.logit_stats = []
        self.step_numbers = []
        self.layer_activations = []  # For monitoring SO8T layer activations
        self.metacognitive_data = []  # For monitoring entropy and dynamic adjustments

        # Create monitoring directory
        self.monitor_dir = Path("monitoring")
        self.monitor_dir.mkdir(exist_ok=True)

        logger.info(f"[MONITOR] SO8T monitoring initialized - log every {log_every_n_steps} steps")

    def on_step_end(self, args, state, control, **kwargs):
        """Monitor SO8T internals at the end of each step."""
        current_step = state.global_step

        if current_step % self.log_every_n_steps == 0:
            self._monitor_so8t_state(current_step, kwargs.get('model', self.model), kwargs)

    def _monitor_so8t_state(self, step, model, kwargs=None):
        """Monitor SO8T model state: Alpha Gates, orthogonality, logits, metacognition."""
        if model is None:
            return

        try:
            # 1. Monitor Alpha Gate values (annealing)
            alpha_values = []
            if hasattr(model, 'alpha_gates'):
                for i, alpha_gate in enumerate(model.alpha_gates):
                    alpha_raw = alpha_gate.item()
                    alpha_sigmoid = torch.sigmoid(torch.tensor(alpha_raw)).item()
                    alpha_values.append(alpha_sigmoid)

                logger.info(f"[MONITOR] Step {step} - Layer {i} Alpha Gate: raw={alpha_raw:.4f}, sigmoid={alpha_sigmoid:.6f}")

            avg_alpha = np.mean(alpha_values) if alpha_values else 0.0
            self.alpha_gate_values.append(avg_alpha)

            # 2. Monitor orthogonality loss for all SO8T layers
            ortho_losses = []
            if hasattr(model, 'so8t_rotations'):
                for i, rotation_params in enumerate(model.so8t_rotations):
                    # Construct SO(8) matrix and compute orthogonality loss
                    rot_matrix = model._construct_so8_matrix(rotation_params.unsqueeze(0))
                    identity = torch.eye(8, device=rot_matrix.device)

                    # Compute orthogonality loss: ||R^T R - I||_F
                    ortho_loss = torch.norm(rot_matrix.T @ rot_matrix - identity, p='fro').item()
                    ortho_losses.append(ortho_loss)

                    logger.info(f"[MONITOR] Step {step} - Layer {i} Orthogonality Loss: {ortho_loss:.6f}")
                    logger.info(f"[MONITOR] Step {step} - Layer {i} Rotation Parameters: {rotation_params.data.cpu().numpy()}")
                    logger.info(f"[MONITOR] Step {step} - Layer {i} Rotation Matrix: {rot_matrix.data.cpu().numpy()}")
                    logger.info(f"[MONITOR] Step {step} - Layer {i} Identity Matrix: {identity.data.cpu().numpy()}") 

            avg_ortho_loss = np.mean(ortho_losses) if ortho_losses else 0.0
            self.orthogonality_losses.append(avg_ortho_loss)

            # 3. Monitor logit statistics
            try:
                # Get logits from the model's last output (if available in kwargs)
                logits = kwargs.get('outputs', {}).get('logits') if 'outputs' in kwargs else None
                if logits is not None:
                    # Calculate logit statistics
                    logit_mean = logits.mean().item()
                    logit_std = logits.std().item()
                    logit_max = logits.max().item()
                    logit_min = logits.min().item()


                    logger.info(f"[MONITOR] Step {step} - Logits: {logits.data.cpu().numpy()}")
                    logit_stats = {
                        'mean': logit_mean,
                        'std': logit_std,
                        'max': logit_max,
                        'min': logit_min
                    }
                    self.logit_stats.append(logit_stats)

                    
                    logger.info(f"[MONITOR] Step {step} - Logits: mean={logit_mean:.4f}, std={logit_std:.4f}, range=[{logit_min:.2f}, {logit_max:.2f}]")    
                else:
                    logger.debug(f"[MONITOR] Step {step} - No logits available for monitoring")
            except Exception as e:
                logger.warning(f"[MONITOR] Failed to monitor logits at step {step}: {e}")

            # 4. Monitor metacognitive adjustments
            metacognitive_info = kwargs.get('metacognitive_info', {})
            self.metacognitive_data.append({
                'step': step,
                'entropy': metacognitive_info.get('entropy'),
                'alpha_adjustment': metacognitive_info.get('alpha_adjustment'),
                'target_alpha': metacognitive_info.get('target_alpha')
            })

            # 5. Monitor layer activations (store for later analysis)
            layer_activation_summary = {
                'step': step,
                'alpha_gates': alpha_values,
                'orthogonality_losses': ortho_losses
            }
            self.layer_activations.append(layer_activation_summary)

            # Keep only recent data to prevent memory issues
            max_history = 1000
            if len(self.layer_activations) > max_history:
                self.layer_activations = self.layer_activations[-max_history:]

            # 5. Log comprehensive statistics
            self.step_numbers.append(step)

            # Save monitoring data periodically
            if step % (self.log_every_n_steps * 10) == 0:
                self._save_monitoring_plots(step)
                self._log_monitoring_summary(step, alpha_values, ortho_losses)

        except Exception as e:
            logger.warning(f"[MONITOR] Monitoring failed at step {step}: {e}")

    def _save_monitoring_plots(self, step):
        """Save monitoring plots to files."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'SO8T/thinking Model Monitoring - Step {step}', fontsize=16)

            steps = self.step_numbers[-100:]  # Last 100 data points

            # Alpha Gate annealing
            if len(self.alpha_gate_values) > 0:
                alpha_data = self.alpha_gate_values[-100:]
                axes[0, 0].plot(steps, alpha_data, 'b-', linewidth=2, marker='o', markersize=3)
                axes[0, 0].set_title('Alpha Gate Annealing (Average)')
                axes[0, 0].set_xlabel('Training Steps')
                axes[0, 0].set_ylabel('Alpha Value (Sigmoid)')
                axes[0, 0].grid(True, alpha=0.3)

                # Add target annealing schedule (Golden Ratio activation)
                import math
                golden_ratio = (1 + math.sqrt(5)) / 2  # φ ≈ 1.618

                warmup_steps = np.linspace(0, self.annealing_warmup_steps, 100)
                target_values = np.minimum(warmup_steps / self.annealing_warmup_steps * golden_ratio, golden_ratio)
                axes[0, 0].plot(warmup_steps, target_values, 'gold', linewidth=2, linestyle='--', alpha=0.8, label=f'Target (φ={golden_ratio:.3f})')
                axes[0, 0].legend()

            # Orthogonality loss
            if len(self.orthogonality_losses) > 0:
                ortho_data = self.orthogonality_losses[-100:]
                axes[0, 1].plot(steps, ortho_data, 'r-', linewidth=2, marker='s', markersize=3)
                axes[0, 1].set_title('SO(8) Orthogonality Loss')
                axes[0, 1].set_xlabel('Training Steps')
                axes[0, 1].set_ylabel('Orthogonality Loss')
                axes[0, 1].set_yscale('log')
                axes[0, 1].grid(True, alpha=0.3)

            # Metacognitive adjustments
            if len(self.metacognitive_data) > 0:
                entropy_data = [d['entropy'] for d in self.metacognitive_data[-100:] if d['entropy'] is not None]
                if entropy_data:
                    axes[1, 0].plot(steps[-len(entropy_data):], entropy_data, 'purple-', linewidth=2, marker='x', markersize=3)
                    axes[1, 0].set_title('Output Entropy & Metacognition')
                    axes[1, 0].set_ylabel('Entropy (normalized)', color='purple')
                    axes[1, 0].tick_params(axis='y', labelcolor='purple')

                    # Add metacognitive adjustment on secondary y-axis
                    ax2 = axes[1, 0].twinx()
                    adjustment_data = [d['alpha_adjustment'] for d in self.metacognitive_data[-100:]]
                    ax2.plot(steps[-len(adjustment_data):], adjustment_data, 'orange--', linewidth=1, alpha=0.7)
                    ax2.set_ylabel('Alpha Adjustment', color='orange')
                    ax2.tick_params(axis='y', labelcolor='orange')
                else:
                    axes[1, 0].text(0.5, 0.5, 'Metacognitive Data\n(Collection Phase)',
                                  ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12)
                    axes[1, 0].set_title('Metacognitive Monitoring')
            else:
                axes[1, 0].text(0.5, 0.5, 'Metacognitive Data\n(Not Available)',
                              ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12)
                axes[1, 0].set_title('Metacognitive Monitoring')

            # Logit statistics
            if len(self.logit_stats) > 0:
                logit_data = [stats['mean'] for stats in self.logit_stats[-100:]]
                axes[1, 1].plot(steps, logit_data, 'purple-', linewidth=2, marker='D', markersize=3)
                axes[1, 1].set_title('Logit Mean Values')
                axes[1, 1].set_xlabel('Training Steps')
                axes[1, 1].set_ylabel('Logit Mean')
                axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = self.monitor_dir / f"so8t_monitoring_step_{step}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            logger.info(f"[MONITOR] Saved monitoring plots to {plot_path}")

        except Exception as e:
            logger.warning(f"[MONITOR] Failed to save monitoring plots: {e}")

    def _log_monitoring_summary(self, step, alpha_values, ortho_losses):
        """Log comprehensive monitoring summary."""
        summary = f"""
[MONITOR SUMMARY] Step {step}
{'='*50}
Alpha Gates (per layer):
{chr(10).join(f'  Layer {i}: {val:.6f}' for i, val in enumerate(alpha_values))}
  Average: {np.mean(alpha_values):.6f}

SO(8) Orthogonality Losses:
{chr(10).join(f'  Layer {i}: {loss:.6f}' for i, loss in enumerate(ortho_losses))}
  Average: {np.mean(ortho_losses):.6f}

Annealing Progress: {min(step / self.annealing_warmup_steps, 1.0):.2%}
Target Alpha (Golden Ratio): {min(step / self.annealing_warmup_steps * ((1 + math.sqrt(5)) / 2), ((1 + math.sqrt(5)) / 2)):.4f}
{'='*50}
"""
        logger.info(summary)

        # Save to file
        summary_path = self.monitor_dir / "so8t_monitoring_summary.txt"
        with open(summary_path, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {summary}\n")


class SO8TPhi3Model(nn.Module):
    """
    Borea-Phi3.5-instinct-jp with SO8T/thinking layers integrated.
    Base model is frozen, SO8T layers are trainable.
    """

    def __init__(self, base_model_path: str, target_layers: List[int] = None, so8t_hidden_size: int = None):
        super().__init__()

        logger.info("[FIRE] Loading Borea-Phi3.5-instinct-jp base model for SO8T/thinking...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        )

        # 🔒 Freeze ALL base model parameters (完全凍結)
        logger.info("[LOCK] Freezing ALL base model parameters...")
        for param in self.base_model.parameters():
            param.requires_grad = False

        # [TARGET] SO(8)回転を適用する層: 4-11層 (0-indexed: 3-10)
        if target_layers is None:
            self.target_layers = list(range(3, 11))  # 層3-10 (4-11層目)
        else:
            self.target_layers = target_layers

        logger.info(f"[TARGET] Applying SO(8) isometric rotations to layers: {[i+1 for i in self.target_layers]}")

        # SO(8)回転パラメータとAlpha Gateの初期化
        self.so8t_rotations = nn.ParameterList()  # SO(8)回転パラメータ (28次元)
        self.alpha_gates = nn.ParameterList()     # Alpha Gateパラメータ

        for layer_idx in self.target_layers:
            # SO(8)回転ゲート: 28パラメータ (SO(8)のLie代数次元)
            rotation_params = nn.Parameter(torch.randn(28) * 0.01)  # 小さなランダム初期化

            # Alpha Gate: 初期値 -5.0 (sigmoid ≈ 0.006) → 0.432までアニーリング
            alpha_gate = nn.Parameter(torch.tensor(-5.0, dtype=torch.float32))

            self.so8t_rotations.append(rotation_params)
            self.alpha_gates.append(alpha_gate)

            logger.info(f"[ATOM] Layer {layer_idx+1}: SO(8) rotation gate + Alpha gate initialized")

        self.num_so8t_layers = len(self.target_layers)
        logger.info(f"[BRAIN] SO8T/thinking layers initialized: {self.num_so8t_layers} layers with SO(8) isometric rotations")

        # [DNA] Mass Gap Monitor用の隠れ状態保存
        self._last_hidden_states = None

        # Projection layers to match dimensions
        base_hidden_size = self.base_model.config.hidden_size
        if so8t_hidden_size is None:
            so8t_hidden_size = base_hidden_size  # Use base model hidden size if not specified

        if base_hidden_size != so8t_hidden_size:
            self.input_projection = nn.Linear(base_hidden_size, so8t_hidden_size)
            self.output_projection = nn.Linear(so8t_hidden_size, base_hidden_size)
        else:
            self.input_projection = nn.Identity()
            self.output_projection = nn.Identity()

        self.so8t_hidden_size = so8t_hidden_size
        logger.info(f"SO8T layers added: {self.num_so8t_layers} layers with hidden size {so8t_hidden_size}")

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for memory efficiency."""
        self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        current_step=0,
        target_alpha=None,
        **kwargs
    ):
        # Forward through base model to get hidden states
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )

        # Get all hidden states for intermediate processing
        all_hidden_states = outputs.hidden_states  # Tuple of [batch, seq, hidden]

        # Apply SO(8) isometric rotations to target layers
        enhanced_hidden_states = list(all_hidden_states)

        # Metacognition: Calculate output entropy for dynamic alpha adjustment
        metacognitive_alpha_adjustment = 0.0
        if hasattr(outputs, 'logits') and outputs.logits is not None:
            # Calculate entropy from output logits (only for the last token to avoid overhead)
            logits = outputs.logits[:, -1, :]  # [batch, vocab_size]
            probs = torch.softmax(logits, dim=-1)

            # Calculate entropy: H = -∑p_i * log(p_i)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)  # [batch]
            avg_entropy = entropy.mean().item()

            # Normalize entropy (assuming vocab_size effects)
            vocab_size = logits.shape[-1]
            max_entropy = math.log(vocab_size)  # Maximum possible entropy
            normalized_entropy = avg_entropy / max_entropy  # 0-1 scale

            # Metacognitive adjustment based on entropy
            # Low entropy (confident predictions) → Heat up (increase SO8T activation)
            # High entropy (hallucinations) → Cool down (decrease SO8T activation)
            entropy_threshold_low = 0.3  # Low entropy threshold
            entropy_threshold_high = 0.7  # High entropy threshold

            if normalized_entropy < entropy_threshold_low:
                # Low entropy: confident predictions, heat up SO8T
                metacognitive_alpha_adjustment = 0.1  # Positive adjustment
                logger.debug(f"Low entropy detected ({normalized_entropy:.4f}), heating up SO8T (α +0.1)")
            elif normalized_entropy > entropy_threshold_high:
                # High entropy: hallucinations, cool down SO8T
                metacognitive_alpha_adjustment = -0.1  # Negative adjustment
                logger.debug(f"High entropy detected ({normalized_entropy:.4f}), cooling down SO8T (α -0.1)")
            else:
                # Moderate entropy: maintain current activation
                metacognitive_alpha_adjustment = 0.0
                logger.debug(f"Moderate entropy ({normalized_entropy:.4f}), maintaining SO8T activation")
        # [TARGET] Apply SO(8) isometric rotations to target layers
        for i, layer_idx in enumerate(self.target_layers):
            if i < len(self.so8t_rotations):
                rotation_params = self.so8t_rotations[i]
                alpha_gate = self.alpha_gates[i]
                hidden_state = all_hidden_states[layer_idx]

                # [ROCKET] Apply Alpha Gate annealing (0.432 target)
                if target_alpha is not None:
                    alpha_raw = torch.tensor(target_alpha, device=alpha_gate.device, dtype=alpha_gate.dtype)
                else:
                    alpha_raw = alpha_gate

                alpha_sigmoid = torch.sigmoid(alpha_raw)

                # [ATOM] Apply SO(8) isometric rotation (等長変換)
                rotated_hidden = self._apply_so8_rotation(hidden_state, rotation_params)

                # [LINK] Combine with Alpha Gate (幾何学的ブレンド)
                enhanced_state = hidden_state + alpha_sigmoid * (rotated_hidden - hidden_state)

                # Update hidden state
                enhanced_hidden_states[layer_idx] = enhanced_state

                logger.debug(f"[ATOM] SO(8) rotation applied to layer {layer_idx+1}, Alpha: {alpha_sigmoid:.6f}")

        # [DNA] Save last hidden states for Mass Gap Monitor
        self._last_hidden_states = enhanced_hidden_states[-1]

        # Store metacognitive information for monitoring
        metacognitive_info = {
            'entropy': normalized_entropy if 'normalized_entropy' in locals() else None,
            'alpha_adjustment': metacognitive_alpha_adjustment,
            'target_alpha': target_alpha
        }

        # Use enhanced intermediate states for final processing
        # Here we use the last enhanced hidden state
        final_hidden = enhanced_hidden_states[-1]

        # Use enhanced hidden states for LM head
        lm_logits = self.base_model.lm_head(final_hidden)

        # Return metacognitive information along with outputs
        outputs['metacognitive_info'] = metacognitive_info
        outputs['logits'] = lm_logits

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # Add SO8T geometric losses for SO(8) rotation parameters
            ortho_loss = 0
            triality_loss = 0

            # Calculate orthogonality loss for each SO(8) rotation parameter
            for rotation_params in self.so8t_rotations:
                rotation_matrices = self._construct_so8_matrix(rotation_params.unsqueeze(0))
                ortho_loss += orthogonality_loss(rotation_matrices)

            ortho_loss = ortho_loss / len(self.so8t_rotations) if len(self.so8t_rotations) > 0 else 0.0
            triality_loss = triality_consistency_loss(final_hidden, labels)

            # Combine losses with annealing for geometric constraints
            annealing_weight = min(0.1, current_step / 100.0 * 0.1) if current_step else 0.01
            total_loss = loss + annealing_weight * ortho_loss + annealing_weight * triality_loss
        else:
            total_loss = loss

        return {
            'loss': total_loss,
            'logits': lm_logits,
            'hidden_states': outputs.hidden_states,
            'so8t_enhanced': final_hidden
        }

    def _apply_so8_rotation(self, hidden_states: torch.Tensor, rotation_params: torch.Tensor) -> torch.Tensor:
        """
        SO(8) isometric rotationを適用 (等長変換)

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            rotation_params: [28] - SO(8) Lie algebra parameters

        Returns:
            Rotated hidden states: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # 隠れ次元を8次元チャンクに分割してSO(8)回転を適用
        # (SO(8)は8次元空間での回転なので、hidden_sizeを8で割れるように処理)
        chunk_size = 8
        num_chunks = hidden_size // chunk_size

        rotated_chunks = []

        for chunk_idx in range(num_chunks):
            start_dim = chunk_idx * chunk_size
            end_dim = start_dim + chunk_size

            # 8次元チャンクを抽出
            chunk = hidden_states[:, :, start_dim:end_dim]  # [batch, seq, 8]

            # SO(8)回転行列を構成
            rotation_matrix = self._construct_so8_matrix(rotation_params.unsqueeze(0))  # [1, 8, 8]

            # 各シーケンス位置に回転を適用
            chunk_flat = chunk.view(-1, chunk_size)  # [batch*seq, 8]

            # 行列積で回転適用
            rotated_chunk_flat = torch.bmm(
                chunk_flat.unsqueeze(1),  # [batch*seq, 1, 8]
                rotation_matrix.transpose(1, 2).expand(chunk_flat.shape[0], -1, -1)  # [batch*seq, 8, 8]
            ).squeeze(1)  # [batch*seq, 8]

            # 元の形状に戻す
            rotated_chunk = rotated_chunk_flat.view(batch_size, seq_len, chunk_size)
            rotated_chunks.append(rotated_chunk)

        # 回転されていない残りの次元はそのまま
        remaining_dims = hidden_size % chunk_size
        if remaining_dims > 0:
            remaining_chunk = hidden_states[:, :, -remaining_dims:]
            rotated_chunks.append(remaining_chunk)

        # 全てのチャンクを結合
        rotated_hidden = torch.cat(rotated_chunks, dim=-1)

        return rotated_hidden

    def _construct_so8_matrix(self, params: torch.Tensor) -> torch.Tensor:
        """
        SO(8)回転行列をLie代数パラメータから構成

        Args:
            params: [batch_size, 28] - Lie algebra parameters

        Returns:
            rotation_matrix: [batch_size, 8, 8] - SO(8) rotation matrix
        """
        batch_size, num_params = params.shape
        device = params.device

        # SO(8)のLie代数要素 (歪対称行列)
        lie_algebra = torch.zeros(batch_size, 8, 8, device=device, dtype=params.dtype)

        idx = 0
        for i in range(8):
            for j in range(i+1, 8):
                if idx < num_params:
                    lie_algebra[:, i, j] = params[:, idx]
                    lie_algebra[:, j, i] = -params[:, idx]
                    idx += 1

        # 行列指数関数で回転行列を生成 (等長変換)
        rotation_matrix = torch.matrix_exp(lie_algebra)

        return rotation_matrix

    def save_pretrained(self, save_directory: str):
        """Save the SO8T-enhanced model."""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save SO8T rotation parameters and alpha gates
        so8t_state = {
            'so8t_rotations': [rot.data for rot in self.so8t_rotations],
            'alpha_gates': [alpha.data for alpha in self.alpha_gates],
            'input_projection': self.input_projection.state_dict(),
            'output_projection': self.output_projection.state_dict(),
            'config': {
                'target_layers': self.target_layers,
                'num_so8t_layers': self.num_so8t_layers,
                'so8t_hidden_size': self.so8t_hidden_size,
                'base_model_path': self.base_model.config._name_or_path
            }
        }

        torch.save(so8t_state, save_path / "so8t_adapter_state.pt")
        logger.info(f"SO8T adapter state saved to {save_path / 'so8t_adapter_state.pt'}")

    @classmethod
    def from_pretrained(cls, base_model_path: str, so8t_checkpoint_path: str):
        """Load SO8T-enhanced model from checkpoint."""
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        # Load SO8T checkpoint
        so8t_state = torch.load(so8t_checkpoint_path, map_location='cpu')

        # Create instance
        instance = cls(
            base_model_path=base_model_path,
            num_so8t_layers=so8t_state['config']['num_so8t_layers'],
            so8t_hidden_size=so8t_state['config']['so8t_hidden_size']
        )

        # Load SO8T state
        instance.so8t_layers.load_state_dict(so8t_state['so8t_layers'])
        instance.input_projection.load_state_dict(so8t_state['input_projection'])
        instance.output_projection.load_state_dict(so8t_state['output_projection'])

        logger.info(f"SO8T-enhanced model loaded from {so8t_checkpoint_path}")
        return instance

# Setup logging
log_filename = f"so8t_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_filename, mode='w')
    ]
)
logger = logging.getLogger(__name__)

class SO8TProgressCallback(TrainerCallback):
    """Enhanced progress callback for SO8T training."""

    def __init__(self):
        self.start_time = time.time()
        self.step_times = []

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step."""
        elapsed = time.time() - self.start_time
        self.step_times.append(elapsed)

        if len(self.step_times) > 1:
            avg_step_time = (self.step_times[-1] - self.step_times[0]) / (len(self.step_times) - 1)
            current_step = state.global_step
            total_steps = state.max_steps
            remaining_steps = total_steps - current_step
            eta_seconds = avg_step_time * remaining_steps
            eta_str = f"{eta_seconds/3600:.1f}h" if eta_seconds > 3600 else f"{eta_seconds/60:.1f}m"

            # Get GPU info
            gpu_info = self._get_gpu_info()
            memory_info = self._get_memory_info()

            logger.info(
                f"Step {current_step}/{total_steps} | "
                f"Loss: {state.log_history[-1].get('train_loss', 'N/A'):.4f} | "
                f"GPU: {gpu_info['utilization']:.1f}% | "
                f"Memory: {gpu_info['memory_used']:.0f}MB | "
                f"Temperature: {gpu_info['temperature']:.0f}°C | "
                f"ETA: {eta_str}"
            )

    def _get_gpu_info(self):
        """Get GPU information."""
        if not GPU_AVAILABLE:
            return {'utilization': 0, 'memory_used': 0, 'temperature': 0}

        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return {
                    'utilization': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'temperature': gpu.temperature
                }
        except:
            pass
        return {'utilization': 0, 'memory_used': 0, 'temperature': 0}

    def _get_memory_info(self):
        """Get system memory information."""
        memory = psutil.virtual_memory()
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
        }

class SO8TDataset(Dataset):
    """Dataset for SO8T/thinking training."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

        logger.info(f"Loading SO8T dataset from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]

        logger.info(f"Loaded {len(self.data)} SO8T reasoning samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Format conversation for SO8T reasoning
        if 'conversation' in item:
            conversation = item['conversation']
            if isinstance(conversation, list):
                text = self.tokenizer.apply_chat_template(conversation, tokenize=False)
            else:
                text = str(conversation)
        elif 'messages' in item:
            conversation = item['messages']
            text = self.tokenizer.apply_chat_template(conversation, tokenize=False)
        else:
            text = item.get('text', str(item))

        # Add SO8T reasoning markers
        if 'reasoning_type' in item:
            reasoning_type = item['reasoning_type']
            if reasoning_type in ['mathematical', 'geometric', 'logical']:
                text = f"<so8t_reasoning_{reasoning_type}>\n{text}\n</so8t_reasoning_{reasoning_type}>"

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
            'labels': tokenized['input_ids'].squeeze()
        }

class SO8TTrainer(Trainer):
    """Custom trainer for SO8T with geometric losses and alpha gate annealing."""

    def __init__(self, *args, annealing_warmup_steps=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.annealing_warmup_steps = annealing_warmup_steps
        self.current_annealing_step = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss with SO8T geometric constraints and alpha gate annealing with metacognition."""
        # Get current step for annealing
        current_step = getattr(self.state, 'global_step', 0)

        # Two-phase annealing: 0.432 → Golden Ratio
        warmup_steps = getattr(self, 'annealing_warmup_steps', 100)
        phase1_steps = warmup_steps // 2  # First half: slowly reach 0.432
        phase2_steps = warmup_steps - phase1_steps  # Second half: reach golden ratio

        if current_step < phase1_steps:
            # Phase 1: Slowly anneal to 0.432 (moderate activation)
            progress = current_step / phase1_steps
            target_alpha = 0.432 * progress  # Gradual increase to 0.432
        else:
            # Phase 2: Anneal from 0.432 to golden ratio
            phase2_progress = (current_step - phase1_steps) / phase2_steps
            golden_ratio = (1 + math.sqrt(5)) / 2
            target_alpha = 0.432 + (golden_ratio - 0.432) * min(phase2_progress, 1.0)

        # Pass target alpha to model for annealing
        outputs = model(current_step=current_step, target_alpha=target_alpha, **inputs)

        # Standard language modeling loss
        lm_loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss

        # Add SO8T geometric losses
        hidden_states = outputs['hidden_states'] if isinstance(outputs, dict) else outputs.hidden_states

        if hidden_states is not None and len(hidden_states) > 0:
            # Orthogonality loss for SO(8) rotations
            ortho_loss = 0
            if hasattr(model, 'so8t_rotations'):
                for rotation_params in model.so8t_rotations:
                    rotation_matrices = model._construct_so8_matrix(rotation_params.unsqueeze(0))
                    ortho_loss += orthogonality_loss(rotation_matrices)

                ortho_loss = ortho_loss / len(model.so8t_rotations) if len(model.so8t_rotations) > 0 else 0.0

            # Triality consistency loss
            last_hidden = hidden_states[-1] if isinstance(hidden_states, tuple) else hidden_states
            triality_loss = triality_consistency_loss(last_hidden, inputs['labels'])

            # Annealing weight for triality loss (gradually increase geometric constraints)
            annealing_weight = min(0.1, current_step / self.annealing_warmup_steps * 0.1)

            # Combine losses
            total_loss = lm_loss + 0.01 * ortho_loss + annealing_weight * triality_loss

            # Log alpha gate values
            alpha_values = []
            if hasattr(model, 'alpha_gates'):
                for alpha_gate in model.alpha_gates:
                    alpha_values.append(torch.sigmoid(alpha_gate).item())

                if alpha_values:
                    avg_alpha = sum(alpha_values) / len(alpha_values)
                    # Log to trainer's log history
                    self.log({"alpha_gate_avg": avg_alpha, "annealing_weight": annealing_weight})
        else:
            total_loss = lm_loss

        return (total_loss, outputs) if return_outputs else total_loss

def create_so8t_qlora_model(base_model_path: str, so8t_hidden_size: int = 2048, place_so8t_in_all_intermediate: bool = True, num_so8t_layers: int = 4):
    """Create SO8T/thinking QLoRA model based on Borea-Phi3.5-instinct-jp."""
    logger.info("Creating SO8T/thinking QLoRA model...")

    # Create SO8T-enhanced model
    if place_so8t_in_all_intermediate:
        target_layers = None  # All intermediate layers
    else:
        # Use specific target layers (4-11)
        target_layers = list(range(3, 11))  # 0-indexed: 3-10 (4-11 layers)

    model = SO8TPhi3Model(
        base_model_path=base_model_path,
        target_layers=target_layers,
        so8t_hidden_size=2048  # Fixed SO(8) hidden size
    )

    # Setup QLoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # LoRA rank
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        inference_mode=False,
    )

    # Apply QLoRA to base model
    model.base_model = get_peft_model(model.base_model, lora_config)
    model.base_model = prepare_model_for_kbit_training(model.base_model)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    logger.info(f"SO8T QLoRA model created")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(".2f")

    return model

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train SO8T/thinking QLoRA Model")
    parser.add_argument("--base-model", type=str, default="models/Borea-Phi-3.5-mini-Instruct-Jp", help="Base model path")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--train-data", type=str, required=True, help="Training data path")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, default=1, help="Per device batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=50, help="Warmup steps")
    parser.add_argument("--save-steps", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--logging-steps", type=int, default=25, help="Log every N steps")
    parser.add_argument("--so8t-hidden-size", type=int, default=2048, help="SO8T hidden size")
    parser.add_argument("--place-so8t-in-all-intermediate", action="store_true", default=True, help="Place SO8T layers in all intermediate layers (default: True)")
    parser.add_argument("--num-so8t-layers", type=int, default=4, help="Number of SO8T reasoning layers (ignored if --place-so8t-in-all-intermediate is True)")
    parser.add_argument("--annealing-warmup", type=int, default=100, help="Annealing warmup steps for Alpha Gate (Golden Ratio activation)")
    parser.add_argument("--enable-mass-gap-monitor", action="store_true", default=True, help="Enable Mass Gap Monitor for phase transition detection")
    parser.add_argument("--monitor-interval", type=int, default=25, help="Mass Gap Monitor check interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    logger.info("=" * 80)
    logger.info("[START] Training SO8T/thinking QLoRA Model")
    logger.info("Borea-Phi3.5-instinct-jp + SO(8) Geometric Reasoning")
    logger.info("=" * 80)
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"SO8T placement: {'All intermediate layers' if args.place_so8t_in_all_intermediate else f'{args.num_so8t_layers} layers'}")
    logger.info(f"SO8T hidden size: {args.so8t_hidden_size}")
    logger.info(f"Alpha Gate annealing warmup: {args.annealing_warmup} steps (Golden Ratio activation to phi=1.618)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create SO8T QLoRA model
    model = create_so8t_qlora_model(
        base_model_path=args.base_model,
        place_so8t_in_all_intermediate=args.place_so8t_in_all_intermediate
    )

    # [BRAIN] Initialize Mass Gap Monitor for phase transition detection
    mass_gap_monitor = None
    mgm_callback = None

    if args.enable_mass_gap_monitor:
        logger.info("[BRAIN] Initializing Mass Gap Monitor for SO8T/thinking phase transition detection...")
        mass_gap_monitor = MassGapMonitor(
            log_interval=args.monitor_interval,
            model_name="so8t_thinking_qlora"
        )
        mass_gap_monitor.start_monitoring()
        mgm_callback = SO8TMassGapCallback(mass_gap_monitor)
        logger.info("[TARGET] Mass Gap Monitor active - watching for geometric awakening!")

    # Create dataset
    train_dataset = SO8TDataset(args.train_data, tokenizer)

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
        run_name=f"so8t_qlora_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Progress callback
    progress_callback = SO8TProgressCallback()

    # SO8T monitoring callback for hidden layers, orthogonality, annealing, and logits
    so8t_monitor = SO8TMonitorCallback(
        log_every_n_steps=args.logging_steps,
        annealing_warmup_steps=args.annealing_warmup,
        model=model
    )

    # Add Mass Gap Monitor callback if enabled
    callbacks = [progress_callback, so8t_monitor]
    if mgm_callback:
        callbacks.append(mgm_callback)

    # Create trainer with annealing warmup and monitoring
    trainer = SO8TTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
        annealing_warmup_steps=args.annealing_warmup,
    )

    logger.info("Starting SO8T/thinking QLoRA training...")
    trainer.train()

    # Stop Mass Gap Monitor if active
    if mass_gap_monitor:
        logger.info("[STOP] Stopping Mass Gap Monitor...")
        mass_gap_monitor.stop_monitoring()

    # Save final model
    final_model_path = output_dir / "final_model"
    final_model_path.mkdir(exist_ok=True)

    # Save LoRA adapters
    trainer.model.base_model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    # Save SO8T layers
    trainer.model.save_pretrained(final_model_path)

    logger.info("SO8T/thinking QLoRA training completed!")
    logger.info(f"Model saved to {final_model_path}")
    logger.info("To use for inference:")
    logger.info(f"  from peft import PeftModel")
    logger.info(f"  model = PeftModel.from_pretrained('{args.base_model}', '{final_model_path}')")

    # Play notification
    try:
        import winsound
        winsound.Beep(1000, 1000)
    except:
        pass

if __name__ == "__main__":
    main()
