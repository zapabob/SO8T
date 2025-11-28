#!/usr/bin/env python3
"""
アルファゲート値ベイズ最適化スクリプト

アルファゲートの初期値をベイズ最適化で決定し、
アニーリング時にはその最適化された値を固定する
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from datasets import load_dataset
import os
import sys
import argparse
import json
from pathlib import Path
import optuna
from optuna import Trial
import numpy as np
from tqdm import tqdm
import time
import math

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def sigmoid(x):
    """Sigmoid function for alpha gate transformation"""
    return 1 / (1 + math.exp(-x))

def create_so8t_lora_config(r=64, lora_alpha=128, lora_dropout=0.05):
    """Create LoRA configuration for SO8T/thinking integration"""
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=[
            "self_attn.qkv_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

def load_base_model_with_adapter(model_path="microsoft/Phi-3.5-mini-instruct"):
    """Load base model with SO8T adapter"""
    print(f"[MODEL] Loading {model_path}...")

    try:
        # Try with 4-bit quantization first
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"
        )
    except Exception as e:
        print(f"[MODEL] 4-bit quantization failed: {e}")
        print("[MODEL] Falling back to 8-bit quantization...")

        # Fallback to 8-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"
        )

    model = prepare_model_for_kbit_training(model)
    return model

def calculate_so8t_loss(outputs, labels, alpha_gate_value):
    """Calculate SO8T-aware loss with alpha gate penalty"""
    logits = outputs.logits

    # Standard LM loss
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss()
    lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # Alpha gate stability penalty (penalize extreme values)
    # Convert raw alpha to sigmoid for penalty calculation
    alpha_sigmoid = sigmoid(alpha_gate_value)
    alpha_penalty = 0.01 * abs(alpha_sigmoid - 1.618)  # Golden ratio target (use abs for float)

    # Orthogonality penalty for SO(8) structure (if available)
    ortho_loss = torch.tensor(0.0, device=logits.device)
    if hasattr(outputs, 'rotation_loss'):
        ortho_loss = outputs.rotation_loss

    total_loss = lm_loss + 0.1 * ortho_loss + alpha_penalty
    return total_loss, lm_loss.item(), ortho_loss.item() if torch.is_tensor(ortho_loss) else ortho_loss

def objective(trial: Trial):
    """Optuna objective function for alpha gate optimization"""

    # Suggest alpha gate initial value
    # Alpha gate ranges from -10 (closed gate) to +10 (fully open gate)
    # Golden ratio target is ~1.618, so we search around that range
    alpha_init = trial.suggest_float('alpha_init', -8.0, 8.0)

    # Fixed hyperparameters for this optimization
    learning_rate = 2e-5
    batch_size = 1  # Reduced batch size to save memory
    max_steps = 10  # Further reduced steps for optimization

    print(f"[OPTUNA] Trial {trial.number}: Alpha Init={alpha_init:.4f}")
    try:
        # Load model and tokenizer
        model = load_base_model_with_adapter()
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Apply LoRA
        lora_config = create_so8t_lora_config()
        model = get_peft_model(model, lora_config)

        # Set fixed alpha gate value (no annealing)
        device = next(model.parameters()).device
        alpha_gate_tensor = torch.tensor(alpha_init, device=device, requires_grad=False)
        model.alpha_gate = alpha_gate_tensor

        # Create simple dummy dataset for optimization
        # Use a small sample text for testing
        sample_texts = [
            "これはテスト用のサンプルテキストです。アルファゲートの最適化をテストしています。",
            "機械学習モデルにおいて、ハイパーパラメータの最適化は重要です。",
            "SO8Tモデルは回転ゲートを使用して表現力を向上させます。",
            "ベイズ最適化は効率的にハイパーパラメータを探索できます。",
            "アルファゲートはモデルの安定性を制御する重要なパラメータです。"
        ]

        # Tokenize and prepare data
        tokenized_data = []
        for text in sample_texts:
            tokens = tokenizer(text, truncation=True, max_length=512, padding=False, return_tensors="pt")
            tokens["labels"] = tokens["input_ids"].clone()
            tokenized_data.append(tokens)

        # Create simple dataloader
        train_dataloader = tokenized_data

        # Optimizer
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

        # Training loop with FIXED alpha gate
        model.train()
        device = next(model.parameters()).device

        losses = []
        alpha_sigmoid_values = []

        for step in range(max_steps):
            # Cycle through dummy data
            batch_idx = step % len(train_dataloader)
            batch = train_dataloader[batch_idx]

            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Alpha gate remains FIXED throughout training
            current_alpha_sigmoid = sigmoid(alpha_init)
            alpha_sigmoid_values.append(current_alpha_sigmoid)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(**batch)

            # Calculate loss with fixed alpha gate
            loss, lm_loss, ortho_loss = calculate_so8t_loss(outputs, batch["labels"], alpha_init)

            if not torch.isnan(loss) and torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())
            else:
                losses.append(100.0)  # Penalize NaN losses

        # Calculate objective value
        if losses:
            # Use average loss of last 10 steps for stability
            final_loss = np.mean(losses[-10:]) if len(losses) >= 10 else np.mean(losses)
            alpha_stability = np.std(alpha_sigmoid_values)  # Should be 0 since alpha is fixed
            alpha_final_sigmoid = alpha_sigmoid_values[-1]

            # Objective: minimize loss + small penalty for deviation from golden ratio
            objective_value = final_loss + 0.001 * abs(alpha_final_sigmoid - 1.618)
        else:
            objective_value = 100.0

        print(f"[OPTUNA] Trial {trial.number} completed - Objective: {objective_value:.4f}")
        return objective_value

    except Exception as e:
        print(f"[OPTUNA] Trial {trial.number} failed: {e}")
        return 100.0  # Return high penalty for failed trials

def main():
    parser = argparse.ArgumentParser(description="Alpha Gate Bayesian Optimization")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--study-name", type=str, default="alpha_gate_bayes_opt", help="Optuna study name")
    parser.add_argument("--output-dir", type=str, default="models/alpha_gate_bayes_opt_results", help="Output directory")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create Optuna study
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner()
    )

    print(f"[OPTIMIZATION] Starting alpha gate optimization with {args.n_trials} trials...")

    # Run optimization
    study.optimize(objective, n_trials=args.n_trials)

    # Save results
    best_params = study.best_params
    best_value = study.best_value

    print("\n[BEST RESULT]")
    print(f"Objective Value: {best_value:.4f}")
    print(f"Optimal Alpha Initial Value: {best_params['alpha_init']:.4f}")
    print(f"Optimal Alpha Sigmoid Value: {sigmoid(best_params['alpha_init']):.4f}")
    print(f"Golden Ratio Target: {1.618:.4f}")

    # Save to JSON
    result = {
        "best_objective": best_value,
        "best_params": best_params,
        "optimal_alpha_init": best_params['alpha_init'],
        "optimal_alpha_sigmoid": sigmoid(best_params['alpha_init']),
        "golden_ratio_target": 1.618,
        "n_trials": args.n_trials,
        "study_name": args.study_name,
        "timestamp": time.time()
    }

    with open(os.path.join(args.output_dir, "optimal_alpha_gate.json"), "w", encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Plot optimization history (if plotly available and trials succeeded)
    try:
        import plotly
        # Only plot if we have at least one successful trial
        successful_trials = [t for t in study.trials if t.value < 100.0]
        if successful_trials:
            fig = optuna.visualization.plot_optimization_history(study)
            fig.write_html(os.path.join(args.output_dir, "optimization_history.html"))

            fig = optuna.visualization.plot_param_importances(study)
            fig.write_html(os.path.join(args.output_dir, "param_importances.html"))
        else:
            print("[INFO] No successful trials found, skipping visualization")
    except ImportError:
        print("[INFO] Plotly not available, skipping visualization")
    except Exception as e:
        print(f"[INFO] Visualization failed: {e}, skipping visualization")

    print(f"[COMPLETE] Results saved to {args.output_dir}")
    print("[RECOMMENDATION] Use optimal_alpha_init in your training script for fixed alpha gate during annealing")

if __name__ == "__main__":
    main()
