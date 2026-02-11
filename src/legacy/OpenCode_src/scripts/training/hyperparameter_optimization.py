#!/usr/bin/env python3
"""
SO8T/thinking ハイパーパラメータ最適化スクリプト

Alpha Gate + Loss最小化を目的関数とした数値最適化を実行
- Optunaを使用したベイズ最適化
- Alphaアニーリングスケジュールの最適化
- 学習率、バッチサイズ、ウォームアップステップの最適化
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

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

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

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
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

def calculate_so8t_loss(outputs, labels, alpha_gate):
    """Calculate SO8T-aware loss with orthogonality penalty"""
    logits = outputs.logits

    # Standard LM loss
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss()
    lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # Orthogonality penalty for SO(8) structure
    ortho_loss = torch.tensor(0.0, device=logits.device)
    if hasattr(outputs, 'rotation_loss'):
        ortho_loss = outputs.rotation_loss

    # Alpha Gate stability penalty (penalize extreme values)
    alpha_penalty = 0.01 * torch.abs(alpha_gate - 1.618)  # Golden ratio target

    total_loss = lm_loss + 0.1 * ortho_loss + alpha_penalty
    return total_loss, lm_loss.item(), ortho_loss.item() if torch.is_tensor(ortho_loss) else ortho_loss

def objective(trial: Trial):
    """Optuna objective function for hyperparameter optimization"""

    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [1, 2, 4])
    warmup_steps = trial.suggest_int('warmup_steps', 2, 10)
    alpha_warmup_steps = trial.suggest_int('alpha_warmup_steps', 5, 20)
    max_steps = 20  # Fixed for optimization

    print(f"[OPTUNA] Trial {trial.number}: LR={learning_rate:.2e}, BS={batch_size}, Warmup={warmup_steps}, AlphaWarmup={alpha_warmup_steps}")

    try:
        # Load model and tokenizer
        model = load_base_model_with_adapter()
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Apply LoRA
        lora_config = create_so8t_lora_config()
        model = get_peft_model(model, lora_config)

        # Initialize Alpha Gate
        device = next(model.parameters()).device
        model.alpha_gate = nn.Parameter(torch.tensor(-5.0, device=device))

        # Load dataset
        dataset = load_dataset("TFMC/imatrix-dataset-for-japanese-llm", split="train", streaming=True)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        # Create dataloader
        def tokenize_function(examples):
            return tokenizer(examples['text'], truncation=True, max_length=512, padding=False)

        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
        train_dataloader = DataLoader(
            tokenized_dataset,
            batch_size=batch_size,
            collate_fn=data_collator,
            shuffle=False
        )

        # Optimizer and scheduler
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return max(0.1, 0.5 * (1 + torch.cos(torch.tensor(torch.pi * progress))))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Alpha annealing function
        def alpha_lambda(step, target_alpha=1.618):
            if step < alpha_warmup_steps:
                progress = step / alpha_warmup_steps
                return -5.0 + progress * (target_alpha + 5.0)
            else:
                return target_alpha

        # Training loop
        model.train()
        device = next(model.parameters()).device
        data_iter = iter(train_dataloader)

        losses = []
        alpha_values = []

        for step in range(max_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_dataloader)
                batch = next(data_iter)

            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items() if v is not None}
            if not batch or "input_ids" not in batch:
                continue

            # Update Alpha Gate
            current_alpha = alpha_lambda(step)
            model.alpha_gate.data = torch.tensor(current_alpha, device=device)
            alpha_values.append(current_alpha)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(**batch)

            # Calculate loss
            loss, lm_loss, ortho_loss = calculate_so8t_loss(outputs, batch["labels"], model.alpha_gate)

            if not torch.isnan(loss) and torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                losses.append(loss.item())
            else:
                losses.append(100.0)  # Penalize NaN losses

        # Calculate objective (minimize final loss + alpha stability)
        final_loss = np.mean(losses[-5:]) if losses else 100.0  # Average of last 5 steps
        alpha_stability = np.std(alpha_values) if alpha_values else 10.0
        alpha_final = alpha_values[-1] if alpha_values else 0.0

        # Objective: minimize loss + penalty for unstable alpha
        objective_value = final_loss + 0.1 * alpha_stability + 0.01 * abs(alpha_final - 1.618)

        print(f"[OPTUNA] Trial {trial.number} completed - Objective: {objective_value:.4f}")
        return objective_value

    except Exception as e:
        print(f"[OPTUNA] Trial {trial.number} failed: {e}")
        return 100.0  # Return high penalty for failed trials

def main():
    parser = argparse.ArgumentParser(description="SO8T Hyperparameter Optimization")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of optimization trials")
    parser.add_argument("--study-name", type=str, default="so8t_hyperopt", help="Optuna study name")
    parser.add_argument("--output-dir", type=str, default="models/so8t_hyperopt_results", help="Output directory")

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

    print(f"[OPTIMIZATION] Starting hyperparameter optimization with {args.n_trials} trials...")

    # Run optimization
    study.optimize(objective, n_trials=args.n_trials)

    # Save results
    best_params = study.best_params
    best_value = study.best_value

    print("\n[BEST RESULT]")
    print(f"Objective Value: {best_value:.4f}")
    print(f"Parameters: {best_params}")

    # Save to JSON
    result = {
        "best_objective": best_value,
        "best_params": best_params,
        "n_trials": args.n_trials,
        "study_name": args.study_name,
        "timestamp": time.time()
    }

    with open(os.path.join(args.output_dir, "best_hyperparams.json"), "w") as f:
        json.dump(result, f, indent=2)

    # Plot optimization history (if plotly available)
    try:
        import plotly
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(os.path.join(args.output_dir, "optimization_history.html"))

        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(os.path.join(args.output_dir, "param_importances.html"))
    except ImportError:
        print("[INFO] Plotly not available, skipping visualization")

    print(f"[COMPLETE] Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
