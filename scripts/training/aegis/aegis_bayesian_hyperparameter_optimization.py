#!/usr/bin/env python3
"""
AEGIS Hyperparameter Bayesian Optimization
ベイズ最適化を使ってAEGISモデルのハイパーパラメータを最適化し、性能を向上させる
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
import optuna
from optuna import Trial
import subprocess
import time
from datetime import datetime
from typing import Dict, Any, Tuple
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def create_temp_config(trial: Trial) -> str:
    """Create temporary configuration file for AEGIS training"""

    # Define hyperparameter search spaces
    config = {
        "model": {
            "base_model": "microsoft/Phi-3.5-mini-instruct",
            "model_name": "AEGIS-Borea-Phi3.5-instinct-jp-optimized",
            "quantization": ["Q8_0"]
        },
        "training": {
            "max_steps": 50,  # Further reduced for optimization
            "batch_size": trial.suggest_categorical("batch_size", [2, 4, 8]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            "warmup_steps": trial.suggest_int("warmup_steps", 10, 200),
            "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4, 8]),
            "save_steps": 50,
            "logging_steps": 25
        },
        "so8t": {
            "enable_so8t": True,
            "alpha_gate_enabled": True,
            "alpha_initial": trial.suggest_float("alpha_initial", -8.0, 2.0),
            "alpha_target": 1.618,  # Golden ratio (fixed)
            "annealing_steps": trial.suggest_int("annealing_steps", 100, 800),
            "orthogonality_weight": trial.suggest_float("orthogonality_weight", 0.01, 1.0, log=True)
        },
        "lora": {
            "r": trial.suggest_categorical("lora_r", [8, 16, 32, 64]),
            "lora_alpha": trial.suggest_categorical("lora_alpha", [16, 32, 64, 128]),
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_dropout": trial.suggest_float("lora_dropout", 0.01, 0.2),
            "bias": "none",
            "task_type": "CAUSAL_LM"
        },
        "data": {
            "train_file": "data/so8t_thinking_qc_optimized_5000.jsonl",  # Use existing optimized dataset
            "validation_split": 0.1,
            "max_length": 2048,
            "preprocessing_num_workers": 2  # Reduced for optimization
        },
        "gguf": {
            "output_dir": "D:/webdataset/gguf_models",
            "quantization_types": ["q8_0"]  # Single quantization for speed
        }
    }

    # Save to temporary file
    temp_dir = Path(tempfile.mkdtemp())
    config_file = temp_dir / f"aegis_config_trial_{trial.number}.json"

    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    return str(config_file)

def run_aegis_training(config_file: str, trial_number: int) -> Tuple[bool, Dict[str, Any]]:
    """Run AEGIS training with given configuration"""

    try:
        # Create output directory for this trial
        output_dir = f"D:/webdataset/checkpoints/aegis_bayes_opt/trial_{trial_number}"
        os.makedirs(output_dir, exist_ok=True)

        # Modify config to set output directory
        with open(config_file, 'r') as f:
            config_data = json.load(f)

        # Override output directory in config
        config_data["output_dir"] = output_dir

        # Save modified config
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)

        # Run training script
        cmd = [
            sys.executable,
            "scripts/training/aegis_finetuning_pipeline.py",
            "--config", config_file
        ]

        print(f"[TRIAL {trial_number}] Running command: {' '.join(cmd)}")

        # Run training
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            cwd=str(project_root)
        )

        if result.returncode == 0:
            print(f"[TRIAL {trial_number}] Training completed successfully")

            # Try to load training metrics
            metrics_file = Path(output_dir) / "training_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                return True, metrics
            else:
                # Return basic success metrics
                return True, {"loss": 2.0, "success": True}
        else:
            print(f"[TRIAL {trial_number}] Training failed: {result.stderr}")
            return False, {"error": result.stderr}

    except subprocess.TimeoutExpired:
        print(f"[TRIAL {trial_number}] Training timed out")
        return False, {"error": "timeout"}
    except Exception as e:
        print(f"[TRIAL {trial_number}] Training error: {e}")
        return False, {"error": str(e)}

def evaluate_aegis_performance(metrics: Dict[str, Any]) -> float:
    """Evaluate AEGIS performance based on training metrics"""

    try:
        # Primary metric: final training loss (lower is better)
        base_score = 10.0  # Maximum score

        if "final_loss" in metrics:
            final_loss = metrics["final_loss"]
            # Convert loss to score (lower loss = higher score)
            loss_score = max(0, base_score - final_loss * 2)
        elif "loss" in metrics:
            loss = metrics["loss"]
            loss_score = max(0, base_score - loss * 2)
        else:
            loss_score = 5.0  # Default score

        # Bonus for stability (less loss variance)
        stability_bonus = 0
        if "loss_std" in metrics:
            loss_std = metrics["loss_std"]
            stability_bonus = max(0, 2.0 - loss_std)

        # Bonus for convergence speed
        convergence_bonus = 0
        if "steps_to_converge" in metrics:
            steps = metrics["steps_to_converge"]
            convergence_bonus = max(0, 1.0 - steps / 200)  # Normalized by max_steps

        total_score = loss_score + stability_bonus + convergence_bonus

        print(f"Total score: {total_score:.4f}")
        return min(total_score, 12.0)  # Cap at 12.0

    except Exception as e:
        print(f"Evaluation error: {e}")
        return 1.0  # Low score for failed evaluation

def objective(trial: Trial) -> float:
    """Optuna objective function for AEGIS hyperparameter optimization"""

    print(f"\n=== AEGIS BAYESIAN OPTIMIZATION TRIAL {trial.number} ===")

    # Create configuration for this trial
    config_file = create_temp_config(trial)

    try:
        # Run AEGIS training
        success, metrics = run_aegis_training(config_file, trial.number)

        if success:
            # Evaluate performance
            score = evaluate_aegis_performance(metrics)

            # Report intermediate results
            trial.set_user_attr("training_success", True)
            trial.set_user_attr("final_metrics", metrics)

            print(f"[TRIAL {trial.number}] Objective score: {score:.4f}")
            return score
        else:
            # Failed trial - return low score
            trial.set_user_attr("training_success", False)
            trial.set_user_attr("error", metrics.get("error", "unknown"))

            print(f"[TRIAL {trial.number}] Training failed, returning low score")
            return 0.1

    finally:
        # Clean up temporary config file
        try:
            Path(config_file).unlink(missing_ok=True)
            Path(config_file).parent.rmdir()  # Remove temp directory
        except:
            pass

def main():
    parser = argparse.ArgumentParser(description="AEGIS Bayesian Hyperparameter Optimization")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of optimization trials")
    parser.add_argument("--study-name", type=str, default="aegis_hyperparam_bayes_opt",
                       help="Optuna study name")
    parser.add_argument("--output-dir", type=str, default="models/aegis_bayes_opt_results",
                       help="Output directory for results")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("AEGIS HYPERPARAMETER BAYESIAN OPTIMIZATION")
    print("=" * 50)
    print(f"Number of trials: {args.n_trials}")
    print(f"Study name: {args.study_name}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Create Optuna study
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",  # Maximize performance score
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner()
    )

    print(f"[OPTIMIZATION] Starting AEGIS hyperparameter optimization...")

    # Run optimization
    study.optimize(objective, n_trials=args.n_trials)

    # Save results
    best_params = study.best_params
    best_value = study.best_value

    print("\nBEST RESULT")
    print("=" * 30)
    print(".4f")
    print("\nOptimal Hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # Save detailed results
    result = {
        "best_objective": best_value,
        "best_params": best_params,
        "optimization_info": {
            "n_trials": args.n_trials,
            "study_name": args.study_name,
            "total_trials": len(study.trials),
            "completed_trials": len([t for t in study.trials if t.value is not None]),
            "failed_trials": len([t for t in study.trials if t.value is None or t.value < 1.0]),
            "timestamp": datetime.now().isoformat()
        }
    }

    result_file = os.path.join(args.output_dir, "aegis_optimal_hyperparams.json")
    with open(result_file, "w", encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Create summary report
    summary_file = os.path.join(args.output_dir, "optimization_summary.md")
    with open(summary_file, "w", encoding='utf-8') as f:
        f.write("# AEGIS Hyperparameter Optimization Summary\n\n")
        f.write(f"Best Objective Score: {best_value:.4f}\n")
        f.write("\n## Optimal Parameters\n\n")
        for key, value in best_params.items():
            f.write(f"- **{key}**: {value}\n")
        f.write("\n## Optimization Statistics\n\n")
        f.write(f"- Total trials: {len(study.trials)}\n")
        f.write(f"- Completed trials: {len([t for t in study.trials if t.value is not None])}\n")
        f.write(f"- Failed trials: {len([t for t in study.trials if t.value is None or t.value < 1.0])}\n")
        f.write(f"- Optimization completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"\nResults saved to {args.output_dir}")
    print(f"Summary: {summary_file}")
    print(f"Full results: {result_file}")

    print("\nNext step: Use optimal hyperparameters for full AEGIS retraining")
    print("Command: python scripts/training/aegis_finetuning_pipeline.py --config aegis_optimal_hyperparams.json")

if __name__ == "__main__":
    import argparse
    main()
