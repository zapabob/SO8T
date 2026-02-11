#!/usr/bin/env python3
"""
Alpha Gateベイズ最適化スクリプト（直交誤差=0、α=Φ^(-2)=0.432を目標）

目的:
1. Alpha Gateのシグモイドアニーリングでα=0.432（Φ^(-2)）を目標
2. 直交誤差を0に保つためのハイパーパラメータ最適化
3. PET正則化による学習発散防止の最適化

最適化パラメータ:
- alpha_gate_orthogonal_weight: 直交誤差の重み（0.0-10.0）
- alpha_gate_pet_weight: PET正則化の重み（0.0-1.0）
- alpha_gate_steepness: シグモイドアニーリングの急激さ（5.0-20.0）
- alpha_gate_annealing_steps: アニーリングステップ数（500-2000）

評価指標:
- 直交誤差: 0に近いほど良い
- Alpha Gate値: 0.432に近いほど良い
- 学習損失: 低いほど良い
- 学習発散: 発生しないことが重要
"""

import torch
import torch.nn as nn
import numpy as np
import optuna
from optuna import Trial
import json
import math
from pathlib import Path
import sys
import logging
from typing import Dict, Tuple, Optional
import time

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from so8t.core.safety_aware_so8t import SafetyAwareSO8TModel, SafetyAwareSO8TConfig
from so8t.core.strict_so8_rotation_gate import StrictSO8RotationGate

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ターゲット値
ALPHA_GATE_TARGET = 0.432  # Φ^(-2) = (1/1.618)^2 ≈ 0.382, ユーザー指定: 0.432
ORTHOGONAL_ERROR_TARGET = 0.0  # 直交誤差は0に保つ

def create_dummy_model(config: SafetyAwareSO8TConfig) -> SafetyAwareSO8TModel:
    """ダミーモデルを作成（最適化用）"""
    # 軽量なモデルを使用（最適化のため）
    base_model_name = "microsoft/Phi-3.5-mini-instruct"
    
    try:
        model = SafetyAwareSO8TModel(
            base_model_name_or_path=base_model_name,
            so8t_config=config,
            quantization_config=None  # 最適化時は量子化なし
        )
        return model
    except Exception as e:
        logger.error(f"[ERROR] Failed to create model: {e}")
        raise

def simulate_training_step(
    model: SafetyAwareSO8TModel,
    config: SafetyAwareSO8TConfig,
    num_steps: int = 100
) -> Dict[str, float]:
    """
    学習ステップをシミュレーションして評価指標を計算
    
    Args:
        model: SO8Tモデル
        config: SO8T設定
        num_steps: シミュレーションステップ数
    
    Returns:
        評価指標の辞書
    """
    device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
    
    # ダミー入力データ
    batch_size = 2
    seq_len = 32
    vocab_size = model.base_model.config.vocab_size
    
    # 評価指標を記録
    orthogonal_errors = []
    alpha_gate_values = []
    losses = []
    pet_losses = []
    
    model.eval()  # 評価モード（勾配計算は行うが、ドロップアウト等は無効化）
    
    for step in range(num_steps):
        # ダミー入力データを生成
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)
        labels = input_ids.clone()
        
        # Forward pass
        with torch.set_grad_enabled(True):  # 勾配計算を有効化
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True
            )
        
        # 評価指標を記録
        if outputs.get("so8_orth_loss") is not None:
            orthogonal_errors.append(outputs["so8_orth_loss"].item())
        
        if outputs.get("alpha_gate_value") is not None:
            alpha_gate_values.append(outputs["alpha_gate_value"])
        
        if outputs.get("loss") is not None:
            losses.append(outputs["loss"].item())
        
        if outputs.get("pet_loss") is not None:
            pet_losses.append(outputs["pet_loss"].item())
    
    # 評価指標を計算
    avg_orthogonal_error = np.mean(orthogonal_errors) if orthogonal_errors else 1.0
    final_alpha_gate = alpha_gate_values[-1] if alpha_gate_values else 0.0
    avg_loss = np.mean(losses) if losses else 100.0
    avg_pet_loss = np.mean(pet_losses) if pet_losses else 0.0
    
    # 学習発散の検出（損失がNaNまたはInfになった場合）
    divergence_detected = any(not np.isfinite(loss) for loss in losses)
    
    return {
        "orthogonal_error": avg_orthogonal_error,
        "alpha_gate_value": final_alpha_gate,
        "loss": avg_loss,
        "pet_loss": avg_pet_loss,
        "divergence": divergence_detected
    }

def objective(trial: Trial) -> float:
    """
    Optuna目的関数
    
    最適化目標:
    1. 直交誤差を0に保つ（最小化）
    2. Alpha Gate値を0.432に近づける（最小化）
    3. 学習損失を最小化
    4. 学習発散を防止（ペナルティ）
    """
    # ハイパーパラメータを提案
    alpha_gate_orthogonal_weight = trial.suggest_float(
        "alpha_gate_orthogonal_weight", 
        0.0, 10.0, 
        log=True  # 対数スケールで探索
    )
    
    alpha_gate_pet_weight = trial.suggest_float(
        "alpha_gate_pet_weight", 
        0.0, 1.0
    )
    
    alpha_gate_steepness = trial.suggest_float(
        "alpha_gate_steepness", 
        5.0, 20.0
    )
    
    alpha_gate_annealing_steps = trial.suggest_int(
        "alpha_gate_annealing_steps", 
        500, 2000, 
        step=100
    )
    
    logger.info(
        f"[TRIAL {trial.number}] "
        f"orthogonal_weight={alpha_gate_orthogonal_weight:.4f}, "
        f"pet_weight={alpha_gate_pet_weight:.4f}, "
        f"steepness={alpha_gate_steepness:.2f}, "
        f"annealing_steps={alpha_gate_annealing_steps}"
    )
    
    try:
        # SO8T設定を作成
        config = SafetyAwareSO8TConfig(
            use_alpha_gate=True,
            alpha_gate_target=ALPHA_GATE_TARGET,
            alpha_gate_start=-5.0,
            alpha_gate_annealing_steps=alpha_gate_annealing_steps,
            alpha_gate_steepness=alpha_gate_steepness,
            alpha_gate_orthogonal_weight=alpha_gate_orthogonal_weight,
            alpha_gate_pet_weight=alpha_gate_pet_weight,
            use_strict_so8_rotation=True,
            so8_apply_to_intermediate_layers=True,
            pet_apply_to_intermediate_layers=True
        )
        
        # モデルを作成
        model = create_dummy_model(config)
        
        # 学習シミュレーション
        metrics = simulate_training_step(model, config, num_steps=50)  # 最適化時は少ないステップ数
        
        # 目的関数値を計算
        # 1. 直交誤差を0に保つ（重み付き）
        orthogonal_penalty = metrics["orthogonal_error"] * alpha_gate_orthogonal_weight
        
        # 2. Alpha Gate値を0.432に近づける
        alpha_gate_penalty = abs(metrics["alpha_gate_value"] - ALPHA_GATE_TARGET) ** 2
        
        # 3. 学習損失
        loss_penalty = metrics["loss"]
        
        # 4. 学習発散ペナルティ（大きい値）
        divergence_penalty = 1000.0 if metrics["divergence"] else 0.0
        
        # 5. PET損失（学習発散防止）
        pet_penalty = metrics["pet_loss"] * alpha_gate_pet_weight
        
        # 総合目的関数値（最小化）
        objective_value = (
            orthogonal_penalty +
            alpha_gate_penalty * 10.0 +  # Alpha Gateは重要なので重みを大きく
            loss_penalty * 0.1 +  # 損失は重要だが、他の指標とのバランスを取る
            divergence_penalty +
            pet_penalty
        )
        
        logger.info(
            f"[TRIAL {trial.number}] "
            f"orthogonal_error={metrics['orthogonal_error']:.6f}, "
            f"alpha_gate={metrics['alpha_gate_value']:.4f}, "
            f"loss={metrics['loss']:.4f}, "
            f"divergence={metrics['divergence']}, "
            f"objective={objective_value:.4f}"
        )
        
        # 中間値を記録（Optunaの履歴に保存）
        trial.set_user_attr("orthogonal_error", metrics["orthogonal_error"])
        trial.set_user_attr("alpha_gate_value", metrics["alpha_gate_value"])
        trial.set_user_attr("loss", metrics["loss"])
        trial.set_user_attr("divergence", metrics["divergence"])
        
        return objective_value
        
    except Exception as e:
        logger.error(f"[TRIAL {trial.number}] Failed: {e}")
        import traceback
        traceback.print_exc()
        return 10000.0  # 失敗時は大きなペナルティ

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Alpha Gate Bayesian Optimization (Orthogonal Error = 0, α = 0.432)")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of optimization trials")
    parser.add_argument("--study-name", type=str, default="alpha_gate_orthogonal_optimization", help="Optuna study name")
    parser.add_argument("--output-dir", type=str, default="D:/webdataset/alpha_gate_bayes_opt", help="Output directory")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL (optional)")
    
    args = parser.parse_args()
    
    # 出力ディレクトリを作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Optuna studyを作成
    study_kwargs = {
        "study_name": args.study_name,
        "direction": "minimize",
        "sampler": optuna.samplers.TPESampler(seed=42),
        "pruner": optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    }
    
    if args.storage:
        study_kwargs["storage"] = args.storage
        study = optuna.create_study(**study_kwargs, load_if_exists=True)
    else:
        study = optuna.create_study(**study_kwargs)
    
    logger.info(f"[OPTIMIZATION] Starting optimization with {args.n_trials} trials...")
    logger.info(f"[OPTIMIZATION] Target: α={ALPHA_GATE_TARGET}, Orthogonal Error={ORTHOGONAL_ERROR_TARGET}")
    
    # 最適化を実行
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    
    # 結果を保存
    best_params = study.best_params
    best_value = study.best_value
    best_trial = study.best_trial
    
    logger.info("\n" + "="*80)
    logger.info("[BEST RESULT]")
    logger.info("="*80)
    logger.info(f"Objective Value: {best_value:.6f}")
    logger.info(f"Best Parameters:")
    for key, value in best_params.items():
        logger.info(f"  {key}: {value:.6f}")
    
    logger.info(f"\nBest Trial Metrics:")
    logger.info(f"  Orthogonal Error: {best_trial.user_attrs.get('orthogonal_error', 'N/A'):.6f}")
    logger.info(f"  Alpha Gate Value: {best_trial.user_attrs.get('alpha_gate_value', 'N/A'):.4f}")
    logger.info(f"  Loss: {best_trial.user_attrs.get('loss', 'N/A'):.4f}")
    logger.info(f"  Divergence: {best_trial.user_attrs.get('divergence', 'N/A')}")
    
    # JSON形式で保存
    result = {
        "best_objective": best_value,
        "best_params": best_params,
        "best_trial_metrics": {
            "orthogonal_error": best_trial.user_attrs.get("orthogonal_error"),
            "alpha_gate_value": best_trial.user_attrs.get("alpha_gate_value"),
            "loss": best_trial.user_attrs.get("loss"),
            "divergence": best_trial.user_attrs.get("divergence")
        },
        "targets": {
            "alpha_gate_target": ALPHA_GATE_TARGET,
            "orthogonal_error_target": ORTHOGONAL_ERROR_TARGET
        },
        "n_trials": args.n_trials,
        "study_name": args.study_name,
        "timestamp": time.time()
    }
    
    result_file = output_dir / "optimal_alpha_gate_orthogonal.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n[COMPLETE] Results saved to {result_file}")
    
    # 可視化（オプション）
    try:
        import plotly
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(str(output_dir / "optimization_history.html"))
        
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(str(output_dir / "param_importances.html"))
        
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_html(str(output_dir / "parallel_coordinate.html"))
        
        logger.info(f"[VISUALIZATION] Plots saved to {output_dir}")
    except ImportError:
        logger.warning("[VISUALIZATION] Plotly not available, skipping visualization")
    except Exception as e:
        logger.warning(f"[VISUALIZATION] Failed to create plots: {e}")

if __name__ == "__main__":
    main()

