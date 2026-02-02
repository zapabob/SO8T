#!/usr/bin/env python3
"""
アルファゲート黄金比収束速度最適化スクリプト

目的関数: アルファゲートが黄金比(1.618)に最速で収束するスケジュールの探索
最適化手法: グリッドサーチ + ベイズ最適化
評価指標: 収束速度 (1 / 収束ステップ数)
"""

import torch
import numpy as np
from itertools import product
import json
from pathlib import Path
import time
from typing import Dict, List, Tuple

# 黄金比
GOLDEN_RATIO = 1.618033988749895

def simulate_alpha_annealing(
    warmup_steps: int,
    total_steps: int,
    annealing_type: str = "linear",
    **params
) -> Tuple[List[float], int]:
    """
    Alpha Gateアニーリングをシミュレーション

    Args:
        warmup_steps: ウォームアップステップ数
        total_steps: 合計ステップ数
        annealing_type: アニーリングタイプ ("linear", "exponential", "sigmoid")
        **params: アニーリングパラメータ

    Returns:
        alpha_values: 各ステップのalpha値リスト
        convergence_step: 黄金比に収束したステップ数 (-1: 未収束)
    """
    alpha_values = []
    convergence_threshold = 0.01  # |alpha - golden_ratio| < 0.01 で収束判定

    for step in range(total_steps):
        if annealing_type == "linear":
            alpha = linear_annealing(step, warmup_steps, total_steps, **params)
        elif annealing_type == "exponential":
            alpha = exponential_annealing(step, warmup_steps, total_steps, **params)
        elif annealing_type == "sigmoid":
            alpha = sigmoid_annealing(step, warmup_steps, total_steps, **params)
        else:
            raise ValueError(f"Unknown annealing type: {annealing_type}")

        alpha_values.append(alpha)

        # 収束判定
        if abs(alpha - GOLDEN_RATIO) < convergence_threshold:
            return alpha_values, step

    return alpha_values, -1  # 未収束

def linear_annealing(step: int, warmup_steps: int, total_steps: int,
                    warmup_end_ratio: float = 0.8) -> float:
    """線形アニーリング"""
    warmup_end = int(total_steps * warmup_end_ratio)

    if step < warmup_steps:
        # 初期ウォームアップ
        progress = step / warmup_steps
        return -5.0 + progress * (-2.0 - (-5.0))  # -5.0 → -2.0
    elif step < warmup_end:
        # 黄金比への収束
        progress = (step - warmup_steps) / (warmup_end - warmup_steps)
        return -2.0 + progress * (GOLDEN_RATIO - (-2.0))
    else:
        # 安定
        return GOLDEN_RATIO

def exponential_annealing(step: int, warmup_steps: int, total_steps: int,
                         decay_rate: float = 0.95) -> float:
    """指数関数アニーリング"""
    if step < warmup_steps:
        progress = step / warmup_steps
        return -5.0 + progress * (-2.0 - (-5.0))
    else:
        # 指数関数的に黄金比に近づく
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        current_alpha = -2.0 + progress * (GOLDEN_RATIO - (-2.0))
        # 指数関数で加速
        exp_factor = 1 - np.exp(-decay_rate * progress)
        return -2.0 + exp_factor * (GOLDEN_RATIO - (-2.0))

def sigmoid_annealing(step: int, warmup_steps: int, total_steps: int,
                     steepness: float = 10.0) -> float:
    """シグモイドアニーリング"""
    if step < warmup_steps:
        progress = step / warmup_steps
        return -5.0 + progress * (-2.0 - (-5.0))
    else:
        # シグモイド関数で滑らかに収束
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        sigmoid_value = 1 / (1 + np.exp(-steepness * (progress - 0.5)))
        return -2.0 + sigmoid_value * (GOLDEN_RATIO - (-2.0))

def evaluate_convergence_speed(alpha_values: List[float], convergence_step: int,
                              total_steps: int) -> float:
    """
    収束速度を評価する目的関数

    Args:
        alpha_values: Alpha値の履歴
        convergence_step: 収束ステップ数 (-1: 未収束)
        total_steps: 合計ステップ数

    Returns:
        score: 目的関数値 (高いほど良い)
    """
    if convergence_step == -1:
        # 未収束の場合はペナルティ
        final_alpha = alpha_values[-1]
        distance = abs(final_alpha - GOLDEN_RATIO)
        return 1.0 / (1.0 + distance)  # 0-1の範囲
    else:
        # 収束した場合は収束速度を評価
        convergence_ratio = convergence_step / total_steps
        # 収束が早いほど高スコア
        return 1.0 / (1.0 + convergence_ratio)

def grid_search_optimization():
    """グリッドサーチで最適なアニーリングパラメータを探す"""

    print("[GRID SEARCH] Starting Alpha Gate convergence optimization...")

    # パラメータグリッド
    param_grid = {
        'annealing_type': ['linear', 'exponential', 'sigmoid'],
        'warmup_steps': [5, 10, 15, 20],
        'warmup_end_ratio': [0.6, 0.7, 0.8, 0.9],  # linear用
        'decay_rate': [0.8, 0.9, 0.95, 0.99],     # exponential用
        'steepness': [5.0, 10.0, 15.0, 20.0]      # sigmoid用
    }

    total_steps = 100  # シミュレーションの合計ステップ数
    best_score = 0.0
    best_params = {}
    best_trajectory = []

    results = []

    # グリッドサーチ実行
    for annealing_type in param_grid['annealing_type']:
        if annealing_type == 'linear':
            param_combinations = product(
                param_grid['warmup_steps'],
                param_grid['warmup_end_ratio'],
                [0.0], [0.0]  # dummy values for exponential/sigmoid params
            )
            param_names = ['warmup_steps', 'warmup_end_ratio', 'decay_rate', 'steepness']
        elif annealing_type == 'exponential':
            param_combinations = product(
                param_grid['warmup_steps'],
                [0.0],  # dummy
                param_grid['decay_rate'],
                [0.0]   # dummy
            )
            param_names = ['warmup_steps', 'warmup_end_ratio', 'decay_rate', 'steepness']
        else:  # sigmoid
            param_combinations = product(
                param_grid['warmup_steps'],
                [0.0],  # dummy
                [0.0],  # dummy
                param_grid['steepness']
            )
            param_names = ['warmup_steps', 'warmup_end_ratio', 'decay_rate', 'steepness']

        for param_values in param_combinations:
            params = dict(zip(param_names, param_values))

            try:
                # アニーリングシミュレーション
                alpha_values, convergence_step = simulate_alpha_annealing(
                    warmup_steps=params['warmup_steps'],
                    total_steps=total_steps,
                    annealing_type=annealing_type,
                    **{k: v for k, v in params.items() if k not in ['warmup_steps']}
                )

                # 収束速度評価
                score = evaluate_convergence_speed(alpha_values, convergence_step, total_steps)

                result = {
                    'annealing_type': annealing_type,
                    'params': params,
                    'score': score,
                    'convergence_step': convergence_step,
                    'final_alpha': alpha_values[-1] if alpha_values else None
                }
                results.append(result)

                print(f"[GRID] {annealing_type} {params} -> Score: {score:.4f}")
                # ベスト更新
                if score > best_score:
                    best_score = score
                    best_params = {'annealing_type': annealing_type, **params}
                    best_trajectory = alpha_values

            except Exception as e:
                print(f"[ERROR] Failed with params {params}: {e}")
                continue

    return best_params, best_score, best_trajectory, results

def bayesian_optimization():
    """ベイズ最適化でAlpha Gateアニーリングを最適化"""
    try:
        import optuna
    except ImportError:
        print("[WARNING] Optuna not available, skipping Bayesian optimization")
        return None, 0.0, [], []

    def objective(trial):
        annealing_type = trial.suggest_categorical('annealing_type', ['linear', 'exponential', 'sigmoid'])
        warmup_steps = trial.suggest_int('warmup_steps', 5, 25)
        total_steps = 100

        params = {'warmup_steps': warmup_steps}

        if annealing_type == 'linear':
            params['warmup_end_ratio'] = trial.suggest_float('warmup_end_ratio', 0.5, 0.9)
        elif annealing_type == 'exponential':
            params['decay_rate'] = trial.suggest_float('decay_rate', 0.7, 0.99)
        else:  # sigmoid
            params['steepness'] = trial.suggest_float('steepness', 3.0, 25.0)

        # シミュレーション実行
        alpha_values, convergence_step = simulate_alpha_annealing(
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            annealing_type=annealing_type,
            **{k: v for k, v in params.items() if k != 'warmup_steps'}
        )

        score = evaluate_convergence_speed(alpha_values, convergence_step, total_steps)
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    best_score = study.best_value

    # ベストパラメータで軌跡生成
    alpha_values, convergence_step = simulate_alpha_annealing(
        warmup_steps=best_params['warmup_steps'],
        total_steps=100,
        annealing_type=best_params['annealing_type'],
        **{k: v for k, v in best_params.items() if k not in ['annealing_type', 'warmup_steps']}
    )

    return best_params, best_score, alpha_values, []

def main():
    """メイン実行関数"""
    print("[ALPHA GATE OPTIMIZER] Starting convergence speed optimization...")
    print(f"[TARGET] Golden Ratio: {GOLDEN_RATIO}")

    # グリッドサーチ実行
    print("\n[PHASE 1] Grid Search Optimization...")
    grid_best_params, grid_best_score, grid_trajectory, grid_results = grid_search_optimization()

    print("\n[GRID SEARCH BEST]")
    print(f"Score: {grid_best_score:.4f}")
    print(f"Params: {grid_best_params}")
    if grid_trajectory:
        print(f"Convergence: Step {len([a for a in grid_trajectory if abs(a - GOLDEN_RATIO) < 0.01][:1]) or 'Not converged'}")

    # ベイズ最適化実行
    print("\n[PHASE 2] Bayesian Optimization...")
    bayes_best_params, bayes_best_score, bayes_trajectory, _ = bayesian_optimization()

    if bayes_best_params:
        print("\n[BAYESIAN BEST]")
        print(f"Score: {bayes_best_score:.4f}")
        print(f"Params: {bayes_best_params}")
        if bayes_trajectory:
            convergence_step = next((i for i, a in enumerate(bayes_trajectory) if abs(a - GOLDEN_RATIO) < 0.01), -1)
            print(f"Convergence: Step {convergence_step if convergence_step != -1 else 'Not converged'}")

    # 最適なものを選択
    if bayes_best_params and bayes_best_score > grid_best_score:
        best_params = bayes_best_params
        best_score = bayes_best_score
        best_trajectory = bayes_trajectory
        method = "Bayesian"
    else:
        best_params = grid_best_params
        best_score = grid_best_score
        best_trajectory = grid_trajectory
        method = "Grid Search"

    print(f"\n[FINAL RESULT] Best method: {method}")
    print(f"Best Score: {best_score:.4f}")
    print(f"Best Params: {best_params}")

    # 結果保存
    output_dir = Path("models/alpha_gate_optimization")
    output_dir.mkdir(exist_ok=True)

    result = {
        'optimization_method': method,
        'best_score': best_score,
        'best_params': best_params,
        'golden_ratio': GOLDEN_RATIO,
        'trajectory': best_trajectory,
        'timestamp': time.time()
    }

    with open(output_dir / "best_alpha_schedule.json", "w", encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # グリッドサーチ結果も保存
    with open(output_dir / "grid_search_results.json", "w", encoding='utf-8') as f:
        json.dump(grid_results, f, indent=2, ensure_ascii=False)

    print(f"\n[COMPLETE] Results saved to {output_dir}")
    print("[RECOMMENDATION] Use these parameters in your training script:")
    print(f"  annealing_type='{best_params.get('annealing_type', 'linear')}'")
    print(f"  warmup_steps={best_params.get('warmup_steps', 10)}")
    if 'warmup_end_ratio' in best_params:
        print(f"  warmup_end_ratio={best_params.get('warmup_end_ratio', 0.8)}")
    if 'decay_rate' in best_params:
        print(f"  decay_rate={best_params.get('decay_rate', 0.95)}")
    if 'steepness' in best_params:
        print(f"  steepness={best_params.get('steepness', 10.0)}")

if __name__ == "__main__":
    main()
