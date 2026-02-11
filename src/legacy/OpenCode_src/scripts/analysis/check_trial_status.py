#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optunaトライアルのトレーニング状態確認スクリプト
"""

from pathlib import Path
import json

def check_trial_status(trial_num):
    """指定トライアルの状態を確認"""
    trial_path = Path(f'H:/from_D/webdataset/checkpoints/aegis_v21_training/sft_optuna_trial_{trial_num}/checkpoint-20/trainer_state.json')

    if not trial_path.exists():
        print(f"Trial {trial_num}: trainer_state.json not found")
        return None

    try:
        with open(trial_path, 'r', encoding='utf-8') as f:
            state = json.load(f)

        log_history = state.get('log_history', [])
        if log_history:
            print(f'Trial {trial_num} training log (last 5 entries):')
            for i, entry in enumerate(log_history[-5:]):
                step = entry.get('step', '?')
                loss = entry.get('train_loss', '?')
                lr = entry.get('learning_rate', '?')
                print(f'  Step {step}: loss={loss}, lr={lr}')

        global_step = state.get('global_step', 0)
        print(f'Global step: {global_step}')
        print(f'Total log entries: {len(log_history)}')

        # 最後の損失を取得
        final_loss = log_history[-1].get('train_loss', float('inf')) if log_history else float('inf')
        print(f'Final loss: {final_loss}')

        return final_loss

    except Exception as e:
        print(f"Error reading trial {trial_num}: {e}")
        return None

def find_best_trial():
    """最良のトライアルを見つける"""
    training_dir = Path('H:/from_D/webdataset/checkpoints/aegis_v21_training')
    results = []

    for trial_dir in training_dir.glob('sft_optuna_trial_*'):
        if trial_dir.is_dir():
            trial_num = int(trial_dir.name.replace('sft_optuna_trial_', ''))
            loss = check_trial_status(trial_num)
            if loss is not None and loss != float('inf'):
                results.append((trial_num, loss))
            print("-" * 40)

    if results:
        best_trial, best_loss = min(results, key=lambda x: x[1])
        print(f"\nBest Trial: {best_trial} with loss {best_loss:.6f}")
        return best_trial, best_loss
    else:
        print("\nNo valid trials found")
        return None, None

if __name__ == "__main__":
    find_best_trial()
