#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
サンシャイン実験最終結果表示
"""

import json

print("=== SUNSHINE EXPERIMENT FINAL RESULTS ===")

try:
    with open('logs/sunshine/sunshine_run_baseline_metrics.json', 'r', encoding='utf-8') as f:
        baseline = json.load(f)
        print('BASELINE (LoRA only):')
        print(f'  Final Loss: {baseline.get("final_train_loss", "N/A")}')
        print(f'  Avg Grad Norm: {baseline.get("avg_grad_norm", "N/A")}')
        print(f'  Total Steps: {baseline.get("total_steps", 0)}')

    with open('logs/sunshine/sunshine_run_so8t_metrics.json', 'r', encoding='utf-8') as f:
        so8t = json.load(f)
        print('SO8T (LoRA + SO(8)):')
        print(f'  Final Loss: {so8t.get("final_train_loss", "N/A")}')
        print(f'  Avg SO8 Ortho Error: {so8t.get("avg_so8_ortho_error", "N/A")}')
        print(f'  Avg Grad Norm: {so8t.get("avg_grad_norm", "N/A")}')
        print(f'  Total Steps: {so8t.get("total_steps", 0)}')

    print('\n✅ Sunshine Pipeline Completed Successfully!')

except Exception as e:
    print(f'Error reading results: {e}')
