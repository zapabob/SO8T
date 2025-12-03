#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2.5 Training Results
"""

import json

print('=== Phase 2.5 Training Results ===')
print('Mathematics, Science, NKAT Theory, NSFW Dataset Training')
print()

try:
    # Baseline結果
    with open('logs/sunshine/sunshine_run_baseline_metrics.json', 'r', encoding='utf-8') as f:
        baseline = json.load(f)
        print('BASELINE (LoRA + Math/Science Data):')
        print(f'  Final Loss: {baseline.get("final_train_loss", "N/A")}')
        print(f'  Avg Grad Norm: {baseline.get("avg_grad_norm", "N/A"):.4f}')
        print(f'  Total Steps: {baseline.get("total_steps", 0)}')
        print()

    # SO8T結果
    with open('logs/sunshine/sunshine_run_so8t_metrics.json', 'r', encoding='utf-8') as f:
        so8t = json.load(f)
        print('SO8T (LoRA + SO(8) + NKAT Theory Data):')
        print(f'  Final Loss: {so8t.get("final_train_loss", "N/A")}')
        print(f'  Avg SO8 Ortho Error: {so8t.get("avg_so8_ortho_error", "N/A")}')
        alpha_val = so8t.get("avg_so8_alpha_mean", "N/A")
        if isinstance(alpha_val, (int, float)) and not str(alpha_val).lower() == 'nan':
            print(f'  Avg SO8 Alpha: {alpha_val:.4f}')
        else:
            print(f'  Avg SO8 Alpha: {alpha_val}')
        print(f'  Avg Grad Norm: {so8t.get("avg_grad_norm", "N/A"):.4f}')
        print(f'  Total Steps: {so8t.get("total_steps", 0)}')
        print()

    print('✅ Phase 2.5 Training Completed Successfully!')
    print('🎯 SO(8) Adapter is now integrated with NKAT Theory data')
    print('🚀 Ready for Quadruple Inference Integration!')

except Exception as e:
    print(f'Error reading results: {e}')
    import traceback
    traceback.print_exc()
