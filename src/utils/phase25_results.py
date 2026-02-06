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

    # 比較分析
    print('=== Performance Comparison Analysis ===')
    baseline_loss = baseline.get("final_train_loss", float('inf'))
    so8t_loss = so8t.get("final_train_loss", float('inf'))

    if baseline_loss != float('inf') and so8t_loss != float('inf'):
        if baseline_loss > 0:
            loss_ratio = so8t_loss / baseline_loss
            loss_improvement = (baseline_loss - so8t_loss) / baseline_loss * 100
            print('.2f')
            print('.2f')
        else:
            print('[NG] Baseline loss is 0 or invalid - cannot compute ratio')
    print()

    # 問題点の分析
    issues = []
    if so8t_loss == 0.0:
        issues.append("SO8T Final Loss is 0.0 - adapter may not be learning")
    if str(so8t.get("avg_grad_norm", "")).lower() == 'nan':
        issues.append("SO8T gradient norms are NaN - training not working")
    if str(so8t.get("avg_so8_ortho_error", "")).lower() == 'nan':
        issues.append("SO8T orthogonality error not computed")

    if issues:
        print("[WARN]  WARNING: Issues detected:")
        for issue in issues:
            print(f"   - {issue}")
        print()
        print("[FIX] DEBUGGING REQUIRED:")
        print("   1. Check SO(8) adapter alpha parameter updates")
        print("   2. Verify gradient flow to adapter parameters")
        print("   3. Confirm orthogonality constraints are working")
        print("   4. Test with higher learning rate for adapter")
        print()
        print("[NG] NOT READY for LM-eval testing - fix training issues first")
    else:
        print("[OK] Phase 2.5 Training Completed Successfully!")
        print("[TARGET] SO(8) Adapter is now integrated with NKAT Theory data")
        print("[START] Ready for Quadruple Inference Integration!")
        print("[STATS] Ready for LM-eval AB testing")

except Exception as e:
    print(f'Error reading results: {e}')
    import traceback
    traceback.print_exc()
