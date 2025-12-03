#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sunshine Pipeline Analysis Script
サンシャイン実験結果分析
"""

import json
import pandas as pd
from pathlib import Path

def main():
    print('📊 SUNSHINE EXPERIMENT RESULTS')
    print('=' * 50)

    baseline_metrics = Path('logs/sunshine/sunshine_run_baseline_metrics.json')
    so8t_metrics = Path('logs/sunshine/sunshine_run_so8t_metrics.json')

    results = {}
    for name, path in [('Baseline', baseline_metrics), ('SO8T', so8t_metrics)]:
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results[name] = data
                print(f'{name}:')
                print(f'  Final Loss: {data.get("final_train_loss", "N/A")}')
                print(f'  Avg SO8 Ortho Error: {data.get("avg_so8_ortho_error", "N/A")}')
                print(f'  Total Steps: {data.get("total_steps", 0)}')
        else:
            print(f'{name}: Metrics file not found')
        print()

    # CSVログ比較
    baseline_csv = Path('logs/sunshine/sunshine_run_baseline_training_log.csv')
    so8t_csv = Path('logs/sunshine/sunshine_run_so8t_training_log.csv')

    for name, path in [('Baseline', baseline_csv), ('SO8T', so8t_csv)]:
        if path.exists():
            df = pd.read_csv(path)
            print(f'{name} Training Progress:')
            print(f'  Steps recorded: {len(df)}')
            if not df.empty:
                initial_loss = df["train_loss"].dropna().iloc[0] if not df["train_loss"].dropna().empty else "N/A"
                final_loss = df["train_loss"].dropna().iloc[-1] if not df["train_loss"].dropna().empty else "N/A"
                print(f'  Initial loss: {initial_loss}')
                print(f'  Final loss: {final_loss}')
            print()

if __name__ == "__main__":
    main()
