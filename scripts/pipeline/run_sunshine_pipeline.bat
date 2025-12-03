@echo off
chcp 65001 >nul
echo 🌞 SO8T SUNSHINE PIPELINE 🌞
echo ================================
echo.

echo [STEP 1] Preparing Sunshine Environment...
echo.

REM ログディレクトリ作成
if not exist "logs\sunshine" mkdir logs\sunshine

echo [STEP 2] Starting Baseline Run (LoRA only)...
echo.
py -3 scripts/pipeline/sunshine_pipeline.py baseline

echo.
echo [STEP 3] Starting SO8T Run (LoRA + SO(8) Adapter)...
echo.
py -3 scripts/pipeline/sunshine_pipeline.py so8t

echo.
echo [STEP 4] Analyzing Results...
echo.

REM 結果分析（簡易版）
echo [ANALYSIS] Training Logs Summary:
echo ===========================================
if exist "logs\sunshine\sunshine_run_baseline_metrics.json" (
    echo Baseline metrics:
    type logs\sunshine\sunshine_run_baseline_metrics.json | findstr "final_train_loss\|avg_so8_ortho_error"
) else (
    echo Baseline metrics not found
)

if exist "logs\sunshine\sunshine_run_so8t_metrics.json" (
    echo.
    echo SO8T metrics:
    type logs\sunshine\sunshine_run_so8t_metrics.json | findstr "final_train_loss\|avg_so8_ortho_error"
) else (
    echo SO8T metrics not found
)

echo.
echo [STEP 5] Generating Comparison Report...
echo.

REM Pythonで比較レポート生成
py -3 -c "
import json
import pandas as pd
from pathlib import Path

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
            print(f'  Final Loss: {data.get(\"final_train_loss\", \"N/A\")}')
            print(f'  Avg SO8 Ortho Error: {data.get(\"avg_so8_ortho_error\", \"N/A\")}')
            print(f'  Total Steps: {data.get(\"total_steps\", 0)}')
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
            print(f'  Initial loss: {df[\"train_loss\"].dropna().iloc[0]:.4f}')
            print(f'  Final loss: {df[\"train_loss\"].dropna().iloc[-1]:.4f}')
        print()
"

echo [STEP 6] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts/utils/play_audio_notification.ps1"

echo.
echo ✅ SUNSHINE PIPELINE COMPLETED!
echo 📁 Check results in: logs/sunshine/
echo 📊 View comparison charts with: py -3 scripts/analysis/analyze_sunshine_results.py
echo.
