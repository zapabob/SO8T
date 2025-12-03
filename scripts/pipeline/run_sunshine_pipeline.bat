@echo off
chcp 65001 >nul
echo 🌞 SO8T SUNSHINE PIPELINE 🌞
echo ================================
echo.

echo [STEP 1] Preparing Sunshine Environment...
echo.

REM ログディレクトリ作成
if not exist "logs\sunshine" mkdir logs\sunshine

REM コマンドライン引数チェック: skip_baseline を指定するとBaselineをスキップ
if "%1"=="skip_baseline" goto skip_baseline

echo [STEP 2] Starting Baseline Run (LoRA only)...
echo.
py -3 scripts/pipeline/sunshine_pipeline.py baseline
goto so8t_run

:skip_baseline
echo [STEP 2] Skipping Baseline Run (already completed)...
echo.

:so8t_run
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
py -3 scripts\pipeline\sunshine_analysis.py

echo [STEP 6] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts/utils/play_audio_notification.ps1"

echo.
echo ✅ SUNSHINE PIPELINE COMPLETED!
echo 📁 Check results in: logs/sunshine/
echo 📊 View comparison charts with: py -3 scripts/analysis/analyze_sunshine_results.py
echo.
