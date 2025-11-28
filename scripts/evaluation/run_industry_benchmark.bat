@echo off
setlocal enabledelayedexpansion

:: UTF-8エンコーディング設定
chcp 65001 > nul

echo ========================================================
echo   Industry Standard Benchmark Suite
echo   Model A (Borea-Phi3.5) vs AEGIS
echo ========================================================
echo.

:: Pythonパス設定
set PYTHON_CMD=.\.venv\Scripts\python.exe

:: 1. ベンチマーク実行
echo [1/4] Running Integrated Benchmark...
%PYTHON_CMD% scripts/evaluation/integrated_industry_benchmark.py
if %ERRORLEVEL% NEQ 0 (
    echo Error: Benchmark failed.
    exit /b %ERRORLEVEL%
)

:: 2. 可視化生成
echo.
echo [2/4] Generating Visualizations...
%PYTHON_CMD% scripts/evaluation/visualize_industry_benchmark.py
if %ERRORLEVEL% NEQ 0 (
    echo Warning: Visualization failed.
)

:: 3. README更新
echo.
echo [3/4] Updating README...
%PYTHON_CMD% scripts/evaluation/update_readme_benchmarks.py
if %ERRORLEVEL% NEQ 0 (
    echo Warning: README update failed.
)

:: 4. 完了通知
echo.
echo [4/4] Benchmark Suite Completed!
if exist "scripts/utils/play_audio_notification.ps1" (
    powershell -ExecutionPolicy Bypass -File "scripts/utils/play_audio_notification.ps1"
)

echo.
echo Results are saved in D:/webdataset/benchmark_results/industry_standard/
pause































