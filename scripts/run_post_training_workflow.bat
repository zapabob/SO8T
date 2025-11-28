@echo off
REM SO8T Post-Training Workflow Runner
REM HFモデル完成後のGGUF変換→ベンチマーク→ABテスト自動実行

echo [POST-TRAINING] Starting SO8T Post-Training Workflow...
echo [POST-TRAINING] Current time: %DATE% %TIME%

REM UTF-8エンコーディング設定
chcp 65001 >nul

REM プロジェクトルートに移動
cd /d "%~dp0\.."
if errorlevel 1 (
    echo [ERROR] Failed to change directory to project root
    pause
    exit /b 1
)

REM Python環境設定
set PYTHONPATH=%CD%;%CD%\so8t-mmllm\src;%PYTHONPATH%

echo [POST-TRAINING] Checking for completed training sessions...

REM トレーニング完了チェック
python -3 scripts/post_training_workflow.py --run-once

if errorlevel 0 (
    echo [SUCCESS] Post-training workflow completed successfully
) else (
    echo [INFO] No completed training found or workflow in progress
)

echo [POST-TRAINING] Workflow check completed at %DATE% %TIME%

REM 音声通知
powershell -ExecutionPolicy Bypass -File "scripts/utils/play_audio_notification.ps1"

pause
