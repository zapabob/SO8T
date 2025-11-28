@echo off
REM SO8T Training Completion Monitor Starter

echo [MONITOR] SO8T Training Completion Monitor
echo [MONITOR] Current time: %DATE% %TIME%

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

echo [MONITOR] Starting training completion monitor...

python -3 scripts/monitor_training_completion.py --start

if errorlevel 0 (
    echo [SUCCESS] Training monitor started successfully
) else (
    echo [ERROR] Failed to start training monitor
)

REM 音声通知
powershell -ExecutionPolicy Bypass -File "scripts/utils/play_audio_notification.ps1"

pause
