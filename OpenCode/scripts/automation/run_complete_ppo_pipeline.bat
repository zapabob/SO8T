@echo off
chcp 65001 >nul
echo [SO8T] Starting Complete PPO Learning Pipeline
echo ===============================================

cd /d "%~dp0\..\.."

echo [STEP 1] Running PPO Training Pipeline...
python scripts\automation\complete_ppo_pipeline_with_power_on_automation.py

if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Complete PPO Pipeline finished successfully!
    echo [AUDIO] Playing completion notification...
    powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"
) else (
    echo [ERROR] PPO Pipeline failed with error code %ERRORLEVEL%
    echo [AUDIO] Playing error notification...
    powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"
)

echo [DONE] Pipeline execution completed
pause
