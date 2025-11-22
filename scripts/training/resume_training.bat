@echo off
chcp 65001 >nul
echo [INFO] Resuming training from checkpoint...
echo ========================================

cd /d "%~dp0\..\.."

echo [STEP 1] Checking for existing checkpoints...
if exist "D:\webdataset\checkpoints\training\borea_phi35_so8t_thinking\training_session.json" (
    echo [OK] Found session file
    type "D:\webdataset\checkpoints\training\borea_phi35_so8t_thinking\training_session.json"
) else (
    echo [INFO] No session file found. Will start new training if no checkpoints exist.
)

echo.
echo [STEP 2] Starting training with auto-resume...
py -3 scripts/training/train_borea_phi35_so8t_thinking.py ^
    --config configs/train_borea_phi35_so8t_thinking.yaml ^
    --auto-resume

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Training failed with error code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [AUDIO] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

pause


















































