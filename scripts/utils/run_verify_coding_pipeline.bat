@echo off
chcp 65001 >nul
echo [VERIFY] Running coding pipeline verification...
echo ========================================

cd /d "%~dp0\..\.."

py -3 scripts\utils\verify_coding_pipeline.py --config configs\unified_master_pipeline_config.yaml

if %ERRORLEVEL% EQU 0 (
    echo [OK] Verification completed successfully
) else (
    echo [ERROR] Verification failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo [AUDIO] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"































































