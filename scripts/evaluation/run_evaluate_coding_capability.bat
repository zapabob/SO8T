@echo off
chcp 65001 >nul
echo [EVAL] Running coding capability evaluation...
echo ========================================

cd /d "%~dp0\..\.."

if "%1"=="" (
    echo [ERROR] Please provide test data path as argument
    echo Usage: run_evaluate_coding_capability.bat D:\webdataset\coding_training_data
    exit /b 1
)

set TEST_DATA=%1
set OUTPUT_DIR=D:\webdataset\evaluation\coding_capability

if "%2"=="" (
    set MODEL_NAME=so8t_coding_model
) else (
    set MODEL_NAME=%2
)

py -3 scripts\evaluation\evaluate_coding_capability.py --test-data "%TEST_DATA%" --output "%OUTPUT_DIR%" --model-name "%MODEL_NAME%"

if %ERRORLEVEL% EQU 0 (
    echo [OK] Evaluation completed successfully
) else (
    echo [ERROR] Evaluation failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo [AUDIO] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"












































































