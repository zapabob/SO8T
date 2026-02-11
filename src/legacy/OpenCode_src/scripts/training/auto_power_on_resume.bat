@echo off
REM SO8T Training Auto-Resume Script
REM This script runs on power-on and automatically resumes incomplete training

echo [AUTO-RESUME] Starting SO8T training auto-resume check...
echo [AUTO-RESUME] Current time: %DATE% %TIME%

REM Change to project root directory
cd /d "%~dp0\..\.."
if errorlevel 1 (
    echo [ERROR] Failed to change directory to project root
    pause
    exit /b 1
)

REM Setup Python environment
set PYTHONPATH=%CD%;%CD%\so8t-mmllm\src;%PYTHONPATH%

REM Default configuration
set DATASET_PATH=data\integrated\so8t_integrated_training_dataset_utf8.jsonl
set CONFIG_PATH=configs\train_borea_phi35_so8t_thinking.yaml
set OUTPUT_BASE=D:/webdataset/checkpoints/training
set TIMESTAMP=%DATE:~0,4%%DATE:~5,2%%DATE:~8,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%
set OUTPUT_DIR=%OUTPUT_BASE%/so8t_auto_resume_%TIMESTAMP%

REM Create output directory if it doesn't exist
if not exist "%OUTPUT_BASE%" mkdir "%OUTPUT_BASE%"

echo [AUTO-RESUME] Checking for incomplete training sessions...

REM Search for incomplete training sessions
for /d %%i in ("%OUTPUT_BASE%\so8t_*") do (
    echo [AUTO-RESUME] Checking session: %%i

    REM Resume only if final_model doesn't exist
    if not exist "%%i\final_model" (
        REM Resume only if checkpoints exist
        dir /b "%%i\checkpoint_*" 2>nul | findstr /r "checkpoint_" >nul
        if not errorlevel 1 (
            echo [AUTO-RESUME] Found incomplete training session: %%i
            echo [AUTO-RESUME] Resuming training from checkpoint...

            REM Execute training resume
            python -3 scripts/training/train_so8t_quadruple_ppo.py ^
                --dataset "%DATASET_PATH%" ^
                --config "%CONFIG_PATH%" ^
                --output-dir "%%i" ^
                --power-on-resume

            REM Continue to next session after completion
            if errorlevel 0 (
                echo [AUTO-RESUME] Training session completed successfully: %%i
            ) else (
                echo [AUTO-RESUME] Training session failed or interrupted: %%i
            )
        )
    )
)

REM Check if new training session is needed
dir /b "%OUTPUT_BASE%\so8t_*" 2>nul | findstr /r "so8t_" >nul
if errorlevel 1 (
    echo [AUTO-RESUME] No existing training sessions found. Starting new session...

    REM Start new training session
    python -3 scripts/training/train_so8t_quadruple_ppo.py ^
        --dataset "%DATASET_PATH%" ^
        --config "%CONFIG_PATH%" ^
        --output-dir "%OUTPUT_DIR%" ^
        --auto-resume

    if errorlevel 0 (
        echo [AUTO-RESUME] New training session completed successfully
    ) else (
        echo [AUTO-RESUME] New training session failed or interrupted
    )
)

echo [AUTO-RESUME] Auto-resume check completed at %DATE% %TIME%

REM Audio notification (optional)
powershell -ExecutionPolicy Bypass -File "C:\Users\downl\Desktop\SO8T\scripts\utils\play_audio_notification.ps1"

pause

