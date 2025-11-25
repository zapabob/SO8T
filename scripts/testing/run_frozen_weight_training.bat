@echo off
chcp 65001 > nul
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..\..

echo.
echo ================================================================================
echo  SO8T Frozen Weight Training
echo ================================================================================
echo.

set CONFIG_FILE=%PROJECT_ROOT%\configs\train_borea_phi35_so8t_thinking_frozen.yaml
set OUTPUT_DIR=%PROJECT_ROOT%\output\frozen_weight_training

echo Using config: %CONFIG_FILE%
echo Output directory: %OUTPUT_DIR%
echo.

python "%SCRIPT_DIR%..\training\train_borea_phi35_so8t_thinking.py" --config "%CONFIG_FILE%" --output_dir "%OUTPUT_DIR%"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Training failed!
    powershell -ExecutionPolicy Bypass -File "%PROJECT_ROOT%\scripts\utils\play_audio_notification.ps1"
    exit /b %ERRORLEVEL%
) else (
    echo.
    echo [SUCCESS] Training completed successfully!
    powershell -ExecutionPolicy Bypass -File "%PROJECT_ROOT%\scripts\utils\play_audio_notification.ps1"
)

echo.
echo Training results saved to: %OUTPUT_DIR%
echo.
pause
