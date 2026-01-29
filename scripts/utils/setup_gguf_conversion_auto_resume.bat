@echo off
chcp 65001 >nul
echo [AUTO-RESUME] Setting up GGUF conversion auto-resume on power-up
echo ============================================================

REM ============================================================
REM AUTOSTART SETUP DISABLED (default)
REM To enable creation of startup tasks, run with:
REM   set SO8T_ENABLE_AUTOSTART=1 && setup_gguf_conversion_auto_resume.bat
REM ============================================================
if /I NOT "%SO8T_ENABLE_AUTOSTART%"=="1" (
    echo [INFO] GGUF conversion auto-resume task setup is disabled by default.
    echo [INFO] Set SO8T_ENABLE_AUTOSTART=1 to intentionally enable.
    exit /b 0
)

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%..\.."
set "CONVERSION_SCRIPT=%PROJECT_ROOT%\scripts\conversion\convert_aegis_v22_with_imatrix.py"
set "LOG_DIR=%PROJECT_ROOT%\_docs"
set "TASK_NAME=GGUF_Conversion_Auto_Resume"

echo [INFO] Project root: %PROJECT_ROOT%
echo [INFO] Conversion script: %CONVERSION_SCRIPT%
echo [INFO] Task name: %TASK_NAME%

REM 管理者権限チェック
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [OK] Running with administrator privileges
) else (
    echo [ERROR] Administrator privileges required
    echo Please run as administrator
    pause
    exit /b 1
)

REM ログディレクトリ作成
if not exist "%LOG_DIR%" (
    mkdir "%LOG_DIR%"
    echo [OK] Created log directory: %LOG_DIR%
)

REM 既存のタスクを削除（存在する場合）
schtasks /delete /tn "%TASK_NAME%" /f >nul 2>&1
if %errorLevel% == 0 (
    echo [OK] Deleted existing task: %TASK_NAME%
)

REM 新しいタスクを作成（電源投入時に実行）
schtasks /create /tn "%TASK_NAME%" /tr "cmd /c \"cd /d \"%PROJECT_ROOT%\" && py -3 \"%CONVERSION_SCRIPT%\" --hf-model models/aegis_v22_hf --output-dir H:/from_D/webdataset/gguf_models/aegis_v22_imatrix --calibration-data data/calibration/math_calibration_data.txt > \"%LOG_DIR%\%TASK_NAME%_log.txt\" 2>&1\"" /sc onlogon /rl highest /f

if %errorLevel% == 0 (
    echo [SUCCESS] Auto-resume task created successfully
    echo [INFO] Task will run on user logon with highest privileges
    echo [INFO] Log file: %LOG_DIR%\%TASK_NAME%_log.txt
    echo [INFO] The system will automatically resume GGUF conversion after power-up
    echo.
    echo [AUDIO] Playing completion notification...
    powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%play_audio_notification.ps1"
) else (
    echo [ERROR] Failed to create auto-resume task
    echo [INFO] Check Windows Task Scheduler for details
)

echo.
echo Press any key to continue...
pause >nul
