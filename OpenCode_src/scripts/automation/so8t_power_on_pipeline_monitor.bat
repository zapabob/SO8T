@echo off
chcp 65001 >nul
echo [SO8T] Power-on Pipeline Monitor - Auto Start System
echo ====================================================
echo Starting SO8T PPO Pipeline with automatic monitoring
echo - Monitors pipeline progress continuously
echo - Stops on error or HF model completion
echo - Provides audio notifications
echo ====================================================
echo.

REM Change to project root
cd /d "%~dp0\..\.."
if errorlevel 1 (
    echo [ERROR] Failed to change to project root directory
    pause
    exit /b 1
)

REM Set environment
set PYTHONPATH=%CD%;%CD%\so8t-mmllm\src;%PYTHONPATH%
set ATTN_IMPLEMENTATION=eager

REM Create log directory
if not exist "logs" mkdir "logs"

REM Timestamp for log file
set TIMESTAMP=%DATE:~0,4%%DATE:~5,2%%DATE:~8,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%
set LOG_FILE=logs\so8t_power_on_monitor_%TIMESTAMP%.log

echo [SO8T-MONITOR] Starting Power-on Pipeline Monitor at %DATE% %TIME% > "%LOG_FILE%"
echo [SO8T-MONITOR] Log file: %LOG_FILE%
echo.

REM Start the monitoring Python script
echo [SO8T-MONITOR] Launching pipeline monitor...

REM Python実行ファイルの検索
set PYTHON_EXE=
if exist "C:\Python312\python.exe" (
    set PYTHON_EXE=C:\Python312\python.exe
) else if exist "C:\Python311\python.exe" (
    set PYTHON_EXE=C:\Python311\python.exe
) else if exist "C:\Python310\python.exe" (
    set PYTHON_EXE=C:\Python310\python.exe
) else if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe" (
    set PYTHON_EXE=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe
) else if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\python.exe" (
    set PYTHON_EXE=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\python.exe
) else (
    REM py launcherを使用
    set PYTHON_EXE=py -3
)

echo Using Python: %PYTHON_EXE%
%PYTHON_EXE% scripts/automation/so8t_pipeline_monitor.py >> "%LOG_FILE%" 2>&1

set MONITOR_RESULT=%errorlevel%

echo.
if %MONITOR_RESULT% equ 0 (
    echo ====================================================
    echo [SUCCESS] SO8T Pipeline Monitor completed successfully!
    echo ====================================================
    echo HF model has been completed and uploaded.
    echo.
    call powershell -ExecutionPolicy Bypass -File "scripts/utils/play_audio_notification.ps1"
) else if %MONITOR_RESULT% equ 1 (
    echo ====================================================
    echo [ERROR] SO8T Pipeline Monitor stopped due to error!
    echo ====================================================
    echo Check log file for details: %LOG_FILE%
    echo.
    REM エラー通知（ビープ音）
    echo [BEEP] Error notification
    powershell -Command "[System.Console]::Beep(800, 1000); [System.Console]::Beep(600, 1000)"
) else if %MONITOR_RESULT% equ 2 (
    echo ====================================================
    echo [STOP] SO8T Pipeline Monitor stopped by user request!
    echo ====================================================
    echo.
    powershell -ExecutionPolicy Bypass -Command "[System.Console]::Beep(1000, 500)"
) else (
    echo ====================================================
    echo [UNKNOWN] SO8T Pipeline Monitor ended with unknown code: %MONITOR_RESULT%
    echo ====================================================
    echo Check log file for details: %LOG_FILE%
    echo.
    powershell -ExecutionPolicy Bypass -Command "[System.Console]::Beep(400, 1000)"
)

echo.
echo [SO8T-MONITOR] Monitor session completed at %DATE% %TIME%
echo [SO8T-MONITOR] Full log available at: %LOG_FILE%

REM Save completion status
echo COMPLETED_AT=%DATE% %TIME% >> "%LOG_FILE%"
echo EXIT_CODE=%MONITOR_RESULT% >> "%LOG_FILE%"

pause
