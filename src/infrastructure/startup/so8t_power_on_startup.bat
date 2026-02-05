@echo off
REM SO8T Power-On Startup Script
REM 目的: 電源投入時にパイプラインを自動再開

REM ============================================================
REM AUTOSTART KILL-SWITCH (default: disabled)
REM Set SO8T_ENABLE_AUTOSTART=1 to allow running on startup.
REM ============================================================
if /I NOT "%SO8T_ENABLE_AUTOSTART%"=="1" (
    echo [INFO] SO8T power-on autostart is disabled. Exiting.
    exit /b 0
)

echo ========================================
echo SO8T Power-On Auto Startup
echo ========================================
echo Started at: %DATE% %TIME%
echo Computer: %COMPUTERNAME%
echo User: %USERNAME%
echo ========================================

REM 作業ディレクトリ
cd /d "C:\Users\downl\Desktop\SO8T"

REM ログ設定
set LOG_DIR=logs\startup
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

set TIMESTAMP=%DATE:~0,4%%DATE:~5,2%%DATE:~8,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set TIMESTAMP=%TIMESTAMP::=%
set TIMESTAMP=%TIMESTAMP:/=_%
set LOG_FILE=%LOG_DIR%\so8t_startup_%TIMESTAMP%.log

REM チェックポイント設定
set SO8T_CHECKPOINT_INTERVAL=300
set SO8T_ROLLING_CHECKPOINTS=5

echo Starting Moonshot 2025-2026 Pipeline... >> "%LOG_FILE%"
echo Log file: %LOG_FILE% >> "%LOG_FILE%"
echo ======================================== >> "%LOG_FILE%"

REM Python起動 (優先的に py -3)
set PYTHON_EXE=py -3

if not exist "run_moonshot_pipeline_2025_2026.py" (
    echo [ERROR] Python script not found: run_moonshot_pipeline_2025_2026.py >> "%LOG_FILE%"
    goto :error
)

echo [INFO] Starting Moonshot pipeline (auto-resume enabled)... >> "%LOG_FILE%"
start "SO8T_Automated_Pipeline" /B cmd /c "%PYTHON_EXE% run_moonshot_pipeline_2025_2026.py --use-existing-datasets >> \"%LOG_FILE%\" 2>&1"

timeout /t 5 /nobreak > nul
tasklist /FI "IMAGENAME eq python.exe" 2>nul | findstr "python.exe" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Moonshot pipeline started successfully >> "%LOG_FILE%"
    echo Process started in background >> "%LOG_FILE%"
) else (
    echo [ERROR] Failed to start Moonshot pipeline >> "%LOG_FILE%"
    goto :error
)

echo ======================================== >> "%LOG_FILE%"
echo SO8T Power-On Startup completed successfully >> "%LOG_FILE%"
echo ======================================== >> "%LOG_FILE%"
goto :end

:error
echo ======================================== >> "%LOG_FILE%"
echo SO8T Power-On Startup FAILED >> "%LOG_FILE%"
echo ======================================== >> "%LOG_FILE%"
powershell -ExecutionPolicy Bypass -Command "[System.Console]::Beep(800, 1000)" 2>nul

:end
echo Startup script completed at: %DATE% %TIME%
