@echo off
:: SO8T Automatic Pipeline Runner
:: 3分間隔でSO8Tパイプラインを実行し、ローリングチェックポイントを管理

cd /d "C:\Users\downl\Desktop\SO8T"

:: Create logs directory
if not exist "logs" mkdir logs

:: Timestamp
set TIMESTAMP=%DATE:~0,4%%DATE:~5,2%%DATE:~8,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%

:: Log file
set LOGFILE=logs\so8t_auto_pipeline_%TIMESTAMP%.log

echo [%DATE% %TIME%] ===== SO8T Auto Pipeline Started ===== >> %LOGFILE%
echo [%DATE% %TIME%] Working Directory: %CD% >> %LOGFILE%
echo [%DATE% %TIME%] Log File: %LOGFILE% >> %LOGFILE%

:: Set Python path for module imports
set PYTHONPATH=%CD%;%CD%\scripts;%CD%\utils

:: Check Python availability
py -3 --version >> %LOGFILE% 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%DATE% %TIME%] ERROR: Python not found >> %LOGFILE%
    goto :error
)

:: Start SO8T Auto Pipeline Runner
echo [%DATE% %TIME%] Starting SO8T Auto Pipeline Runner... >> %LOGFILE%
echo [%DATE% %TIME%] Starting SO8T Auto Pipeline Runner...

py -3 scripts/automation/so8t_auto_pipeline_runner.py ^
    --pipeline-script scripts/training/train_borea_phi35_so8t_ppo.py ^
    --dataset-path data/so8t_quadruple_dataset.jsonl ^
    --output-dir outputs/so8t_auto_pipeline ^
    --checkpoint-dir checkpoints/so8t_rolling ^
    --interval-minutes 3 ^
    --max-checkpoints 5 ^
    >> %LOGFILE% 2>&1

:: Check exit code
set EXITCODE=%ERRORLEVEL%
echo [%DATE% %TIME%] Auto Pipeline finished with exit code: %EXITCODE% >> %LOGFILE%

if %EXITCODE% EQU 0 (
    echo [%DATE% %TIME%] SUCCESS: Auto Pipeline completed successfully >> %LOGFILE%
    echo [SUCCESS] SO8T Auto Pipeline completed successfully
    goto :success_notification
) else (
    echo [%DATE% %TIME%] ERROR: Auto Pipeline failed with exit code %EXITCODE% >> %LOGFILE%
    echo [ERROR] SO8T Auto Pipeline failed with exit code %EXITCODE%
    goto :error_handling
)

:success_notification
:: Success notification
echo [%DATE% %TIME%] Playing success notification... >> %LOGFILE%
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1" >> %LOGFILE% 2>&1

goto :finish

:error_handling
    :: Error handling
    echo [%DATE% %TIME%] Error occurred during auto pipeline execution >> %LOGFILE%
    echo [ERROR] Auto Pipeline failed. Check log file: %LOGFILE%

    :: Display error details
    echo.
    echo ========================================
    echo SO8T AUTO PIPELINE ERROR
    echo ========================================
    echo Exit Code: %EXITCODE%
    echo Log File: %LOGFILE%
    echo.
    echo Press any key to exit...
    pause >nul

    goto :finish

:finish
echo [%DATE% %TIME%] ===== SO8T Auto Pipeline Finished ===== >> %LOGFILE%
echo [FINISHED] SO8T Auto Pipeline finished. Check logs for details.
echo Log saved to: %LOGFILE%

goto :eof

:error
echo [CRITICAL] Python environment error. Cannot start SO8T Auto Pipeline.
pause
goto :eof





