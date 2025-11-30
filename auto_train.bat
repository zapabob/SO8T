@echo off
:: AEGIS-v2.0 Auto Training Startup Batch
:: Windows power-on automatic training resume

cd /d "C:\Users\downl\Desktop\SO8T"

:: Create logs directory
if not exist "logs" mkdir logs

:: Timestamp
set TIMESTAMP=%DATE:~0,4%%DATE:~5,2%%DATE:~8,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%

:: Log file
set LOGFILE=logs\auto_training_%TIMESTAMP%.log

echo [%DATE% %TIME%] ===== AEGIS-v2.0 Auto Training Started ===== >> %LOGFILE%
echo [%DATE% %TIME%] Working Directory: %CD% >> %LOGFILE%
echo [%DATE% %TIME%] Log File: %LOGFILE% >> %LOGFILE%


:: Set Python path for module imports
set PYTHONPATH=%CD%;%CD%\scripts;%CD%\utils

:: Check system resources
echo [%DATE% %TIME%] Checking system resources... >> %LOGFILE%
py -3 scripts/utils/check_system_resources.py >> %LOGFILE% 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%DATE% %TIME%] WARNING: System resource check failed >> %LOGFILE%
)

:: Start AEGIS training
echo [%DATE% %TIME%] Starting AEGIS training pipeline... >> %LOGFILE%
echo [%DATE% %TIME%] Starting AEGIS training pipeline...
py -3 scripts/training/aegis_v2_training_pipeline.py >> %LOGFILE% 2>&1

:: Check exit code
set EXITCODE=%ERRORLEVEL%
echo [%DATE% %TIME%] Training process finished with exit code: %EXITCODE% >> %LOGFILE%

if %EXITCODE% EQU 0 (
    echo [%DATE% %TIME%] SUCCESS: Training completed successfully >> %LOGFILE%
    echo [SUCCESS] AEGIS training completed successfully
    goto :success_notification
) else (
    echo [%DATE% %TIME%] ERROR: Training failed with exit code %EXITCODE% >> %LOGFILE%
    echo [ERROR] AEGIS training failed with exit code %EXITCODE%
    goto :error_handling
)

:success_notification
:: Success notification
echo [%DATE% %TIME%] Playing success notification... >> %LOGFILE%
powershell -Command "[console]::beep(800, 300); start-sleep -milliseconds 200; [console]::beep(1000, 300); start-sleep -milliseconds 200; [console]::beep(1200, 500)" >> %LOGFILE% 2>&1

goto :finish

:error_handling
    :: Error handling
    echo [%DATE% %TIME%] Error occurred during training >> %LOGFILE%
    echo [ERROR] Training failed. Check log file: %LOGFILE%

    :: Display error details
echo.
echo ========================================
echo AEGIS TRAINING ERROR
echo ========================================
echo Exit Code: %EXITCODE%
echo Log File: %LOGFILE%
echo.
echo Press any key to exit...
pause >nul

goto :finish

:finish
echo [%DATE% %TIME%] ===== AEGIS-v2.0 Auto Training Finished ===== >> %LOGFILE%
echo [FINISHED] AEGIS training finished. Check logs for details.
echo Log saved to: %LOGFILE%

goto :eof

