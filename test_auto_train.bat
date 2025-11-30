@echo off
:: Test version of auto_train.bat
cd /d "C:\Users\downl\Desktop\SO8T"

:: Set Python path for module imports
set PYTHONPATH=%CD%;%CD%\scripts;%CD%\utils

:: Create logs directory
if not exist "logs" mkdir logs

:: Simple timestamp
set TIMESTAMP=%DATE:~0,4%%DATE:~5,2%%DATE:~8,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%

:: Log file
set LOGFILE=logs\test_auto_training_%TIMESTAMP%.log

echo [%DATE% %TIME%] ===== TEST AEGIS Auto Training ===== >> %LOGFILE%

:: Check system resources
echo [%DATE% %TIME%] Testing system resources... >> %LOGFILE%
py -3 scripts/utils/check_system_resources.py >> %LOGFILE% 2>&1

:: Check exit code
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] System check passed >> %LOGFILE%
    echo Test completed successfully. Check %LOGFILE% for details.
) else (
    echo [ERROR] System check failed >> %LOGFILE%
    echo Test failed. Check %LOGFILE% for details.
)

pause
