@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

echo [SCHEDULER] Installing ABC Pipeline Task Scheduler
echo =====================================================
echo.

set "SCRIPT_DIR=%~dp0..\.."
set "PWSH_SCRIPT=%SCRIPT_DIR%\scripts\pipeline\run_abc_continuous.ps1"
set "TASK_NAME=SO8T-ABC-Pipeline"

echo [INFO] Script: %PWSH_SCRIPT%
echo [INFO] Task Name: %TASK_NAME%
echo.

if exist "%PWSH_SCRIPT%" (
    echo [OK] PowerShell script found
) else (
    echo [ERROR] PowerShell script not found: %PWSH_SCRIPT%
    exit /b 1
)

echo [SCHEDULER] Creating task "%TASK_NAME%"...
schtasks /create ^
    /tn "%TASK_NAME%" ^
    /tr "powershell -ExecutionPolicy Bypass -File '%PWSH_SCRIPT%'" ^
    /sc onlogon ^
    /rl highest ^
    /f

if %ERRORLEVEL% equ 0 (
    echo [OK] Task created successfully
    echo.
    echo [INFO] Task will run at user login.
    echo [INFO] To run immediately: schtasks /run /tn "%TASK_NAME%"
    echo [INFO] To delete: schtasks /delete /tn "%TASK_NAME%" /f
) else (
    echo [ERROR] Failed to create task
    exit /b 1
)

echo.
echo [COMPLETE] Task Scheduler installation finished
pause
