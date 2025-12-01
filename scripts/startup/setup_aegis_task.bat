@echo off
chcp 65001 >nul
echo [AEGIS] Setting up automatic pipeline task...

REM Get Python path
for /f "tokens=*" %%i in ('where python') do set PYTHON_PATH=%%i

REM Set script path
set SCRIPT_PATH=%~dp0..\automation\automatic_aegis_phi35_thinking_pipeline.py

REM Create task command with delayed expansion
setlocal enabledelayedexpansion
set TASK_COMMAND="!PYTHON_PATH!" "!SCRIPT_PATH!" --resume

echo Task Command: !TASK_COMMAND!

REM Create the task (requires admin privileges)
schtasks /create /tn "SO8T_AEGIS_Automatic_Pipeline" /tr "!TASK_COMMAND!" /sc ONLOGON /rl HIGHEST /delay 0000:30 /f

if %ERRORLEVEL% EQU 0 (
    echo [OK] Task created successfully!
    echo.
    echo Task Details:
    schtasks /query /tn "SO8T_AEGIS_Automatic_Pipeline" /v /fo list | findstr /C:"TaskName" /C:"Task To Run" /C:"Status"
) else (
    echo [NG] Failed to create task (Error: %ERRORLEVEL%)
)

echo.
echo Press any key to continue...
pause >nul

REM Play audio notification
powershell -ExecutionPolicy Bypass -File "%~dp0utils\play_audio_notification.ps1"
