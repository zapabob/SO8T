@echo off
REM Moonshot Pipeline v3.0 - Startup Launcher
REM
REM Usage:
REM   startup.bat                    - Run pipeline normally
REM   startup.bat --setup-startup    - Register Windows Task Scheduler
REM   startup.bat --remove-startup   - Remove Task Scheduler entry
REM   startup.bat --status           - Check startup status
REM
REM Features:
REM   - Rolling checkpoints every 3 minutes (5 kept)
REM   - Power failure auto-resume
REM   - SQL progress tracking
REM   - Simple English progress logging

echo ========================================================
echo Moonshot Pipeline v3.0 - Boot Launcher
echo ========================================================
echo.

REM Get Python path
where py >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON=py
) else (
    where python >nul 2>&1
    if %errorlevel% equ 0 (
        set PYTHON=python
    ) else (
        echo [ERROR] Python not found. Please install Python 3.12+
        exit /b 1
    )
)

echo [INFO] Using Python: %PYTHON%
echo.

REM Get script directory
set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%
cd %SCRIPT_DIR%

REM Parse arguments
if "%1"=="--setup-startup" goto setup_startup
if "%1"=="--remove-startup" goto remove_startup
if "%1"=="--status" goto check_status

REM Run pipeline normally
echo [RUN] Starting Moonshot Pipeline v3.0...
echo [RUN] Checkpoint interval: 180 seconds (3 min)
echo [RUN] Max rolling checkpoints: 5
echo.

%PYTHON% -3 "%SCRIPT_DIR%\scripts\utils\boot_pipeline_launcher.py" --use-existing-datasets

echo.
echo [DONE] Pipeline finished. Exit code: %errorlevel%
exit /b %errorlevel%

:setup_startup
echo [SETUP] Configuring Windows Task Scheduler for auto-start...
echo.
%PYTHON% -3 "%SCRIPT_DIR%\scripts\utils\boot_pipeline_launcher.py" --setup-startup
goto :eof

:remove_startup
echo [SETUP] Removing Windows Task Scheduler entry...
echo.
%PYTHON% -3 "%SCRIPT_DIR%\scripts\utils\boot_pipeline_launcher.py" --remove-startup
goto :eof

:check_status
echo [STATUS] Checking Task Scheduler...
echo.
powershell -Command "Get-ScheduledTask -TaskName 'MoonshotPipelineV3*' 2>$null | Format-Table TaskName, State, Date"
echo.
echo [STATUS] Check logs directory for recent activity:
dir /o-d "%SCRIPT_DIR%\logs\*.log" 2>nul | head -5
goto :eof
