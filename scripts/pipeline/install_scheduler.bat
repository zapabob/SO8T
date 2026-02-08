@echo off
chcp 65001 >nul
echo ========================================
echo   AEGIS Evolved Pipeline - Task Scheduler Install
echo ========================================
echo.
echo This script will install a scheduled task to run
echo the Evolved Shinka Pipeline on power-on.
echo.
echo Running as administrator is required.
echo.

net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Administrator privileges required.
    echo [ERROR] Please run this script as administrator.
    echo.
    echo To run as administrator:
    echo 1. Right-click this file
    echo 2. Select "Run as administrator"
    echo.
    pause
    exit /b 1
)

echo [OK] Administrator privileges confirmed.
echo.

powershell.exe -ExecutionPolicy Bypass -File "%~dp0power_on_auto_resume.ps1" -InstallScheduler

echo.
if %errorlevel% equ 0 (
    echo [OK] Scheduled task installed successfully.
    echo.
    echo The pipeline will now automatically resume
    echo when the computer powers on.
) else (
    echo [NG] Failed to install scheduled task.
)

echo.
pause
