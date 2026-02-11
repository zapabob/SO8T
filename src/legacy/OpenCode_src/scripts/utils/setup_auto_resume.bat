@echo off
REM ========================================
REM SO8T Auto Resume Setup Script
REM Windowsタスクスケジューラーに自動実行タスクを登録
REM ========================================
REM
REM Usage:
REM   cmd /c scripts\setup_auto_resume.bat
REM   or
REM   .\scripts\setup_auto_resume.bat
REM   or
REM   powershell -ExecutionPolicy Bypass -File scripts\setup_auto_resume.ps1
REM

chcp 65001 >nul
setlocal enabledelayedexpansion

REM 管理者権限チェック
net session >nul 2>&1
if errorlevel 1 (
    echo [ERROR] This script requires administrator privileges.
    echo [ERROR] Please run as administrator.
    pause
    exit /b 1
)

REM プロジェクトルートパス
set "PROJECT_ROOT=%~dp0.."
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

REM スタートアップスクリプトのパス
set "STARTUP_SCRIPT=%PROJECT_ROOT%\scripts\auto_resume_startup.bat"

if not exist "%STARTUP_SCRIPT%" (
    echo [ERROR] Startup script not found: %STARTUP_SCRIPT%
    pause
    exit /b 1
)

REM タスク名
set "TASK_NAME=SO8T-AutoResume"

echo ========================================
echo SO8T Auto Resume Setup
echo ========================================
echo.
echo Project Root: %PROJECT_ROOT%
echo Startup Script: %STARTUP_SCRIPT%
echo Task Name: %TASK_NAME%
echo.

REM 既存のタスクを削除（存在する場合）
schtasks /query /tn "%TASK_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Removing existing task...
    schtasks /delete /tn "%TASK_NAME%" /f >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] Failed to remove existing task
    ) else (
        echo [OK] Existing task removed
    )
)

REM 新しいタスクを作成
echo [INFO] Creating new task...
echo.

REM タスクスケジューラーコマンド
set "TASK_COMMAND=schtasks /create /tn \"%TASK_NAME%\" /tr \"%STARTUP_SCRIPT%\" /sc onlogon /rl highest /f"

REM タスクを作成
%TASK_COMMAND%

if errorlevel 1 (
    echo [ERROR] Failed to create task
    echo [ERROR] Command: %TASK_COMMAND%
    pause
    exit /b 1
)

echo.
echo [OK] Task created successfully
echo.
echo ========================================
echo Task Configuration
echo ========================================
echo Task Name: %TASK_NAME%
echo Trigger: On user logon
echo Script: %STARTUP_SCRIPT%
echo.
echo [INFO] The task will run automatically when you log in.
echo [INFO] To view the task: schtasks /query /tn "%TASK_NAME%"
echo [INFO] To delete the task: schtasks /delete /tn "%TASK_NAME%" /f
echo.

REM タスクの詳細を表示
echo [INFO] Task details:
schtasks /query /tn "%TASK_NAME%" /fo list /v

echo.
echo ========================================
echo Setup Completed
echo ========================================
echo.

REM 音声通知
if exist "%PROJECT_ROOT%\.cursor\marisa_owattaze.wav" (
    powershell -Command "if (Test-Path '%PROJECT_ROOT%\.cursor\marisa_owattaze.wav') { Add-Type -AssemblyName System.Windows.Forms; [System.Media.SoundPlayer]::new('%PROJECT_ROOT%\.cursor\marisa_owattaze.wav').Play(); Write-Host '[OK] Audio notification played successfully' -ForegroundColor Green } else { Write-Host '[WARNING] Audio file not found' -ForegroundColor Yellow }"
)

pause
exit /b 0

