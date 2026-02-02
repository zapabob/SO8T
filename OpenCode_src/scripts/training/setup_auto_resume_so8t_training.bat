@echo off
REM ========================================
REM SO8T Training Auto Resume Setup Script
REM Windowsタスクスケジューラーに自動実行タスクを登録
REM ========================================
REM
REM Usage:
REM   cmd /c scripts\training\setup_auto_resume_so8t_training.bat
REM   or
REM   .\scripts\training\setup_auto_resume_so8t_training.bat
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
set "STARTUP_SCRIPT=%PROJECT_ROOT%\scripts\training\auto_resume_so8t_training_startup.bat"

if not exist "%STARTUP_SCRIPT%" (
    echo [ERROR] Startup script not found: %STARTUP_SCRIPT%
    pause
    exit /b 1
)

REM タスク名
set "TASK_NAME=SO8T-Training-AutoResume"

echo ========================================
echo SO8T Training Auto Resume Setup
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

REM タスクスケジューラーコマンド（システム起動時）
set "TASK_COMMAND=schtasks /create /tn \"%TASK_NAME%\" /tr \"%STARTUP_SCRIPT%\" /sc onstart /ru SYSTEM /rl highest /f"

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

REM タスクの詳細を表示
echo [INFO] Task details:
schtasks /query /tn "%TASK_NAME%" /fo LIST /v

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo The task "%TASK_NAME%" will run automatically on system startup.
echo To remove the task, run:
echo   schtasks /delete /tn "%TASK_NAME%" /f
echo.

pause
exit /b 0








