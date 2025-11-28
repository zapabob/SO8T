@echo off
REM ========================================
REM SO8T完全自動化マスターパイプライン セットアップスクリプト
REM Windowsタスクスケジューラへの自動登録
REM ========================================

chcp 65001 >nul
setlocal enabledelayedexpansion

REM 管理者権限チェック
net session >nul 2>&1
if errorlevel 1 (
    echo [INFO] This script requires administrator privileges.
    echo [INFO] Restarting with administrator privileges...
    echo.
    
    REM PowerShellスクリプトを使用して管理者権限で再起動
    powershell -ExecutionPolicy Bypass -Command "Start-Process powershell.exe -Verb RunAs -ArgumentList '-ExecutionPolicy Bypass -File \"%~dp0setup_master_automated_pipeline.ps1\"' -Wait"
    
    exit /b %ERRORLEVEL%
)

REM プロジェクトルートパス
set "PROJECT_ROOT=%~dp0.."
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

echo ========================================
echo SO8T Master Automated Pipeline Setup
echo ========================================
echo.
echo Project Root: %PROJECT_ROOT%
echo.

REM Python実行ファイルの検出
where py >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH
    echo [ERROR] Please install Python or add it to PATH
    pause
    exit /b 1
)

REM セットアップスクリプトの実行
echo [INFO] Running setup script...
echo.

py -3 "%PROJECT_ROOT%\scripts\pipelines\setup_master_automated_pipeline.py"

if errorlevel 1 (
    echo.
    echo [ERROR] Setup failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo [SUCCESS] Setup completed successfully!
echo ========================================
echo.
echo The pipeline will automatically run on system startup.
echo To test manually, run:
echo   py -3 "%PROJECT_ROOT%\scripts\pipelines\master_automated_pipeline.py" --run
echo.
pause

