@echo off
REM ========================================
REM SO8T全自動パイプライン 統合セットアップスクリプト
REM master_automated_pipeline と parallel_pipeline_manager の両方をセットアップ
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
    powershell -ExecutionPolicy Bypass -Command "Start-Process powershell.exe -Verb RunAs -ArgumentList '-ExecutionPolicy Bypass -File \"%~dp0setup_all_auto_start.ps1\"' -Wait"
    
    exit /b %ERRORLEVEL%
)

REM プロジェクトルートパス
set "PROJECT_ROOT=%~dp0.."
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

echo ========================================
echo SO8T All Auto-Start Pipeline Setup
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

echo ========================================
echo [STEP 1] Setting up Master Automated Pipeline
echo ========================================
echo.

py -3 "%PROJECT_ROOT%\scripts\pipelines\setup_master_automated_pipeline.py"

if errorlevel 1 (
    echo.
    echo [ERROR] Master Automated Pipeline setup failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo [STEP 2] Setting up Parallel Pipeline Manager
echo ========================================
echo.

py -3 "%PROJECT_ROOT%\scripts\data\setup_parallel_pipeline_manager.py"

if errorlevel 1 (
    echo.
    echo [ERROR] Parallel Pipeline Manager setup failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo [STEP 3] Verifying All Tasks
echo ========================================
echo.

echo [INFO] Checking Master Automated Pipeline task...
schtasks /query /tn "SO8T-MasterAutomatedPipeline-AutoStart" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Master Automated Pipeline task not found
) else (
    echo [OK] Master Automated Pipeline task is registered
)

echo [INFO] Checking Parallel Pipeline Manager task...
schtasks /query /tn "SO8T-ParallelPipelineManager-AutoStart" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Parallel Pipeline Manager task not found
) else (
    echo [OK] Parallel Pipeline Manager task is registered
)

echo.
echo ========================================
echo [SUCCESS] All Auto-Start Setup Completed!
echo ========================================
echo.
echo Both pipelines will automatically run on system startup:
echo   - Master Automated Pipeline (SO8T-MasterAutomatedPipeline-AutoStart)
echo   - Parallel Pipeline Manager (SO8T-ParallelPipelineManager-AutoStart)
echo.
echo To test manually:
echo   Master Pipeline: py -3 "%PROJECT_ROOT%\scripts\pipelines\master_automated_pipeline.py" --run
echo   Parallel Manager: py -3 "%PROJECT_ROOT%\scripts\data\parallel_pipeline_manager.py" --run --daemon
echo.
pause

