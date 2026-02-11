@echo off
REM Advanced Science Reasoning Pipeline Task Scheduler Setup Script
REM タスクスケジューラに電源投入時自動起動を設定

REM ============================================================
REM AUTOSTART SETUP DISABLED (default)
REM To enable creation of startup tasks, run with:
REM   set SO8T_ENABLE_AUTOSTART=1 && setup_advanced_science_task_scheduler.bat
REM ============================================================
if /I NOT "%SO8T_ENABLE_AUTOSTART%"=="1" (
    echo [INFO] Autostart task setup is disabled by default.
    echo [INFO] Set SO8T_ENABLE_AUTOSTART=1 to intentionally enable.
    exit /b 0
)

echo ========================================
echo Advanced Science Reasoning Pipeline Task Scheduler Setup
echo ========================================
echo Setting up power-on auto startup...
echo ========================================

REM 管理者権限チェック
net session >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Administrator privileges required!
    echo Please run this script as Administrator.
    pause
    exit /b 1
)

REM 作業ディレクトリ設定
cd /d "C:\Users\downl\Desktop\SO8T"
set SCRIPT_DIR=%CD%
set STARTUP_SCRIPT=%SCRIPT_DIR%\scripts\startup\advanced_science_pipeline_startup.bat

REM ログディレクトリ作成
if not exist "logs\startup" mkdir "logs\startup"

echo [INFO] Script directory: %SCRIPT_DIR%
echo [INFO] Startup script: %STARTUP_SCRIPT%

REM 既存タスクの削除（存在する場合）
echo [INFO] Removing existing Advanced Science Pipeline task if present...
schtasks /delete /tn "Advanced_Science_Pipeline_Power_On" /f 2>nul

REM 新規タスク作成
echo [INFO] Creating new Advanced Science Pipeline power-on startup task...
schtasks /create /tn "Advanced_Science_Pipeline_Power_On" ^
    /tr "\"%STARTUP_SCRIPT%\"" ^
    /sc ONLOGON ^
    /rl HIGHEST ^
    /delay 0000:30 ^
    /f

REM タスク作成結果確認
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Task created successfully!
    echo Task Name: Advanced_Science_Pipeline_Power_On
    echo Trigger: At logon (power-on)
    echo Delay: 30 seconds
    echo Run Level: Highest privileges
) else (
    echo [ERROR] Failed to create task!
    goto :error
)

REM タスク確認
echo.
echo [INFO] Verifying task creation...
schtasks /query /tn "Advanced_Science_Pipeline_Power_On" | findstr "Advanced_Science_Pipeline_Power_On"
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Task verification passed!
) else (
    echo [WARNING] Task verification failed!
)

REM 追加設定説明
echo.
echo ========================================
echo Task Scheduler Setup Complete!
echo ========================================
echo.
echo Task Details:
echo - Name: Advanced_Science_Pipeline_Power_On
echo - Trigger: At logon (system startup)
echo - Action: Run %STARTUP_SCRIPT%
echo - Delay: 30 seconds (to ensure system stability)
echo - Privileges: Highest (administrator)
echo.
echo Features:
echo - Automatic checkpoint recovery on startup
echo - 3-minute interval checkpoint saving (max 5)
echo - Full pipeline execution until completion
echo - Audio notification on completion
echo.
echo To modify the task:
echo 1. Open Task Scheduler (taskschd.msc)
echo 2. Navigate to Task Scheduler Library
echo 3. Find and modify "Advanced_Science_Pipeline_Power_On"
echo.
echo ========================================

goto :success

:error
echo ========================================
echo SETUP FAILED!
echo ========================================
echo.
echo Please check:
echo 1. Administrator privileges
echo 2. Script paths are correct
echo 3. Task Scheduler service is running
echo.
pause
exit /b 1

:success
echo Press any key to continue...
pause >nul
