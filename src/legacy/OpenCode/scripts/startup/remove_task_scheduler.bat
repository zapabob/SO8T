@echo off
REM SO(8)T Task Scheduler Removal Script
REM タスクスケジューラからSO8T自動起動を削除

echo ========================================
echo SO(8)T Task Scheduler Removal
echo ========================================
echo Removing SO8T power-on auto startup...
echo ========================================

REM 管理者権限チェック
net session >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Administrator privileges required!
    echo Please run this script as Administrator.
    pause
    exit /b 1
)

REM タスク削除
echo [INFO] Deleting SO8T task from Task Scheduler...
schtasks /delete /tn "SO8T_Power_On_Startup" /f

if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Task deleted successfully!
    echo Task "SO8T_Power_On_Startup" has been removed.
) else (
    echo [WARNING] Task may not exist or deletion failed.
    echo This is normal if the task was already removed.
)

REM タスク確認
echo.
echo [INFO] Verifying task removal...
schtasks /query /tn "SO8T_Power_On_Startup" 2>nul | findstr "SO8T_Power_On_Startup"
if %ERRORLEVEL% NEQ 0 (
    echo [SUCCESS] Task removal verified!
) else (
    echo [WARNING] Task may still exist.
)

REM ログファイル削除オプション
echo.
echo [INFO] Startup log files location:
echo logs\startup\
echo.
set /p choice="Delete startup log files? (y/N): "
if /i "%choice%"=="y" (
    if exist "logs\startup" (
        rmdir /s /q "logs\startup"
        echo [INFO] Startup log files deleted.
    ) else (
        echo [INFO] No startup log files found.
    )
) else (
    echo [INFO] Startup log files preserved.
)

echo.
echo ========================================
echo Task Scheduler Removal Complete!
echo ========================================
echo.
echo The SO8T power-on startup task has been removed.
echo The system will no longer auto-start SO8T on login.
echo.
echo To re-enable, run: setup_task_scheduler.bat
echo.

pause
