@echo off
REM SO(8)T Task Scheduler Setup Script
REM タスクスケジューラに電源投入時自動起動を設定

echo ========================================
echo SO(8)T Task Scheduler Setup
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
set STARTUP_SCRIPT=%SCRIPT_DIR%\scripts\startup\so8t_power_on_startup.bat

REM ログディレクトリ作成
if not exist "logs\startup" mkdir "logs\startup"

echo [INFO] Script directory: %SCRIPT_DIR%
echo [INFO] Startup script: %STARTUP_SCRIPT%

REM 既存タスクの削除（存在する場合）
echo [INFO] Removing existing SO8T task if present...
schtasks /delete /tn "SO8T_Power_On_Startup" /f 2>nul

REM 新規タスク作成
echo [INFO] Creating new SO8T power-on startup task...
schtasks /create /tn "SO8T_Power_On_Startup" ^
    /tr "\"%STARTUP_SCRIPT%\"" ^
    /sc ONLOGON ^
    /rl HIGHEST ^
    /delay 0000:30 ^
    /f

REM タスク作成結果確認
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Task created successfully!
    echo Task Name: SO8T_Power_On_Startup
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
schtasks /query /tn "SO8T_Power_On_Startup" | findstr "SO8T_Power_On_Startup"
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
echo - Name: SO8T_Power_On_Startup
echo - Trigger: At logon (system startup)
echo - Action: Run %STARTUP_SCRIPT%
echo - Delay: 30 seconds (to ensure system stability)
echo - Privileges: Highest (administrator)
echo.
echo Additional Notes:
echo - Task runs when ANY user logs on
echo - 30-second delay prevents startup conflicts
echo - Logs are saved to logs\startup\
echo - Task can be managed via Task Scheduler GUI
echo.
echo To modify the task:
echo 1. Open Task Scheduler (taskschd.msc)
echo 2. Navigate to Task Scheduler Library
echo 3. Find and modify "SO8T_Power_On_Startup"
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
echo Manual setup alternative:
echo 1. Open Task Scheduler (taskschd.msc)
echo 2. Create new task manually
echo 3. Set trigger to "At logon"
echo 4. Set action to run: %STARTUP_SCRIPT%
echo.
pause
exit /b 1

:success
echo Press any key to continue...
pause >nul
