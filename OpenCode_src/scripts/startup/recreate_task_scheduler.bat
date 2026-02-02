@echo off
REM SO(8)T Task Scheduler Recreate Script
REM 既存タスクを削除して新しい設定で再作成

echo ========================================
echo SO(8)T Task Scheduler Recreate
echo ========================================
echo Recreating power-on auto startup task...
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

echo [INFO] Script directory: %SCRIPT_DIR%
echo [INFO] Startup script: %STARTUP_SCRIPT%

REM 既存タスクの削除（存在する場合）
echo [INFO] Removing existing SO8T task if present...
schtasks /delete /tn "SO8T_Power_On_Startup" /f 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [INFO] Old task removed successfully
) else (
    echo [INFO] No existing task to remove
)

REM 新規タスク作成
echo [INFO] Creating new SO8T power-on startup task...
schtasks /create /tn "SO8T_Power_On_Startup" ^
    /tr "\"%STARTUP_SCRIPT%\"" ^
    /sc ONLOGON ^
    /rl HIGHEST ^
    /delay 0000:30 ^
    /f

REM タスク作成結果確認（即時チェック）
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Task created successfully!
    echo Task Name: SO8T_Power_On_Startup
    echo Trigger: At logon (power-on)
    echo Delay: 30 seconds
    echo Run Level: Highest privileges
    echo.
    echo [INFO] Continuing with verification...
) else (
    echo [ERROR] Failed to create task! Error code: %ERRORLEVEL%
    goto :error
)

REM タスク確認
echo.
echo [INFO] Verifying task creation...
schtasks /query /tn "SO8T_Power_On_Startup" | findstr "SO8T_Power_On_Startup" >nul 2>&1
set VERIFY_RESULT=%ERRORLEVEL%
if %VERIFY_RESULT% EQU 0 (
    echo [SUCCESS] Task verification passed!
    REM タスク詳細表示
    echo.
    echo [INFO] Task details:
    schtasks /query /tn "SO8T_Power_On_Startup" /v /fo list | findstr /C:"TaskName:" /C:"Task To Run:" /C:"Schedule:" /C:"Start Time:"
) else (
    echo [WARNING] Task verification failed! Code: %VERIFY_RESULT%
    echo [INFO] Task may still be created. Checking again...
    schtasks /query /tn "SO8T_Power_On_Startup" >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo [INFO] Task exists despite verification warning.
    ) else (
        echo [ERROR] Task creation verification completely failed!
        goto :error
    )
)

echo.
echo ========================================
echo Task Scheduler Recreate Complete!
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
echo - Python path auto-detection implemented
echo.
goto :success

:error
echo ========================================
echo RECREATE FAILED!
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
