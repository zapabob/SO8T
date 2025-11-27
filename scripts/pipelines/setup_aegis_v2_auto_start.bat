@echo off
REM AEGIS v2.0 パイプラインのWindows起動時自動実行設定
REM 管理者権限で実行してください

chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo AEGIS v2.0 Auto-Start Setup
echo ========================================
echo.

REM プロジェクトルートを取得
set "PROJECT_ROOT=%~dp0\..\.."
cd /d "%PROJECT_ROOT%"

REM バッチファイルのフルパス
set "BATCH_FILE=%PROJECT_ROOT%\scripts\pipelines\run_aegis_v2_pipeline.bat"

REM タスク名
set "TASK_NAME=AEGIS_V2_Pipeline_AutoStart"

echo [INFO] Project root: %PROJECT_ROOT%
echo [INFO] Batch file: %BATCH_FILE%
echo [INFO] Task name: %TASK_NAME%
echo.

REM 既存のタスクを削除（存在する場合）
schtasks /query /tn "%TASK_NAME%" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [INFO] Removing existing task...
    schtasks /delete /tn "%TASK_NAME%" /f >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo [OK] Existing task removed
    ) else (
        echo [WARNING] Failed to remove existing task (may require admin rights)
    )
)

REM 新しいタスクを作成（ログオン時実行）
echo [INFO] Creating scheduled task...
schtasks /create /tn "%TASK_NAME%" ^
    /tr "\"%BATCH_FILE%\"" ^
    /sc onlogon ^
    /rl highest ^
    /f >nul 2>&1

if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Scheduled task created successfully!
    echo [INFO] Task will run automatically on Windows startup/login
    echo [INFO] Task name: %TASK_NAME%
    echo.
    echo [INFO] To disable auto-start, run:
    echo   schtasks /delete /tn "%TASK_NAME%" /f
) else (
    echo [ERROR] Failed to create scheduled task
    echo [INFO] This script requires administrator privileges
    echo [INFO] Please run as administrator or create task manually
    echo.
    echo [INFO] Manual task creation command:
    echo   schtasks /create /tn "%TASK_NAME%" /tr "\"%BATCH_FILE%\"" /sc onlogon /rl highest
    exit /b 1
)

echo.
echo [INFO] Setup completed!
echo [INFO] The pipeline will automatically start on next Windows login
echo [INFO] The pipeline will auto-resume from checkpoint if interrupted

REM 音声通知
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

endlocal







