@echo off
chcp 65001 >nul
echo ================================================================================
echo SO8T 完全バックグラウンド並列DeepResearch Webスクレイピングパイプラインマネージャー停止
echo ================================================================================
echo.

REM 設定
set "PROJECT_ROOT=%~dp0..\.."
set "LOG_DIR=%PROJECT_ROOT%\logs"

echo [INFO] 並列パイプラインマネージャーとすべてのインスタンスを停止します
echo.

REM 遅延環境変数展開を有効化
setlocal enabledelayedexpansion

REM PIDファイルからプロセスを停止
for /L %%i in (0,1,9) do (
    set "PID_FILE=%LOG_DIR%\parallel_instance_%%i.pid"
    if exist "!PID_FILE!" (
        echo [INFO] インスタンス %%i を停止中...
        for /f "usebackq tokens=*" %%p in ("!PID_FILE!") do (
            taskkill /F /PID %%p >nul 2>&1
            if !errorlevel! equ 0 (
                echo [OK] インスタンス %%i 停止完了 (PID: %%p)
            ) else (
                echo [WARNING] インスタンス %%i の停止に失敗 (PID: %%p)
            )
        )
        del "!PID_FILE!" >nul 2>&1
    )
)

REM マネージャープロセスを停止
echo [INFO] マネージャープロセスを停止中...
taskkill /F /FI "WINDOWTITLE eq *parallel_pipeline_manager*" >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] マネージャープロセス停止完了
) else (
    echo [WARNING] マネージャープロセスの停止に失敗（既に停止している可能性があります）
)

echo.
echo ================================================================================
echo [SUCCESS] すべてのインスタンスとマネージャーを停止しました
echo ================================================================================
echo.

REM 音声通知
powershell -ExecutionPolicy Bypass -File "%PROJECT_ROOT%\scripts\utils\play_audio_notification.ps1"

exit /b 0

