@echo off
REM ================================================================================
REM SO8T統合マスターパイプラインセットアップスクリプト
REM ================================================================================

setlocal enabledelayedexpansion
chcp 65001 >nul

REM プロジェクトルートの設定
set "PROJECT_ROOT=%~dp0..\.."

echo ================================================================================
echo SO8T統合マスターパイプラインセットアップ
echo ================================================================================
echo.
echo [INFO] すべての全自動パイプラインを統合して、電源投入時に自動実行できるようにします
echo.

REM 管理者権限チェック
net session >nul 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] 管理者権限が必要です
    echo [ERROR] このスクリプトを右クリックして「管理者として実行」を選択してください
    pause
    exit /b 1
)

echo [OK] 管理者権限を確認しました
echo.

REM カレントディレクトリをプロジェクトルートに変更
cd /d "%PROJECT_ROOT%"

REM Pythonパスの確認
set "PYTHON_CMD="
if exist "%PROJECT_ROOT%\venv\Scripts\python.exe" (
    set "PYTHON_CMD=%PROJECT_ROOT%\venv\Scripts\python.exe"
    echo [INFO] Python実行ファイル: !PYTHON_CMD!
) else (
    REM py launcherを確認
    py -3 --version >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_CMD=py -3"
        echo [INFO] Python実行ファイル: py -3
    ) else (
        REM pythonコマンドを確認
        python --version >nul 2>&1
        if !errorlevel! equ 0 (
            set "PYTHON_CMD=python"
            echo [INFO] Python実行ファイル: python
        ) else (
            echo [ERROR] Pythonが見つかりません
            echo [ERROR] venv\Scripts\python.exe、py -3、またはpythonコマンドが必要です
            pause
            exit /b 1
        )
    )
)

echo [INFO] プロジェクトルート: %PROJECT_ROOT%
echo.

REM 設定ファイルの確認
set "CONFIG_FILE=%PROJECT_ROOT%\configs\unified_master_pipeline_config.yaml"
if not exist "%CONFIG_FILE%" (
    echo [ERROR] 設定ファイルが見つかりません: %CONFIG_FILE%
    pause
    exit /b 1
)

echo [OK] 設定ファイルを確認しました: %CONFIG_FILE%
echo.

REM バッチファイルの確認
set "BATCH_FILE=%PROJECT_ROOT%\scripts\pipelines\unified_master_pipeline_autostart.bat"
if not exist "%BATCH_FILE%" (
    echo [ERROR] バッチファイルが見つかりません: %BATCH_FILE%
    pause
    exit /b 1
)

echo [OK] バッチファイルを確認しました: %BATCH_FILE%
echo.

echo [INFO] タスクスケジューラに登録します...
echo.

REM パイプラインスクリプトを実行してセットアップ
if "!PYTHON_CMD:~0,2!"=="py" (
    !PYTHON_CMD! scripts\pipelines\unified_master_pipeline.py --setup --config "%CONFIG_FILE%"
) else (
    "!PYTHON_CMD!" scripts\pipelines\unified_master_pipeline.py --setup --config "%CONFIG_FILE%"
)

if errorlevel 1 (
    echo.
    echo [ERROR] タスクスケジューラ登録に失敗しました
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo [SUCCESS] セットアップが完了しました
echo ================================================================================
echo.
echo [INFO] 電源投入時に自動実行されるように設定されました
echo [INFO] タスク名: SO8T-UnifiedMasterPipeline-AutoStart
echo [INFO] トリガー: システム起動時
echo.

REM 音声通知
powershell -ExecutionPolicy Bypass -File "%PROJECT_ROOT%\scripts\utils\play_audio_notification.ps1"

pause
exit /b 0

