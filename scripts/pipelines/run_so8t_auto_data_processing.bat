@echo off
REM ================================================================================
REM SO8T全自動データ処理パイプライン実行スクリプト
REM ================================================================================

setlocal enabledelayedexpansion
chcp 65001 >nul

REM プロジェクトルートの設定
set "PROJECT_ROOT=%~dp0..\.."
set "CONFIG_FILE=%PROJECT_ROOT%\configs\so8t_auto_data_processing_config.yaml"

echo ================================================================================
echo SO8T全自動データ処理パイプライン
echo ================================================================================
echo.
echo [INFO] 収集されたWebスクレイピングデータに対して、SO8Tを使って
echo [INFO] 漸次ラベル付け、データクレンジング、四値分類を全自動で行います
echo.

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
echo [INFO] 設定ファイル: %CONFIG_FILE%
echo.

REM カレントディレクトリをプロジェクトルートに変更
cd /d "%PROJECT_ROOT%"

REM ログディレクトリの作成
if not exist "logs" mkdir "logs"

echo [INFO] パイプラインを開始します...
echo.

REM パイプライン実行
if "!PYTHON_CMD:~0,2!"=="py" (
    !PYTHON_CMD! scripts\pipelines\so8t_auto_data_processing_pipeline.py --config "%CONFIG_FILE%" --resume
) else (
    "!PYTHON_CMD!" scripts\pipelines\so8t_auto_data_processing_pipeline.py --config "%CONFIG_FILE%" --resume
)

if errorlevel 1 (
    echo.
    echo [ERROR] パイプライン実行中にエラーが発生しました
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo [SUCCESS] パイプライン処理が完了しました
echo ================================================================================
echo.

REM 音声通知
powershell -ExecutionPolicy Bypass -File "%PROJECT_ROOT%\scripts\utils\play_audio_notification.ps1"

pause
exit /b 0



































































































































