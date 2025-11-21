@echo off
REM ================================================================================
REM SO8T完全統合A/Bテストパイプライン実行スクリプト
REM ================================================================================

setlocal enabledelayedexpansion
chcp 65001 >nul

REM プロジェクトルートの設定
set "PROJECT_ROOT=%~dp0..\.."
set "CONFIG_FILE=%PROJECT_ROOT%\configs\complete_so8t_ab_test_pipeline_config.yaml"

echo ================================================================================
echo SO8T完全統合A/Bテストパイプライン
echo ================================================================================
echo.
echo [INFO] Borea-Phi-3.5-mini-Instruct-Jpのmodeling_phi3_so8t.pyをそのままGGUF化したもの（Model A）と
echo [INFO] SO8Tで再学習してデータセットでQLoRA/ファインチューニングで学習させたものをGGUF化したもの（Model B）の
echo [INFO] A/Bテストまで一気通貫で全自動実行します
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
echo [INFO] この処理には長時間かかる場合があります（数時間から数日）
echo.

REM パイプライン実行
if "!PYTHON_CMD:~0,2!"=="py" (
    !PYTHON_CMD! scripts\pipelines\complete_so8t_ab_test_pipeline.py --config "%CONFIG_FILE%" --resume
) else (
    "!PYTHON_CMD!" scripts\pipelines\complete_so8t_ab_test_pipeline.py --config "%CONFIG_FILE%" --resume
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


































































































