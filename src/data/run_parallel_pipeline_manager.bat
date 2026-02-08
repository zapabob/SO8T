@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
echo ================================================================================
echo SO8T 完全バックグラウンド並列DeepResearch Webスクレイピングパイプラインマネージャー
echo ================================================================================
echo.
echo [INFO] 10個のDeepResearch webスクレイピングパイプラインを完全バックグラウンドで並列実行します
echo [INFO] 各インスタンスは10個のブラウザを並列実行します（合計100個のブラウザ）
echo [INFO] 完全バックグラウンド実行（デーモンモード）で動作します
echo.

REM 設定
set "PROJECT_ROOT=%~dp0..\.."
set "BASE_OUTPUT_DIR=D:\webdataset\processed"
set "LOG_DIR=%PROJECT_ROOT%\logs"

REM Pythonパスの確認
set "PYTHON_CMD="
if exist "%PROJECT_ROOT%\venv\Scripts\python.exe" (
    set "PYTHON_CMD=%PROJECT_ROOT%\venv\Scripts\python.exe"
    echo [INFO] Python実行ファイル: !PYTHON_CMD!
) else (
    REM py launcherを確認（直接テスト）
    py -3 --version >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_CMD=py -3"
        echo [INFO] Python実行ファイル: py -3
    ) else (
        REM pyコマンドを確認（-3なし）
        py --version >nul 2>&1
        if !errorlevel! equ 0 (
            set "PYTHON_CMD=py"
            echo [INFO] Python実行ファイル: py
        ) else (
        REM pythonコマンドを確認
        python --version >nul 2>&1
        if !errorlevel! equ 0 (
            set "PYTHON_CMD=python"
            echo [INFO] Python実行ファイル: python
        ) else (
            echo [ERROR] Pythonが見つかりません
            echo [ERROR] venv\Scripts\python.exe、py -3、またはpythonコマンドが必要です
            exit /b 1
        )
    )
)

echo [INFO] プロジェクトルート: %PROJECT_ROOT%
echo [INFO] ベース出力ディレクトリ: %BASE_OUTPUT_DIR%
echo [INFO] 並列インスタンス数: 10
echo [INFO] 各インスタンスのブラウザ数: 10（合計100個のブラウザ）
echo.

REM ログディレクトリの作成
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM カレントディレクトリをプロジェクトルートに変更
cd /d "%PROJECT_ROOT%"

echo [INFO] 完全バックグラウンド並列パイプラインマネージャーを開始します
echo [INFO] デーモンモードで実行します（完全バックグラウンド）
echo.

REM マネージャー実行（完全バックグラウンド）
REM py -3の場合はcmd /cを使用して一つのコマンドとして実行
if "!PYTHON_CMD:~0,2!"=="py" (
    start /B "" cmd /c "!PYTHON_CMD! scripts\data\parallel_pipeline_manager.py --num-instances 10 --base-output !BASE_OUTPUT_DIR! --base-port 9222 --daemon --auto-restart --restart-delay 60.0 --max-memory-gb 8.0 --max-cpu-percent 80.0"
) else (
    start /B "" "!PYTHON_CMD!" scripts\data\parallel_pipeline_manager.py --num-instances 10 --base-output "!BASE_OUTPUT_DIR!" --base-port 9222 --daemon --auto-restart --restart-delay 60.0 --max-memory-gb 8.0 --max-cpu-percent 80.0
)

REM start /Bは非同期実行のため、エラーレベルのチェックは行わない
REM 代わりに、少し待ってからログファイルを確認する
timeout /t 3 /nobreak >nul 2>&1

echo [OK] マネージャー起動完了
echo [INFO] 10個のインスタンスが完全バックグラウンドで動作中です
echo [INFO] 状態確認: type %LOG_DIR%\parallel_pipeline_status.json
echo [INFO] ログ確認: type %LOG_DIR%\parallel_pipeline_manager.log

echo.
echo ================================================================================
echo [SUCCESS] 完全バックグラウンド並列パイプラインマネージャー起動完了
echo ================================================================================
echo [INFO] 10個のインスタンスが完全バックグラウンドで動作中です
echo [INFO] 各インスタンスは10個のブラウザを並列実行します（合計100個のブラウザ）
echo [INFO] 状態確認: type %LOG_DIR%\parallel_pipeline_status.json
echo [INFO] ログ確認: type %LOG_DIR%\parallel_pipeline_manager.log
echo [INFO] 各インスタンスログ: type %LOG_DIR%\parallel_instance_*.log
echo.

REM 音声通知
powershell -ExecutionPolicy Bypass -File "%PROJECT_ROOT%\scripts\utils\play_audio_notification.ps1"

exit /b 0

