@echo off
REM ================================================================================
REM SO8T 完全バックグラウンド並列DeepResearch Webスクレイピングパイプラインマネージャー
REM タスクスケジューラ用自動起動バッチファイル
REM ================================================================================

setlocal enabledelayedexpansion
chcp 65001 >nul

REM バッチファイルのディレクトリからプロジェクトルートを取得（絶対パス）
set "BATCH_DIR=%~dp0"
set "PROJECT_ROOT=%BATCH_DIR%..\.."

REM パスを正規化（..\..を解決）
for %%I in ("%PROJECT_ROOT%") do set "PROJECT_ROOT=%%~fI"

REM カレントディレクトリをプロジェクトルートに変更（絶対パスで確実に）
cd /d "%PROJECT_ROOT%"

REM Pythonスクリプトの絶対パス
set "SCRIPT_PATH=%PROJECT_ROOT%\scripts\data\parallel_pipeline_manager.py"

REM Pythonパスの確認
set "PYTHON_CMD="
if exist "%PROJECT_ROOT%\venv\Scripts\python.exe" (
    set "PYTHON_CMD=%PROJECT_ROOT%\venv\Scripts\python.exe"
) else (
    REM py launcherを確認
    py -3 --version >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_CMD=py -3"
    ) else (
        REM pythonコマンドを確認
        python --version >nul 2>&1
        if !errorlevel! equ 0 (
            set "PYTHON_CMD=python"
        ) else (
            echo [ERROR] Pythonが見つかりません
            exit /b 1
        )
    )
)

REM タスクスケジューラから呼び出される場合の実行
REM --run引数でタスクスケジューラから呼び出されたことを示す
REM システム起動時の遅延処理はPythonスクリプト内で実装
REM 作業ディレクトリをプロジェクトルートに設定してから実行

REM 作業ディレクトリをプロジェクトルートに設定
cd /d "%PROJECT_ROOT%"

REM 絶対パスでPythonスクリプトを実行
if "!PYTHON_CMD:~0,2!"=="py" (
    cd /d "%PROJECT_ROOT%" && !PYTHON_CMD! "%SCRIPT_PATH%" --run --daemon --num-instances 10 --base-output D:/webdataset/processed --base-port 9222 --auto-restart --restart-delay 60.0 --max-memory-gb 8.0 --max-cpu-percent 80.0
) else (
    cd /d "%PROJECT_ROOT%" && "!PYTHON_CMD!" "%SCRIPT_PATH%" --run --daemon --num-instances 10 --base-output D:/webdataset/processed --base-port 9222 --auto-restart --restart-delay 60.0 --max-memory-gb 8.0 --max-cpu-percent 80.0
)

exit /b %ERRORLEVEL%

