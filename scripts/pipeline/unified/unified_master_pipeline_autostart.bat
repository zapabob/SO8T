@echo off
REM ================================================================================
REM SO8T統合マスターパイプライン
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
set "SCRIPT_PATH=%PROJECT_ROOT%\scripts\pipelines\unified_master_pipeline.py"
set "CONFIG_PATH=%PROJECT_ROOT%\configs\unified_master_pipeline_config.yaml"

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
    cd /d "%PROJECT_ROOT%" && !PYTHON_CMD! "%SCRIPT_PATH%" --run --config "%CONFIG_PATH%" --resume
) else (
    cd /d "%PROJECT_ROOT%" && "!PYTHON_CMD!" "%SCRIPT_PATH%" --run --config "%CONFIG_PATH%" --resume
)

exit /b %ERRORLEVEL%

