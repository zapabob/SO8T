@echo off
chcp 65001 >nul
echo ================================================================================
echo Japanese Training Dataset Collection (Cursor Browser)
echo ================================================================================
echo.

REM プロジェクトルートに移動
cd /d "%~dp0\..\.."

REM Python実行ファイルの検出
set "PYTHON_CMD="
if exist "venv\Scripts\python.exe" (
    set "PYTHON_CMD=venv\Scripts\python.exe"
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

echo.
echo [INFO] Starting Japanese training dataset collection...
echo [INFO] Target samples:
echo   - Wikipedia日本語: 40,000 samples
echo   - CC-100日本語: 30,000 samples
echo   - mc4日本語: 30,000 samples
echo   - Total: 100,000 samples
echo.

REM 日本語学習用データセット収集スクリプトを実行
if "!PYTHON_CMD:~0,2!"=="py" (
    !PYTHON_CMD! scripts\data\collect_japanese_training_dataset.py --output D:/webdataset/japanese_training_dataset --use-mcp-chrome-devtools --num-tabs 10 --delay-per-action 2.0
) else (
    "!PYTHON_CMD!" scripts\data\collect_japanese_training_dataset.py --output D:/webdataset/japanese_training_dataset --use-mcp-chrome-devtools --num-tabs 10 --delay-per-action 2.0
)

if !errorlevel! neq 0 (
    echo [ERROR] Japanese training dataset collection failed
    pause
    exit /b 1
)

echo.
echo [OK] Japanese training dataset collection completed
echo.

REM 音声通知を再生
echo [AUDIO] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

pause





































































































































