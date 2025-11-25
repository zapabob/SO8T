@echo off
REM ========================================
REM SO8T リアルタイムシステム監視ダッシュボード起動スクリプト
REM サイバーパンク風Streamlitダッシュボード
REM ========================================

chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo SO8T Real-Time System Monitor Dashboard
echo ========================================
echo.

REM プロジェクトルートパス
set "PROJECT_ROOT=%~dp0.."
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

cd /d "%PROJECT_ROOT%"

REM Python実行ファイルの検出
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

REM Streamlitがインストールされているか確認
if "!PYTHON_CMD:~0,2!"=="py" (
    !PYTHON_CMD! -m streamlit --version >nul 2>&1
) else (
    "!PYTHON_CMD!" -m streamlit --version >nul 2>&1
)

if !errorlevel! neq 0 (
    echo [ERROR] Streamlitがインストールされていません
    echo [INFO] インストール中...
    if "!PYTHON_CMD:~0,2!"=="py" (
        !PYTHON_CMD! -m pip install streamlit plotly pandas psutil pynvml
    ) else (
        "!PYTHON_CMD!" -m pip install streamlit plotly pandas psutil pynvml
    )
)

REM psutilがインストールされているか確認
if "!PYTHON_CMD:~0,2!"=="py" (
    !PYTHON_CMD! -c "import psutil" >nul 2>&1
) else (
    "!PYTHON_CMD!" -c "import psutil" >nul 2>&1
)

if !errorlevel! neq 0 (
    echo [INFO] psutilをインストール中...
    if "!PYTHON_CMD:~0,2!"=="py" (
        !PYTHON_CMD! -m pip install psutil
    ) else (
        "!PYTHON_CMD!" -m pip install psutil
    )
)

REM pynvmlがインストールされているか確認（GPU監視用、オプション）
if "!PYTHON_CMD:~0,2!"=="py" (
    !PYTHON_CMD! -c "import pynvml" >nul 2>&1
) else (
    "!PYTHON_CMD!" -c "import pynvml" >nul 2>&1
)

if !errorlevel! neq 0 (
    echo [INFO] pynvmlをインストール中（GPU監視用）...
    if "!PYTHON_CMD:~0,2!"=="py" (
        !PYTHON_CMD! -m pip install pynvml
    ) else (
        "!PYTHON_CMD!" -m pip install pynvml
    )
)

echo [INFO] Starting Streamlit dashboard...
echo [INFO] Dashboard will be available at: http://localhost:8503
echo [INFO] External access: http://0.0.0.0:8503
echo.

REM Streamlitアプリケーション起動
if "!PYTHON_CMD:~0,2!"=="py" (
    !PYTHON_CMD! -m streamlit run scripts\monitoring\realtime_system_monitor.py --server.port 8503 --server.address 0.0.0.0
) else (
    "!PYTHON_CMD!" -m streamlit run scripts\monitoring\realtime_system_monitor.py --server.port 8503 --server.address 0.0.0.0
)

pause

