@echo off
REM ================================================================================
REM 統合パイプラインダッシュボード起動スクリプト
REM ================================================================================

setlocal enabledelayedexpansion
chcp 65001 >nul

REM プロジェクトルートの設定
set "PROJECT_ROOT=%~dp0..\.."

echo ================================================================================
echo SO8T統合パイプラインダッシュボード
echo ================================================================================
echo.
echo [INFO] 統合パイプラインとすべてのWebスクレイピングの実行状況・進捗・ブラウジング風景を表示します
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
        !PYTHON_CMD! -m pip install streamlit plotly pandas pillow pyyaml
    ) else (
        "!PYTHON_CMD!" -m pip install streamlit plotly pandas pillow pyyaml
    )
)

echo [INFO] プロジェクトルート: %PROJECT_ROOT%
echo.

REM カレントディレクトリをプロジェクトルートに変更
cd /d "%PROJECT_ROOT%"

REM ダッシュボードスクリプトのパス
set "DASHBOARD_SCRIPT=%PROJECT_ROOT%\scripts\dashboard\unified_pipeline_dashboard.py"

if not exist "%DASHBOARD_SCRIPT%" (
    echo [ERROR] ダッシュボードスクリプトが見つかりません: %DASHBOARD_SCRIPT%
    pause
    exit /b 1
)

echo [INFO] ダッシュボードを起動します...
echo [INFO] ブラウザで http://localhost:8501 を開いてください
echo.

REM Streamlitアプリを起動
if "!PYTHON_CMD:~0,2!"=="py" (
    !PYTHON_CMD! -m streamlit run "%DASHBOARD_SCRIPT%" --server.port 8501 --server.address 0.0.0.0
) else (
    "!PYTHON_CMD!" -m streamlit run "%DASHBOARD_SCRIPT%" --server.port 8501 --server.address 0.0.0.0
)

if errorlevel 1 (
    echo.
    echo [ERROR] ダッシュボード起動中にエラーが発生しました
    pause
    exit /b 1
)

exit /b 0

