@echo off
REM ========================================
REM SO8T完全自動化マスターパイプライン監視ダッシュボード起動スクリプト
REM サイバーパンク風Streamlitダッシュボード
REM ========================================

chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo SO8T Pipeline Monitor Dashboard
echo ========================================
echo.

REM プロジェクトルートパス
set "PROJECT_ROOT=%~dp0.."
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

cd /d "%PROJECT_ROOT%"

REM Python実行ファイルの検出
where py >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH
    echo [ERROR] Please install Python or add it to PATH
    pause
    exit /b 1
)

echo [INFO] Starting Streamlit dashboard...
echo [INFO] Dashboard will be available at: http://localhost:8502
echo.

REM Streamlitアプリケーション起動
streamlit run scripts\monitoring\streamlit_dashboard.py --server.port 8502

pause

