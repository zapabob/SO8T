@echo off
chcp 65001 >nul
echo ================================================================================
echo SO8T統制Webスクレイピング集中管理ダッシュボード起動
echo ================================================================================
echo.

REM 設定
set "PROJECT_ROOT=%~dp0..\.."

REM Pythonパスの確認
set "PYTHON="
if exist "%PROJECT_ROOT%\venv\Scripts\python.exe" (
    set "PYTHON=%PROJECT_ROOT%\venv\Scripts\python.exe"
) else (
    where py >nul 2>&1
    if %errorlevel% equ 0 (
        set "PYTHON=py"
    ) else (
        echo [ERROR] Pythonが見つかりません
        exit /b 1
    )
)

echo [INFO] Python実行ファイル: %PYTHON%
echo [INFO] プロジェクトルート: %PROJECT_ROOT%
echo.

REM カレントディレクトリをプロジェクトルートに変更
cd /d "%PROJECT_ROOT%"

echo [INFO] Streamlitダッシュボードを起動します
echo [INFO] ブラウザが自動的に開きます
echo.

REM Streamlitダッシュボードを起動
"%PYTHON%" -m streamlit run scripts\dashboard\so8t_scraping_dashboard.py --server.port 8501 --server.address 0.0.0.0

exit /b 0





