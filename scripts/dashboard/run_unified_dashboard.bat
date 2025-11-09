@echo off
chcp 65001 >nul
echo ================================================================================
echo SO8T統制Webスクレイピング統一管理ダッシュボード起動
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

echo [INFO] Streamlit統一管理ダッシュボードを起動します
echo [INFO] ブラウザが自動的に開きます
echo.

REM Streamlitダッシュボードを起動
"%PYTHON%" -m streamlit run scripts\dashboard\unified_scraping_dashboard.py --server.port 8502 --server.address 0.0.0.0

exit /b 0

