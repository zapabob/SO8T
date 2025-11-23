@echo off
chcp 65001 >nul
echo ================================================================================
echo SO8T統制による完全自動バックグラウンドDeepResearch Webスクレイピング
echo ================================================================================
echo.
echo [INFO] SO8Tモデルの四重推論を使って、完全自動でバックグラウンド実行されます
echo [INFO] ユーザー介入なしで動作します
echo.

REM 設定
set "PROJECT_ROOT=%~dp0..\.."
set "OUTPUT_DIR=D:\webdataset\processed"
set "LOG_DIR=%PROJECT_ROOT%\logs"
set "CONFIG_FILE=%PROJECT_ROOT%\configs\so8t_auto_scraping_config.yaml"

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
echo [INFO] 出力ディレクトリ: %OUTPUT_DIR%
echo [INFO] SO8T統制: 有効
echo.

REM ログディレクトリの作成
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM カレントディレクトリをプロジェクトルートに変更
cd /d "%PROJECT_ROOT%"

echo [INFO] SO8T統制による完全自動バックグラウンドスクレイピングを開始します
echo [INFO] SO8Tモデルの四重推論（Task/Safety/Policy/Final）で動作を統制します
echo [INFO] バックグラウンドで実行され、自動再起動機能が有効です
echo.

REM バックグラウンドで実行
start /B "" "%PYTHON%" -3 scripts\data\so8t_auto_background_scraping.py --output "%OUTPUT_DIR%" --daemon --auto-restart --max-restarts 10 --restart-delay 60.0

if %errorlevel% equ 0 (
    echo [OK] バックグラウンドスクレイピングを開始しました
    echo [INFO] プロセスID: %ERRORLEVEL%
    echo [INFO] ログファイル: %LOG_DIR%\so8t_auto_background_scraping.log
) else (
    echo [NG] バックグラウンドスクレイピングの開始に失敗しました
    exit /b 1
)

echo.
echo ================================================================================
echo [SUCCESS] SO8T統制による完全自動バックグラウンドスクレイピング開始
echo ================================================================================
echo [INFO] バックグラウンドで実行中です
echo [INFO] ログを確認するには: type %LOG_DIR%\so8t_auto_background_scraping.log
echo [INFO] プロセスを停止するには: taskkill /F /IM python.exe /FI "WINDOWTITLE eq *so8t_auto_background_scraping*"
echo.

REM 音声通知
powershell -ExecutionPolicy Bypass -File "%PROJECT_ROOT%\scripts\utils\play_audio_notification.ps1"

exit /b 0





