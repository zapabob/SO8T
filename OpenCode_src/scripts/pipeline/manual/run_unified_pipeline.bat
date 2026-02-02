@echo off
chcp 65001 >nul
echo ================================================================================
echo SO8T統制Webスクレイピング全自動パイプライン起動
echo ================================================================================
echo.
echo [INFO] これまで作成したすべてのスクレイピングスクリプトを統合して実行します
echo.

REM 設定
set "PROJECT_ROOT=%~dp0..\.."
set "OUTPUT_DIR=D:\webdataset\processed"

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
echo.

REM カレントディレクトリをプロジェクトルートに変更
cd /d "%PROJECT_ROOT%"

echo [INFO] 全自動パイプライン処理を開始します
echo [INFO] バックグラウンドで実行されます
echo.

REM バックグラウンドで実行
start /B "" "%PYTHON%" -3 scripts\pipelines\unified_auto_scraping_pipeline.py --output "%OUTPUT_DIR%" --daemon --auto-restart --max-restarts 10 --restart-delay 3600.0

if %errorlevel% equ 0 (
    echo [OK] 全自動パイプライン処理を開始しました
) else (
    echo [NG] 全自動パイプライン処理の開始に失敗しました
    exit /b 1
)

echo.
echo ================================================================================
echo [SUCCESS] SO8T統制Webスクレイピング全自動パイプライン開始
echo ================================================================================
echo.

REM 音声通知
powershell -ExecutionPolicy Bypass -File "%PROJECT_ROOT%\scripts\utils\play_audio_notification.ps1"

exit /b 0

