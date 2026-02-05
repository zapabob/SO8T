@echo off
chcp 65001 >nul
echo [PIPELINE] SO8T統制Webスクレイピング全自動パイプライン
echo ============================================================

cd /d "%~dp0\..\.."

echo [INFO] Current directory: %CD%
echo [INFO] Starting unified auto scraping pipeline...

REM ログディレクトリを作成
if not exist "logs" mkdir logs

REM パイプラインを実行
py -3 scripts\pipelines\unified_auto_scraping_pipeline.py ^
    --output D:\webdataset\processed ^
    --config configs\unified_auto_scraping_pipeline_config.yaml ^
    --daemon ^
    --auto-restart ^
    --max-restarts 10 ^
    --restart-delay 3600.0

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Pipeline execution failed with error code: %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo [OK] Pipeline execution completed
echo [AUDIO] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"





