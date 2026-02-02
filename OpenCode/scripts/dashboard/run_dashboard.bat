@echo off
chcp 65001 >nul
echo [START] SO8T再学習進捗ダッシュボード起動
echo ====================================================

REM 設定ファイルパス（デフォルト）
set CONFIG_FILE=configs\dashboard_config.yaml

REM 引数で設定ファイルが指定されている場合はそれを使用
if not "%1"=="" set CONFIG_FILE=%1

echo [INFO] Configuration file: %CONFIG_FILE%

REM Streamlitアプリケーション起動
streamlit run scripts\dashboard\so8t_training_dashboard.py --server.port 8501

REM エラーチェック
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Dashboard failed with error code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo [COMPLETE] Dashboard closed








