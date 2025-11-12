@echo off
chcp 65001 >nul
echo [START] SO(8) Transformer再学習統合スクリプト実行
echo ====================================================

REM 設定ファイルパス（デフォルト）
set CONFIG_FILE=configs\so8t_borea_phi35_bayesian_recovery_config.yaml

REM 引数で設定ファイルが指定されている場合はそれを使用
if not "%1"=="" set CONFIG_FILE=%1

echo [INFO] Configuration file: %CONFIG_FILE%

REM Pythonスクリプト実行
py -3 scripts\training\train_so8t_borea_phi35_bayesian_recovery.py --config %CONFIG_FILE%

REM エラーチェック
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Training failed with error code %ERRORLEVEL%
    echo [AUDIO] Playing completion notification...
    powershell -ExecutionPolicy Bypass -File "%~dp0..\utils\play_audio.ps1"
    exit /b %ERRORLEVEL%
)

echo [COMPLETE] Training completed successfully
echo [AUDIO] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "%~dp0..\utils\play_audio.ps1"

