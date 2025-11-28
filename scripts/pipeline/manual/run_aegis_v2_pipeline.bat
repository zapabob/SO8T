@echo off
REM AEGIS v2.0 全自動パイプライン実行スクリプト（電源断リカバリー機能付き）
REM 電源投入時に自動的にチェックポイントから再開

chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo AEGIS v2.0 Automated Pipeline
echo Power Failure Recovery Enabled
echo ========================================
echo.

REM プロジェクトルートに移動
cd /d "%~dp0\..\.."

REM 環境変数設定
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"

REM プロンプトファイル（デフォルト）
set "PROMPTS_FILE=data\prompts\thinking_prompts.txt"
if not exist "%PROMPTS_FILE%" (
    echo [WARNING] Prompts file not found: %PROMPTS_FILE%
    echo [INFO] Creating default prompts file...
    mkdir "data\prompts" 2>nul
    echo 日本の首都はどこですか？ > "%PROMPTS_FILE%"
    echo 人工知能の未来について説明してください。 >> "%PROMPTS_FILE%"
    echo 機械学習の基本的な概念を説明してください。 >> "%PROMPTS_FILE%"
)

REM 設定ファイル（デフォルト）
set "CONFIG_FILE=configs\train_borea_phi35_so8t_thinking_frozen.yaml"
if not exist "%CONFIG_FILE%" (
    echo [ERROR] Config file not found: %CONFIG_FILE%
    echo [INFO] Please create the config file or specify with --config
    exit /b 1
)

REM 出力ディレクトリ
set "OUTPUT_DIR=D:\webdataset\aegis_v2.0"

REM APIキー確認
if "%OPENAI_API_KEY%"=="" (
    echo [WARNING] OPENAI_API_KEY not set
)
if "%GEMINI_API_KEY%"=="" (
    echo [WARNING] GEMINI_API_KEY not set
)

echo [INFO] Starting AEGIS v2.0 pipeline...
echo [INFO] Prompts file: %PROMPTS_FILE%
echo [INFO] Config file: %CONFIG_FILE%
echo [INFO] Output directory: %OUTPUT_DIR%
echo [INFO] Auto-resume: Enabled
echo.

REM パイプライン実行
py -3 scripts\pipelines\aegis_v2_automated_pipeline.py ^
    --prompts-file "%PROMPTS_FILE%" ^
    --config "%CONFIG_FILE%" ^
    --output-dir "%OUTPUT_DIR%" ^
    --api-type openai

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Pipeline failed with exit code %ERRORLEVEL%
    echo [INFO] Check logs/aegis_v2_pipeline.log for details
    echo [INFO] Pipeline will auto-resume from checkpoint on next run
    powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"
    exit /b %ERRORLEVEL%
)

echo.
echo [SUCCESS] AEGIS v2.0 pipeline completed successfully!
echo [INFO] AEGIS v2.0 directory: %OUTPUT_DIR%
echo.

REM 音声通知
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

endlocal


