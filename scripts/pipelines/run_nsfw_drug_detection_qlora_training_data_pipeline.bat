@echo off
chcp 65001 >nul
echo [NSFW-DRUG-DETECTION-QLORA] NSFW・違法薬物検知目的QLoRA学習用データ生成全自動パイプライン
echo ================================================================================

set CONFIG_FILE=configs\nsfw_drug_detection_qlora_training_data_pipeline_config.yaml

if not exist "%CONFIG_FILE%" (
    echo [ERROR] 設定ファイルが見つかりません: %CONFIG_FILE%
    pause
    exit /b 1
)

echo [INFO] 設定ファイル: %CONFIG_FILE%
echo [INFO] パイプラインを開始します...
echo [INFO] 電源断からの自動再開機能: 有効
echo.

REM チェックポイントの存在を確認
set CHECKPOINT_DIR=D:\webdataset\checkpoints\nsfw_drug_detection_qlora_training_data_pipeline
if exist "%CHECKPOINT_DIR%\checkpoint_*.pkl" (
    echo [RESUME] チェックポイントを検出しました。自動再開します...
    echo.
)

REM チェックポイントを無視して新規実行する場合は--no-auto-resumeオプションを追加
REM py -3 scripts\pipelines\nsfw_drug_detection_qlora_training_data_pipeline.py --config %CONFIG_FILE% --no-auto-resume

REM 通常実行（チェックポイントから自動再開）
py -3 scripts\pipelines\nsfw_drug_detection_qlora_training_data_pipeline.py --config %CONFIG_FILE%

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] パイプラインが失敗しました
    echo [INFO] チェックポイントが保存されているため、次回実行時に自動再開されます
    pause
    exit /b 1
)

echo [SUCCESS] パイプラインが正常に完了しました
echo [AUDIO] 音声通知を再生中...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

pause


