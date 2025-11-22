@echo off
chcp 65001 >nul
echo [NSFW-DRUG-DETECTION-QLORA] NSFW・違法薬物検知目的QLoRA学習用データ生成全自動パイプライン
echo ================================================================================
echo [INFO] 新規実行モード（チェックポイントを無視して最初から実行）
echo ================================================================================

set CONFIG_FILE=configs\nsfw_drug_detection_qlora_training_data_pipeline_config.yaml

if not exist "%CONFIG_FILE%" (
    echo [ERROR] 設定ファイルが見つかりません: %CONFIG_FILE%
    pause
    exit /b 1
)

echo [INFO] 設定ファイル: %CONFIG_FILE%
echo [INFO] パイプラインを最初から実行します...
echo [INFO] チェックポイントは無視されます
echo.

REM チェックポイントを無視して新規実行
py -3 scripts\pipelines\nsfw_drug_detection_qlora_training_data_pipeline.py --config %CONFIG_FILE% --no-auto-resume

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


















































































































