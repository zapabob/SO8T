@echo off
REM SO8T/thinkingモデル再学習スクリプト
REM 電源断対応機能付き（TimeBasedCheckpointCallback + auto-resume）

chcp 65001 >nul
echo [RETRAIN] SO8T/thinking Model Retraining
echo ==========================================
echo.

REM データセットパスの確認
set DATASET=D:\webdataset\processed\thinking_sft\thinking_sft_dataset.jsonl
echo [CHECK] Verifying dataset...
if not exist "%DATASET%" (
    echo [WARNING] Dataset not found: %DATASET%
    echo [INFO] Creating dataset from four_class data...
    
    REM データセット作成スクリプトを実行
    powershell -Command "$files = Get-ChildItem 'D:\webdataset\processed\four_class\four_class_*.jsonl' | Select-Object -ExpandProperty FullName; if ($files) { py -3 scripts\data\create_thinking_sft_dataset.py --inputs $files --output '%DATASET%' } else { Write-Host '[ERROR] No four_class dataset found' -ForegroundColor Red; exit 1 }"
    
    if errorlevel 1 (
        echo [ERROR] Failed to create dataset
        exit /b 1
    )
    echo [OK] Dataset created: %DATASET%
) else (
    echo [OK] Dataset found: %DATASET%
)
echo.

REM 学習モードの選択（fast または full）
set TRAINING_MODE=fast
if not "%1"=="" set TRAINING_MODE=%1

if "%TRAINING_MODE%"=="fast" (
    echo [INFO] Using fast training configuration
    set CONFIG=configs\train_borea_phi35_so8t_thinking_fast.yaml
    set OUTPUT_DIR=D:\webdataset\checkpoints\training\borea_phi35_so8t_thinking_fast
) else if "%TRAINING_MODE%"=="full" (
    echo [INFO] Using full training configuration
    set CONFIG=configs\train_borea_phi35_so8t_thinking.yaml
    set OUTPUT_DIR=D:\webdataset\checkpoints\training\borea_phi35_so8t_thinking
) else (
    echo [ERROR] Invalid training mode: %TRAINING_MODE%
    echo [INFO] Usage: retrain_so8t_thinking.bat [fast^|full]
    exit /b 1
)

REM 設定ファイルの確認
if not exist "%CONFIG%" (
    echo [ERROR] Config file not found: %CONFIG%
    exit /b 1
)

echo [INFO] Config: %CONFIG%
echo [INFO] Output directory: %OUTPUT_DIR%
echo [INFO] Auto-resume: Enabled
echo [INFO] Time-based checkpoint: Every 3 minutes
echo.

REM 再学習実行（--auto-resume付き）
echo [TRAINING] Starting retraining...
py -3 scripts\training\train_borea_phi35_so8t_thinking.py ^
    --config "%CONFIG%" ^
    --dataset "%DATASET%" ^
    --output-dir "%OUTPUT_DIR%" ^
    --auto-resume

if errorlevel 1 (
    echo [ERROR] Training failed with error code %ERRORLEVEL%
    echo [INFO] Check logs: logs\train_borea_phi35_so8t_thinking.log
    exit /b %ERRORLEVEL%
)

echo.
echo [SUCCESS] Retraining completed successfully!
echo [INFO] Model saved to: %OUTPUT_DIR%\final_model
echo.
echo [AUDIO] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"



