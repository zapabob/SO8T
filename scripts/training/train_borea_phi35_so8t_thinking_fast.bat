@echo off
chcp 65001 >nul
REM 高速版訓練スクリプト（訓練時間短縮）

echo [FAST TRAINING] Starting optimized training...
echo ================================================

REM 高速設定ファイルを使用
set CONFIG=configs\train_borea_phi35_so8t_thinking_fast.yaml
set DATASET=D:\webdataset\processed\thinking_sft\thinking_sft_dataset.jsonl
set OUTPUT_DIR=D:\webdataset\checkpoints\training\borea_phi35_so8t_thinking_fast

echo [INFO] Config: %CONFIG%
echo [INFO] Dataset: %DATASET%
echo [INFO] Output: %OUTPUT_DIR%
echo.

REM 訓練実行（自動再開機能付き）
py -3 scripts\training\train_borea_phi35_so8t_thinking.py ^
    --config %CONFIG% ^
    --dataset "%DATASET%" ^
    --output-dir "%OUTPUT_DIR%" ^
    --auto-resume

echo.
echo [AUDIO] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"






































































































