@echo off
chcp 65001 >nul
REM 電源投入時に自動的に訓練を再開するバッチファイル

set CONFIG_PATH=configs\train_borea_phi35_so8t_thinking.yaml
set OUTPUT_DIR=D:\webdataset\checkpoints\training\borea_phi35_so8t_thinking
set SESSION_FILE=%OUTPUT_DIR%\training_session.json

echo [AUTO-RESUME] Checking for training sessions...

if exist "%SESSION_FILE%" (
    echo [AUTO-RESUME] Found session file: %SESSION_FILE%
    echo [AUTO-RESUME] Resuming training...
    
    REM 訓練スクリプトを自動再開モードで実行
    py -3 scripts\training\train_borea_phi35_so8t_thinking.py ^
        --config %CONFIG_PATH% ^
        --dataset "D:\webdataset\processed\thinking_sft\thinking_sft_dataset.jsonl" ^
        --output-dir %OUTPUT_DIR% ^
        --auto-resume
) else (
    echo [AUTO-RESUME] No session file found. Starting new training...
    
    REM 新しい訓練を開始
    py -3 scripts\training\train_borea_phi35_so8t_thinking.py ^
        --config %CONFIG_PATH% ^
        --dataset "D:\webdataset\processed\thinking_sft\thinking_sft_dataset.jsonl" ^
        --output-dir %OUTPUT_DIR%
)

echo [AUDIO] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

