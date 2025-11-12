@echo off
REM SO8T Complete Pipeline実行スクリプト（Windows）
REM 日本語ドメイン別知識とコーディング能力向上を狙った完全自動化パイプライン

chcp 65001 >nul
setlocal enabledelayedexpansion

echo ==========================================
echo SO8T Complete Pipeline
echo 日本語ドメイン別知識 + コーディング能力向上
echo ==========================================
echo.

REM デフォルト設定
set "BASE_MODEL=models/Borea-Phi-3.5-mini-Instruct-Jp"
set "JAPANESE_DATASET=D:/webdataset/japanese_training_dataset/train.jsonl"
set "CODING_DATASET=D:/webdataset/coding_dataset/train.jsonl"
set "THINKING_DATASET=data/processed/thinking/thinking_20251108_013450.jsonl"
set "FOUR_CLASS_DATASET=data/so8t_safety_dataset.jsonl"
set "VAL_DATA=D:/webdataset/japanese_training_dataset/val.jsonl"
set "OUTPUT_BASE=D:/webdataset/so8t_complete"
set "QUANTIZATION=Q5_K_M"

REM 引数解析
if not "%~1"=="" set "BASE_MODEL=%~1"
if not "%~2"=="" set "JAPANESE_DATASET=%~2"
if not "%~3"=="" set "CODING_DATASET=%~3"
if not "%~4"=="" set "THINKING_DATASET=%~4"
if not "%~5"=="" set "FOUR_CLASS_DATASET=%~5"
if not "%~6"=="" set "OUTPUT_BASE=%~6"

echo [CONFIG] Base model: !BASE_MODEL!
echo [CONFIG] Japanese dataset: !JAPANESE_DATASET!
echo [CONFIG] Coding dataset: !CODING_DATASET!
echo [CONFIG] Thinking dataset: !THINKING_DATASET!
echo [CONFIG] Four-class dataset: !FOUR_CLASS_DATASET!
echo [CONFIG] Output base: !OUTPUT_BASE!
echo [CONFIG] Quantization: !QUANTIZATION!
echo.

REM パイプライン実行
py -3 scripts/training/run_complete_pipeline.py ^
    --base_model "!BASE_MODEL!" ^
    --japanese_dataset "!JAPANESE_DATASET!" ^
    --coding_dataset "!CODING_DATASET!" ^
    --thinking_dataset "!THINKING_DATASET!" ^
    --four_class_dataset "!FOUR_CLASS_DATASET!" ^
    --val_data "!VAL_DATA!" ^
    --output_base "!OUTPUT_BASE!" ^
    --quantization "!QUANTIZATION!"

if errorlevel 1 (
    echo.
    echo [ERROR] Pipeline execution failed!
    exit /b 1
)

echo.
echo [OK] Pipeline execution completed successfully!
echo.
echo Next steps:
echo   1. Test inference: llama.cpp/main.exe -m !OUTPUT_BASE!/gguf_models/*_!QUANTIZATION!.gguf -n 1024 -t 8 --temp 0.7
echo   2. Check calibration results: !OUTPUT_BASE!/calibration/calibration_results.json
echo.

REM 音声通知
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"


