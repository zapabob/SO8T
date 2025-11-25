@echo off
chcp 65001 >nul
echo [HYPERPARAMETER OPTIMIZATION] Starting Bayesian optimization with cross-validation...
echo ========================================

set CONFIG_FILE=configs\train_borea_phi35_so8t_thinking_frozen.yaml
set MODEL_PATH=AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp
set DATASET_PATH=D:/webdataset/processed/thinking_sft/thinking_sft_dataset.jsonl
set OUTPUT_DIR=D:/webdataset/hyperparameter_optimization

echo [CONFIG] Config file: %CONFIG_FILE%
echo [MODEL] Model path: %MODEL_PATH%
echo [DATASET] Dataset path: %DATASET_PATH%
echo [OUTPUT] Output directory: %OUTPUT_DIR%
echo.

echo [STEP 1] Starting hyperparameter optimization...
py -3 scripts\training\hyperparameter_optimization_with_cv.py ^
    --config %CONFIG_FILE% ^
    --model-path %MODEL_PATH% ^
    --dataset %DATASET_PATH% ^
    --output-dir %OUTPUT_DIR% ^
    --n-trials 50 ^
    --n-folds 5

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Hyperparameter optimization failed
    exit /b 1
)

echo.
echo [SUCCESS] Hyperparameter optimization completed
echo [AUDIO] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"



