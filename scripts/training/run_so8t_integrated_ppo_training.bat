@echo off
chcp 65001 >nul
echo [SO8T] Starting SO(8) Integrated PPO Training
echo ============================================

set DATASET_PATH=data\integrated\so8t_integrated_ppo_dataset_main_20251201_205340.jsonl
set MODEL_PATH=models\Borea-Phi-3.5-mini-Instruct-Jp
set CONFIG_PATH=scripts\training\so8t_ppo_config.json

echo [INFO] Dataset: %DATASET_PATH%
echo [INFO] Model: %MODEL_PATH%
echo [INFO] Config: %CONFIG_PATH%

REM GPUメモリチェック
echo [INFO] Checking GPU memory...
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'Current GPU: {torch.cuda.current_device() if torch.cuda.is_available() else \"N/A\"}')" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] GPU check failed, continuing with CPU
)

echo [STEP 1] Starting SO(8) Integrated PPO Training...
echo [INFO] This will take several hours depending on dataset size and hardware

python scripts/training/so8t_integrated_ppo_trainer.py ^
    --model_path "%MODEL_PATH%" ^
    --dataset_path "%DATASET_PATH%" ^
    --config_path "%CONFIG_PATH%"

if %errorlevel% equ 0 (
    echo [SUCCESS] SO(8) Integrated PPO Training completed successfully!
) else (
    echo [ERROR] SO(8) Integrated PPO Training failed with error code %errorlevel%
)

echo [AUDIO] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

echo ============================================
echo [SO8T] SO(8) Integrated PPO Training finished
pause






chcp 65001 >nul
echo [SO8T] Starting SO(8) Integrated PPO Training
echo ============================================

set DATASET_PATH=data\integrated\so8t_integrated_ppo_dataset_main_20251201_205340.jsonl
set MODEL_PATH=models\Borea-Phi-3.5-mini-Instruct-Jp
set CONFIG_PATH=scripts\training\so8t_ppo_config.json

echo [INFO] Dataset: %DATASET_PATH%
echo [INFO] Model: %MODEL_PATH%
echo [INFO] Config: %CONFIG_PATH%

REM GPUメモリチェック
echo [INFO] Checking GPU memory...
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'Current GPU: {torch.cuda.current_device() if torch.cuda.is_available() else \"N/A\"}')" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] GPU check failed, continuing with CPU
)

echo [STEP 1] Starting SO(8) Integrated PPO Training...
echo [INFO] This will take several hours depending on dataset size and hardware

python scripts/training/so8t_integrated_ppo_trainer.py ^
    --model_path "%MODEL_PATH%" ^
    --dataset_path "%DATASET_PATH%" ^
    --config_path "%CONFIG_PATH%"

if %errorlevel% equ 0 (
    echo [SUCCESS] SO(8) Integrated PPO Training completed successfully!
) else (
    echo [ERROR] SO(8) Integrated PPO Training failed with error code %errorlevel%
)

echo [AUDIO] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

echo ============================================
echo [SO8T] SO(8) Integrated PPO Training finished
pause






chcp 65001 >nul
echo [SO8T] Starting SO(8) Integrated PPO Training
echo ============================================

set DATASET_PATH=data\integrated\so8t_integrated_ppo_dataset_main_20251201_205340.jsonl
set MODEL_PATH=models\Borea-Phi-3.5-mini-Instruct-Jp
set CONFIG_PATH=scripts\training\so8t_ppo_config.json

echo [INFO] Dataset: %DATASET_PATH%
echo [INFO] Model: %MODEL_PATH%
echo [INFO] Config: %CONFIG_PATH%

REM GPUメモリチェック
echo [INFO] Checking GPU memory...
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'Current GPU: {torch.cuda.current_device() if torch.cuda.is_available() else \"N/A\"}')" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] GPU check failed, continuing with CPU
)

echo [STEP 1] Starting SO(8) Integrated PPO Training...
echo [INFO] This will take several hours depending on dataset size and hardware

python scripts/training/so8t_integrated_ppo_trainer.py ^
    --model_path "%MODEL_PATH%" ^
    --dataset_path "%DATASET_PATH%" ^
    --config_path "%CONFIG_PATH%"

if %errorlevel% equ 0 (
    echo [SUCCESS] SO(8) Integrated PPO Training completed successfully!
) else (
    echo [ERROR] SO(8) Integrated PPO Training failed with error code %errorlevel%
)

echo [AUDIO] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

echo ============================================
echo [SO8T] SO(8) Integrated PPO Training finished
pause











