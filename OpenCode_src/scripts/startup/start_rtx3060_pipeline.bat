@echo off
chcp 65001 >nul
echo [RTX3060] SO8T PPO Pipeline Startup
echo ===================================

echo [STEP 1] Checking CUDA availability...
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"

echo [STEP 2] Setting RTX3060 memory limits...
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo [STEP 3] Starting SO8T PPO training for RTX3060...
echo RTX3060 Specs: 12GB VRAM + 32GB System RAM
echo Memory Limit: 75%% (9GB VRAM)
echo Gradient Checkpointing: Enabled
echo CPU Offload: Enabled

cd /d "%~dp0..\.."
py -3 scripts/training/train_aegis_v2_ppo_so8t.py --config aegis_v2_test_config.json

echo [STEP 4] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

echo [RTX3060] Pipeline completed!
pause
