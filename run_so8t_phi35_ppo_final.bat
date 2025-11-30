@echo off
chcp 65001 >nul
echo [SO8T] SO8T Phi-3.5 PPO Final Training with SO(8) Residual Adapters
echo ================================================================
echo.

echo [STEP 1] Checking system resources...
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  Device {i}: {torch.cuda.get_device_name(i)}')
        print(f'  Memory: {torch.cuda.get_device_properties(i).total_memory // 1024**3}GB')
else:
    print('CUDA not available, using CPU')
"
echo.

echo [STEP 2] Starting SO8T Phi-3.5 PPO training...
echo Training configuration:
echo - Model: microsoft/Phi-3.5-mini-instruct
echo - Dataset: data/so8t_phi35_tagged
echo - SO(8) Adapter layers: [8, 16, 24]
echo - Max steps: 10000
echo - RTX 3060 optimization enabled
echo.

REM Execute training
py -3 scripts/training/train_so8t_phi35_ppo_final.py --max_steps 10000

echo.
echo [STEP 3] Training completed!
echo.

echo [AUDIO] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

echo.
echo [INFO] SO8T Phi-3.5 PPO training with SO(8) adapters completed!
echo Output directory: D:/webdataset/checkpoints/ppo_so8t_phi35_final
echo.
echo Next steps:
echo 1. Evaluate the trained model
echo 2. Test Phi-3.5 thinking process generation
echo 3. Test SO(8) geometric reasoning capabilities
echo 4. Deploy for inference
echo.
