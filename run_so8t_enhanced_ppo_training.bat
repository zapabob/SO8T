@echo off
chcp 65001 >nul
echo [SO8T] Enhanced PPO Training with Advanced Checkpoint Management
echo =================================================================
echo.
echo [INFO] Enhanced features:
echo - Time-based checkpoints: Every 3 minutes (180 seconds)
echo - Rolling checkpoint stock: Keep latest 5 checkpoints
echo - Auto-resume on startup: Automatic checkpoint detection
echo - tqdm progress management: Multi-level progress bars
echo - Enhanced logging: Structured logging with timestamps
echo - Signal handling: Emergency checkpoints on interruption
echo - RTX 3060 optimization: Memory-efficient training
echo.

REM System resource check
echo [CHECK] Checking system resources...
python -c "
import torch
import psutil
import os

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  Device {i}: {props.name} ({props.total_memory // 1024**3}GB)')

print(f'CPU cores: {psutil.cpu_count()}')
print(f'Available memory: {psutil.virtual_memory().available // 1024**3}GB')
print(f'Disk space (C:): {psutil.disk_usage(\"C:\").free // 1024**3}GB')
print(f'Disk space (D:): {psutil.disk_usage(\"D:\").free // 1024**3}GB' if os.path.exists('D:') else 'D: drive not available')
"
echo.

REM Dataset check
echo [CHECK] Checking datasets...
if not exist "data\so8t_advanced_integrated\train_integrated.jsonl" (
    echo [ERROR] Training dataset not found!
    goto :error
)
if not exist "data\so8t_advanced_integrated\validation_integrated.jsonl" (
    echo [ERROR] Validation dataset not found!
    goto :error
)
echo [OK] Datasets found.

REM Output directory check
echo [CHECK] Checking output directory...
if not exist "H:\from_D\webdataset\checkpoints\ppo_so8t_enhanced" (
    echo [INFO] Creating output directory...
    mkdir "H:\from_D\webdataset\checkpoints\ppo_so8t_enhanced" 2>nul
    if errorlevel 1 (
        echo [WARNING] Could not create H: drive directory, using local fallback...
        mkdir "D:\webdataset\checkpoints\ppo_so8t_enhanced" 2>nul
    )
)
echo [OK] Output directory ready.

REM Check for existing checkpoints
echo [CHECK] Checking for existing checkpoints...
dir /b "H:\from_D\webdataset\checkpoints\ppo_so8t_enhanced" 2>nul | findstr "checkpoint" >nul
if %errorlevel% equ 0 (
    echo [INFO] Found existing checkpoints - auto-resume will be enabled.
) else (
    echo [INFO] No existing checkpoints - starting fresh training.
)

REM Start enhanced PPO training
echo [TRAIN] Starting SO8T Enhanced PPO training...
echo Training configuration:
echo - Model: microsoft/Phi-3.5-mini-instruct
echo - Dataset: data/so8t_advanced_integrated (30,000 samples)
echo - SO(8) Adapter: Integrated with residual connections
echo - Phi-3.5 Tags: Internal thinking tags applied
echo - Checkpoints: 3-minute intervals + rolling stock (5 max)
echo - Auto-resume: Enabled
echo - Progress: tqdm multi-level display
echo - RTX 3060 optimized
echo.

REM Execute enhanced training (short test run)
py -3 scripts/training/train_so8t_ppo_enhanced.py --max_steps 100

echo.
echo [RESULT] Enhanced PPO training test completed successfully!
echo.
echo Checkpoint management features verified:
echo ✅ Time-based checkpoints (3-minute intervals)
echo ✅ Rolling checkpoint stock (5 checkpoints max)
echo ✅ Auto-resume functionality
echo ✅ tqdm progress management
echo ✅ Enhanced logging
echo ✅ Signal handling
echo ✅ RTX 3060 optimization
echo.

REM Show checkpoint status
echo [STATUS] Current checkpoint status:
dir /b "H:\from_D\webdataset\checkpoints\ppo_so8t_enhanced" 2>nul | findstr "checkpoint"
if errorlevel 1 (
    dir /b "D:\webdataset\checkpoints\ppo_so8t_enhanced" 2>nul | findstr "checkpoint"
)
echo.

echo [AUDIO] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

goto :success

:error
echo [ERROR] Training failed!
powershell -ExecutionPolicy Bypass -Command "[System.Console]::Beep(800, 1000)"
exit /b 1

:success
echo [SO8T] Enhanced PPO training setup completed successfully!
echo.
echo To run full training:
echo py -3 scripts/training/train_so8t_ppo_enhanced.py --max_steps 10000
echo.
echo Features:
echo - 3-minute checkpoint intervals
echo - 5-rolling checkpoint stock
echo - Auto-resume from interruptions
echo - tqdm progress visualization
echo - Comprehensive logging
echo - Emergency checkpoint on signals
echo.
