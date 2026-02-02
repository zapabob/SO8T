@echo off
chcp 65001 >nul
echo [SO8T] Installing Enhanced PPO Training Dependencies
echo ====================================================
echo.

echo [STEP 1] Upgrading transformers and huggingface-hub...
pip install transformers -U
pip install huggingface-hub -U

echo.
echo [STEP 2] Installing TRL (Transformer Reinforcement Learning)...
pip install trl

echo.
echo [STEP 3] Installing additional dependencies...
pip install accelerate
pip install peft
pip install bitsandbytes

echo.
echo [STEP 4] Installing tqdm and other utilities...
pip install tqdm
pip install psutil

echo.
echo [STEP 5] Checking GPU availability...
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  Device {i}: {props.name} ({props.total_memory // 1024**3}GB)')
else:
    print('CUDA not available - will run on CPU')
"

echo.
echo [STEP 6] Testing imports...
python -c "
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print('✓ transformers import successful')
except ImportError as e:
    print(f'✗ transformers import failed: {e}')

try:
    from trl import PPOTrainer, PPOConfig
    print('✓ TRL import successful')
except ImportError as e:
    print(f'✗ TRL import failed: {e}')

try:
    import tqdm
    print('✓ tqdm import successful')
except ImportError as e:
    print(f'✗ tqdm import failed: {e}')

try:
    import psutil
    print('✓ psutil import successful')
except ImportError as e:
    print(f'✗ psutil import failed: {e}')

print('Dependencies check completed!')
"

echo.
echo [SUCCESS] SO8T Enhanced PPO dependencies installation completed!
echo.
echo You can now run:
echo py -3 scripts/training/train_so8t_ppo_enhanced.py --max_steps 100
echo.
