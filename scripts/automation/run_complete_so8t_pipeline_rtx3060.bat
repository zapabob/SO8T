@echo off
chcp 65001 >nul
echo [RTX3060] Starting SO8T Complete Automation Pipeline
echo ====================================================
echo Frozen base weights + QLoRA fine-tuning for RTX3060
echo ====================================================

REM RTX3060メモリチェック
echo [CHECK] Checking RTX3060 GPU memory...
python -c "
import torch
if torch.cuda.is_available():
    mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'[OK] GPU Memory: {mem:.1f}GB')
    if mem < 8:
        print('[ERROR] Insufficient GPU memory for RTX3060')
        exit(1)
else:
    echo [ERROR] CUDA not available
    exit(1)
"

if %errorlevel% neq 0 (
    echo [ERROR] RTX3060 check failed
    goto :error
)

REM 環境変数設定（RTX3060最適化）
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
set TORCH_USE_CUDA_DSA=1
set HF_HUB_DISABLE_PROGRESS_BARS=1

echo [START] Running RTX3060 SO8T Automation Pipeline...
python scripts/automation/complete_so8t_automation_pipeline_rtx3060.py

if %errorlevel% equ 0 (
    echo [SUCCESS] RTX3060 SO8T pipeline completed successfully!
    goto :audio_success
) else (
    echo [ERROR] RTX3060 SO8T pipeline failed
    goto :error
)

:audio_success
echo [AUDIO] Playing success notification...
powershell -ExecutionPolicy Bypass -Command "
try {
    Add-Type -AssemblyName System.Windows.Forms
    $player = New-Object System.Media.SoundPlayer
    $player.SoundLocation = 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav'
    $player.PlaySync()
    Write-Host '[OK] marisa_owattaze.wav played successfully'
} catch {
    Write-Host '[FALLBACK] Using beep sound'
    [System.Console]::Beep(1000, 500)
}
"
goto :end

:error
echo [ERROR] Pipeline execution failed
powershell -ExecutionPolicy Bypass -Command "
try {
    [System.Console]::Beep(800, 1000)
} catch {
    echo Audio notification failed
}
"
exit /b 1

:end
echo [DONE] RTX3060 SO8T automation pipeline finished
