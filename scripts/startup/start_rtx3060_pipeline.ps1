# RTX3060向けSO8T PPO Pipeline 自動起動スクリプト
# 電源投入時にタスクスケジューラーで実行されることを想定

param(
    [switch]$TestMode = $false
)

Write-Host "[RTX3060] SO8T PPO Pipeline Power-On Startup" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# RTX3060のスペック確認
Write-Host "[STEP 1] RTX3060 Hardware Check..." -ForegroundColor Yellow
$cudaInfo = python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Devices: {torch.cuda.device_count()}')" 2>$null
Write-Host $cudaInfo

# 環境変数設定 (RTX3060向け)
Write-Host "[STEP 2] Setting RTX3060 Environment..." -ForegroundColor Yellow
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:512"

# 作業ディレクトリ設定
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent $scriptDir)
Set-Location $projectRoot

Write-Host "[STEP 3] RTX3060 Memory Configuration:" -ForegroundColor Yellow
Write-Host "  - GPU Memory Limit: 75% (9GB of 12GB VRAM)" -ForegroundColor White
Write-Host "  - CPU Offload: Enabled" -ForegroundColor White
Write-Host "  - Gradient Checkpointing: Enabled" -ForegroundColor White
Write-Host "  - Max Steps: 100 (RTX3060 optimized)" -ForegroundColor White

if ($TestMode) {
    Write-Host "[TEST MODE] Running in test mode..." -ForegroundColor Magenta
    # テストモードでは短い実行
    $configFile = "aegis_v2_test_config.json"
} else {
    Write-Host "[PRODUCTION MODE] Running full pipeline..." -ForegroundColor Green
    $configFile = "aegis_v2_config.json"
}

# パイプライン実行
Write-Host "[STEP 4] Starting SO8T PPO Training..." -ForegroundColor Yellow
try {
    & py -3 scripts/training/train_aegis_v2_ppo_so8t.py --config $configFile
    Write-Host "[SUCCESS] Pipeline completed successfully!" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Pipeline failed: $($_.Exception.Message)" -ForegroundColor Red
}

# 完了通知
Write-Host "[STEP 5] Playing completion notification..." -ForegroundColor Yellow
try {
    & powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"
} catch {
    Write-Host "[WARNING] Audio notification failed" -ForegroundColor Yellow
}

Write-Host "[RTX3060] Power-on startup completed!" -ForegroundColor Cyan


