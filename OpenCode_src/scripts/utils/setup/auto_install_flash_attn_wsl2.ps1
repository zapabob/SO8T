# Fully Automated Flash Attention 2.5.0 Installation for WSL2 (PowerShell wrapper)
# This script automates the entire installation process from Windows

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Fully Automated Flash Attention Installation" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[INFO] Starting fully automated installation..." -ForegroundColor Green
Write-Host "[INFO] This will:" -ForegroundColor Yellow
Write-Host "  1. Check Python and PyTorch" -ForegroundColor White
Write-Host "  2. Install CUDA toolkit (if needed, requires sudo password)" -ForegroundColor White
Write-Host "  3. Install flash-attention 2.5.0" -ForegroundColor White
Write-Host "  4. Verify installation" -ForegroundColor White
Write-Host ""

Write-Host "[WARNING] CUDA toolkit installation requires sudo password" -ForegroundColor Yellow
Write-Host "[WARNING] This may take 30-50 minutes total" -ForegroundColor Yellow
Write-Host ""

$response = Read-Host "Continue? (Y/N)"
if ($response -ne "Y" -and $response -ne "y") {
    Write-Host "[INFO] Installation cancelled" -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "[STEP] Executing installation script in WSL2..." -ForegroundColor Cyan
wsl bash -c "cd /mnt/c/Users/downl/Desktop/SO8T && bash scripts/utils/setup/auto_install_flash_attn_wsl2.sh"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "[OK] Installation completed successfully!" -ForegroundColor Green
    Write-Host "[INFO] Flash Attention 2.5.0 is now available in WSL2" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "[ERROR] Installation failed" -ForegroundColor Red
    Write-Host "[INFO] Check the error messages above for details" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "[AUDIO] Playing completion notification..." -ForegroundColor Green
& "scripts\utils\play_audio_notification.ps1"
