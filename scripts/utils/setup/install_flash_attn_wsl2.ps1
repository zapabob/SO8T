# Flash Attention WSL2 インストールガイド
# Usage: .\scripts\utils\setup\install_flash_attn_wsl2.ps1

Write-Host "[INFO] Flash Attention WSL2 Installation Guide" -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "[STEP 1] WSL2環境の確認" -ForegroundColor Yellow
Write-Host "  WSL2がインストールされているか確認します..." -ForegroundColor White

# WSL2の確認
$wslInstalled = wsl --status 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] WSL2 is installed" -ForegroundColor Green
    wsl --list --verbose
} else {
    Write-Host "  [WARNING] WSL2 is not installed" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "[INFO] WSL2のインストール方法:" -ForegroundColor Cyan
    Write-Host "  1. PowerShell (管理者)で実行:" -ForegroundColor White
    Write-Host "     wsl --install" -ForegroundColor Green
    Write-Host "  2. 再起動後、WSL2が利用可能になります" -ForegroundColor White
    Write-Host ""
    Write-Host "  または、手動でインストール:" -ForegroundColor White
    Write-Host "    - Microsoft StoreからUbuntuをインストール" -ForegroundColor Cyan
    Write-Host "    - wsl --set-default-version 2" -ForegroundColor Cyan
    exit 1
}

Write-Host ""
Write-Host "[STEP 2] WSL2でのインストール手順" -ForegroundColor Yellow
Write-Host "  WSL2環境で以下のコマンドを実行してください:" -ForegroundColor White
Write-Host ""
Write-Host "  # WSL2に接続" -ForegroundColor Cyan
Write-Host "  wsl" -ForegroundColor Green
Write-Host ""
Write-Host "  # プロジェクトディレクトリに移動" -ForegroundColor Cyan
Write-Host "  cd /mnt/c/Users/downl/Desktop/SO8T" -ForegroundColor Green
Write-Host ""
Write-Host "  # スクリプトに実行権限を付与" -ForegroundColor Cyan
Write-Host "  chmod +x scripts/utils/setup/*.sh" -ForegroundColor Green
Write-Host ""
Write-Host "  # Flash Attentionインストールスクリプトを実行" -ForegroundColor Cyan
Write-Host "  bash scripts/utils/setup/install_flash_attn_uv.sh" -ForegroundColor Green
Write-Host ""
Write-Host "  または、直接インストール:" -ForegroundColor Cyan
Write-Host "  python3 -m pip install flash-attn==2.5.8 --no-build-isolation" -ForegroundColor Green
Write-Host ""

Write-Host "[STEP 3] インストール後の確認" -ForegroundColor Yellow
Write-Host "  WSL2環境で以下を実行して確認:" -ForegroundColor White
Write-Host "  python3 -c 'from flash_attn import flash_attn_func; print(\"Flash Attention installed successfully\")'" -ForegroundColor Green
Write-Host ""

Write-Host "[INFO] 注意事項:" -ForegroundColor Cyan
Write-Host "  - WSL2環境ではビルドが成功しやすいです" -ForegroundColor White
Write-Host "  - CUDA ToolkitがWSL2にインストールされている必要があります" -ForegroundColor White
Write-Host "  - インストールには10-30分かかる場合があります" -ForegroundColor White
Write-Host ""

Write-Host "[INFO] WSL2でのCUDAセットアップが必要な場合:" -ForegroundColor Cyan
Write-Host "  1. NVIDIAドライバーが最新であることを確認" -ForegroundColor White
Write-Host "  2. WSL2用のCUDA Toolkitをインストール" -ForegroundColor White
Write-Host "     https://developer.nvidia.com/cuda-downloads" -ForegroundColor Green
Write-Host "  3. WSL2環境でnvidia-smiを実行して確認" -ForegroundColor White





















