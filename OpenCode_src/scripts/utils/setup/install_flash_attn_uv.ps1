# Flash Attention インストールスクリプト (uv版)
# Usage: .\scripts\utils\setup\install_flash_attn_uv.ps1

Write-Host "[INFO] Flash Attention Installation with uv" -ForegroundColor Green
Write-Host "===========================================" -ForegroundColor Cyan

# Pythonインタープリターの検出
Write-Host "[STEP 0] Detecting Python interpreter..." -ForegroundColor Yellow
try {
    $pythonExe = py -3 -c "import sys; print(sys.executable)" 2>&1
    if ($LASTEXITCODE -eq 0 -and $pythonExe) {
        Write-Host "  Found Python: $pythonExe" -ForegroundColor Green
        $pythonArg = "--python"
        $pythonValue = $pythonExe
    } else {
        throw "Could not detect Python"
    }
} catch {
    Write-Host "[WARNING] Could not detect Python with py launcher, trying python command..." -ForegroundColor Yellow
    try {
        $pythonExe = python -c "import sys; print(sys.executable)" 2>&1
        if ($LASTEXITCODE -eq 0 -and $pythonExe) {
            Write-Host "  Found Python: $pythonExe" -ForegroundColor Green
            $pythonArg = "--python"
            $pythonValue = $pythonExe
        } else {
            throw "Could not detect Python"
        }
    } catch {
        Write-Host "[ERROR] Could not detect Python interpreter" -ForegroundColor Red
        Write-Host "  Please install Python 3.10+ or fix Python path" -ForegroundColor Yellow
        exit 1
    }
}

# 環境確認
Write-Host "[STEP 1] Checking environment..." -ForegroundColor Yellow
$pythonVersion = py -3 --version 2>&1
Write-Host "  Python: $pythonVersion" -ForegroundColor White

$cudaAvailable = py -3 -c "import torch; print(torch.cuda.is_available())" 2>&1
Write-Host "  CUDA Available: $cudaAvailable" -ForegroundColor White

# Windows環境の確認
if ($env:OS -like '*Windows*') {
    Write-Host "[WARNING] Windows環境ではFlash Attentionのビルドが困難です" -ForegroundColor Yellow
    Write-Host "  推奨: WSL2またはLinux環境でインストール" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Windowsでのインストールを試行しますか？ (y/N): " -ForegroundColor Yellow -NoNewline
    $response = Read-Host
    if ($response -ne 'y' -and $response -ne 'Y') {
        Write-Host "[INFO] インストールをスキップしました" -ForegroundColor Yellow
        Write-Host "  Flash Attentionはオプショナルな依存関係です" -ForegroundColor White
        exit 0
    }
}

# Flash Attention インストール
Write-Host "[STEP 2] Installing flash-attention with uv..." -ForegroundColor Yellow
try {
    uv pip install $pythonArg $pythonValue "flash-attn>=2.5.8" --no-build-isolation
    Write-Host "[OK] Flash Attention installed successfully!" -ForegroundColor Green
    
    # 動作確認
    Write-Host "[STEP 3] Verifying installation..." -ForegroundColor Yellow
    $importResult = py -3 -c "from flash_attn import flash_attn_func; print('[OK] Flash Attention imported successfully')" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host $importResult -ForegroundColor Green
        Write-Host "[OK] Flash Attention is ready to use!" -ForegroundColor Green
    } else {
        Write-Host "[WARNING] Flash Attention installed but import failed" -ForegroundColor Yellow
        Write-Host $importResult -ForegroundColor Red
    }
} catch {
    Write-Host "[ERROR] Flash Attention installation failed" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    Write-Host "[INFO] Flash Attention is optional - the system will work without it" -ForegroundColor Yellow
    Write-Host "  Standard attention will be used instead" -ForegroundColor White
}

