# uv 依存関係インストールスクリプト
# Usage: .\scripts\utils\setup\install_dependencies_uv.ps1

Write-Host "[INFO] Installing dependencies with uv..." -ForegroundColor Green

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

# PyTorch with CUDA 12.1 support
Write-Host "[STEP 1] Installing PyTorch with CUDA 12.1..." -ForegroundColor Yellow
uv pip install $pythonArg $pythonValue torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 基本依存関係のインストール
Write-Host "[STEP 2] Installing core dependencies..." -ForegroundColor Yellow
# requirements.txtから主要な依存関係をインストール（optuna-dashboardはスキップ）
$requirements = Get-Content requirements.txt | Where-Object { $_ -notmatch '^#' -and $_ -notmatch '^--index-url' -and $_ -notmatch 'optuna-dashboard' -and $_.Trim() -ne '' }
$tempReq = [System.IO.Path]::GetTempFileName()
$requirements | Out-File -FilePath $tempReq -Encoding UTF8
try {
    uv pip install $pythonArg $pythonValue -r $tempReq
} finally {
    Remove-Item $tempReq -ErrorAction SilentlyContinue
}

# Flash Attention (オプショナル、Windowsではスキップ)
if ($env:OS -notlike '*Windows*') {
    Write-Host "[STEP 3] Installing flash-attention (Linux/WSL2 only)..." -ForegroundColor Yellow
    uv pip install $pythonArg $pythonValue "flash-attn>=2.5.8" --no-build-isolation
} else {
    Write-Host "[STEP 3] Skipping flash-attention (Windows build is difficult)" -ForegroundColor Yellow
    Write-Host "  Use WSL2 or Linux for flash-attention installation" -ForegroundColor Cyan
    Write-Host "  Or install manually: uv pip install $pythonArg $pythonValue flash-attn>=2.5.8 --no-build-isolation" -ForegroundColor Cyan
}

# 開発依存関係のインストール
Write-Host "[STEP 4] Installing dev dependencies..." -ForegroundColor Yellow
# 開発依存関係はrequirements.txtに含まれているためスキップ
Write-Host "  Dev dependencies are included in requirements.txt" -ForegroundColor Cyan

Write-Host "[OK] Dependencies installed successfully!" -ForegroundColor Green
Write-Host "[INFO] To install flash-attention on Windows, use WSL2 or install manually" -ForegroundColor Cyan
