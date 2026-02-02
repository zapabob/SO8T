# Flash Attention インストール状況まとめ
# Windows環境での問題と解決策

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Flash Attention インストール状況まとめ" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. Python環境の確認
Write-Host "[1] Python環境の確認" -ForegroundColor Yellow

Write-Host "Python 3.11: " -NoNewline
try {
    $py311Version = py -3.11 -c "import sys; print(sys.version.split()[0])" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host $py311Version -ForegroundColor Green
        Write-Host "  torch: " -NoNewline
        $py311Torch = py -3.11 -c "import torch; print(torch.__version__)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host $py311Torch -ForegroundColor Green
        } else {
            Write-Host "未インストール（ディスク容量不足）" -ForegroundColor Red
        }
    } else {
        Write-Host "見つかりません" -ForegroundColor Red
    }
} catch {
    Write-Host "見つかりません" -ForegroundColor Red
}

Write-Host "Python 3.12: " -NoNewline
try {
    $py312Version = py -3.12 -c "import sys; print(sys.version.split()[0])" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host $py312Version -ForegroundColor Green
        Write-Host "  torch: " -NoNewline
        $py312Torch = py -3.12 -c "import torch; print(torch.__version__)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host $py312Torch -ForegroundColor Green
        } else {
            Write-Host "未インストール" -ForegroundColor Red
        }
    } else {
        Write-Host "見つかりません" -ForegroundColor Red
    }
} catch {
    Write-Host "見つかりません" -ForegroundColor Red
}

Write-Host ""

# 2. ビルドツールの確認
Write-Host "[2] ビルドツールの確認" -ForegroundColor Yellow
$vsPath = "${env:ProgramFiles}\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC"
if (Test-Path $vsPath) {
    Write-Host "Visual Studio Build Tools: " -NoNewline
    Write-Host "インストール済み" -ForegroundColor Green
} else {
    Write-Host "Visual Studio Build Tools: " -NoNewline
    Write-Host "未インストール" -ForegroundColor Red
    Write-Host "  → インストールが必要: https://visualstudio.microsoft.com/downloads/" -ForegroundColor Yellow
}

Write-Host ""

# 3. CUDA Toolkitの確認
Write-Host "[3] CUDA Toolkitの確認" -ForegroundColor Yellow
$cudaPath = "${env:CUDA_PATH}"
if ($cudaPath) {
    Write-Host "CUDA_PATH: " -NoNewline
    Write-Host $cudaPath -ForegroundColor Green
} else {
    Write-Host "CUDA_PATH: " -NoNewline
    Write-Host "未設定" -ForegroundColor Red
    Write-Host "  → インストールが必要: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Yellow
}

Write-Host ""

# 4. Flash Attentionの状態
Write-Host "[4] Flash Attentionの状態" -ForegroundColor Yellow
Write-Host "Python 3.11: " -NoNewline
py -3.11 -c "from flash_attn import flash_attn_func" 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "インストール済み" -ForegroundColor Green
} else {
    Write-Host "未インストール" -ForegroundColor Red
}

Write-Host "Python 3.12: " -NoNewline
py -3.12 -c "from flash_attn import flash_attn_func" 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "インストール済み" -ForegroundColor Green
} else {
    Write-Host "未インストール" -ForegroundColor Red
}

Write-Host ""

# 5. 推奨される解決策
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "推奨される解決策" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "[推奨] Flash Attentionなしで動作" -ForegroundColor Green
Write-Host "  - flash_attnはオプショナルな依存関係" -ForegroundColor White
Write-Host "  - インストールされていなくても動作する" -ForegroundColor White
Write-Host "  - 標準のattentionでフォールバック" -ForegroundColor White
Write-Host "  - パフォーマンスは若干低下するが、機能は正常に動作" -ForegroundColor White
Write-Host ""
Write-Host "[代替] Flash Attentionをインストールする場合" -ForegroundColor Yellow
Write-Host "  1. Visual Studio Build Tools 2022をインストール" -ForegroundColor White
Write-Host "     https://visualstudio.microsoft.com/downloads/" -ForegroundColor Cyan
Write-Host "     - C++ build tools" -ForegroundColor White
Write-Host "     - Windows 10/11 SDK" -ForegroundColor White
Write-Host ""
Write-Host "  2. CUDA Toolkit 12.1をインストール（既にインストール済みの可能性あり）" -ForegroundColor White
Write-Host "     https://developer.nvidia.com/cuda-downloads" -ForegroundColor Cyan
Write-Host ""
Write-Host "  3. Python 3.12環境でインストール試行:" -ForegroundColor White
Write-Host "     py -3.12 -m pip install flash_attn==2.5.8 --no-build-isolation" -ForegroundColor Cyan
Write-Host ""
Write-Host "  4. または、Linux環境（WSL2）でビルドしてwheelファイルを作成" -ForegroundColor White
Write-Host ""
