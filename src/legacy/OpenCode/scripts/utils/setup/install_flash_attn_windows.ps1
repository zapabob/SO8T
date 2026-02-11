# Flash Attention 2.5.8 Windows インストールスクリプト
# RTX3080 CUDA12環境用

Write-Host "[INFO] Flash Attention 2.5.8 Windows インストール開始..." -ForegroundColor Green

# 1. PyTorchの確認
Write-Host "[STEP 1] PyTorchの確認中..." -ForegroundColor Yellow
py -3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
if ($LASTEXITCODE -ne 0) {
    Write-Error "[ERROR] PyTorchがインストールされていません。先にPyTorchをインストールしてください。"
    exit 1
}

# 2. ビルドツールの確認
Write-Host "[STEP 2] ビルドツールの確認中..." -ForegroundColor Yellow
$vsPath = "${env:ProgramFiles}\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC"
if (-not (Test-Path $vsPath)) {
    Write-Warning "[WARNING] Visual Studio Build Toolsが見つかりません。"
    Write-Host "[INFO] Visual Studio Build Toolsのインストールが必要です:" -ForegroundColor Yellow
    Write-Host "  https://visualstudio.microsoft.com/downloads/" -ForegroundColor Cyan
    Write-Host "[INFO] または、flash_attnなしで動作することも可能です（パフォーマンスは低下します）" -ForegroundColor Yellow
}

# 3. CUDA Toolkitの確認
Write-Host "[STEP 3] CUDA Toolkitの確認中..." -ForegroundColor Yellow
$cudaPath = "${env:CUDA_PATH}"
if (-not $cudaPath) {
    Write-Warning "[WARNING] CUDA_PATH環境変数が設定されていません。"
    Write-Host "[INFO] CUDA Toolkitのインストールが必要です:" -ForegroundColor Yellow
    Write-Host "  https://developer.nvidia.com/cuda-downloads" -ForegroundColor Cyan
}

# 4. flash_attnのインストール試行
Write-Host "[STEP 4] Flash Attention 2.5.8のインストール試行中..." -ForegroundColor Yellow
Write-Host "[INFO] この処理には時間がかかる場合があります（10-30分）..." -ForegroundColor Yellow

# ビルド依存関係を明示的に指定
$env:PIP_NO_BUILD_ISOLATION = "0"
py -3 -m pip install flash_attn==2.5.8 --no-build-isolation --verbose

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Flash Attention 2.5.8のインストールが完了しました！" -ForegroundColor Green
    
    # インストール確認
    Write-Host "[STEP 5] インストール確認中..." -ForegroundColor Yellow
    py -3 -c "from flash_attn import flash_attn_func; print('[OK] Flash Attentionが正常にインポートできました')"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Flash Attentionのインストールと動作確認が完了しました！" -ForegroundColor Green
    } else {
        Write-Warning "[WARNING] Flash Attentionのインポートに失敗しました。"
    }
} else {
    Write-Warning "[WARNING] Flash Attention 2.5.8のインストールに失敗しました。"
    Write-Host "[INFO] Flash Attentionはオプショナルな依存関係です。" -ForegroundColor Yellow
    Write-Host "[INFO] インストールされていなくても、標準のattentionで動作します（パフォーマンスは低下します）。" -ForegroundColor Yellow
    Write-Host "[INFO] 代替手段:" -ForegroundColor Yellow
    Write-Host "  1. Linux環境でビルドしてwheelファイルを作成" -ForegroundColor Cyan
    Write-Host "  2. 標準のattentionを使用（既に実装済み）" -ForegroundColor Cyan
    Write-Host "  3. Visual Studio Build ToolsとCUDA Toolkitをインストールして再試行" -ForegroundColor Cyan
}

Write-Host "[INFO] インストール処理が完了しました。" -ForegroundColor Green

