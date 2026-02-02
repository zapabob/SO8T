# AEGIS Soul Fusion Workflow Test Script
# 魂の定着ワークフローをテスト実行する

param(
    [switch]$SkipTraining,
    [switch]$SkipFusion,
    [switch]$SkipGGUF
)

Write-Host "[WORKFLOW] AEGIS Soul Fusion Test" -ForegroundColor Green
Write-Host "===================================" -ForegroundColor Green

# UTF-8設定
chcp 65001 > $null

# 作業ディレクトリ確認
$currentDir = Get-Location
Write-Host "[INFO] Working Directory: $currentDir" -ForegroundColor Cyan

# 依存関係チェック
Write-Host "[CHECK] Checking dependencies..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[OK] Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python not found" -ForegroundColor Red
    exit 1
}

try {
    $torchVersion = python -c "import torch; print(torch.__version__)" 2>&1
    Write-Host "[OK] PyTorch: $torchVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] PyTorch not found" -ForegroundColor Red
    exit 1
}

# STEP 1: Soul Injection Training
if (-not $SkipTraining) {
    Write-Host "`n[STEP 1] Starting Soul Injection Training..." -ForegroundColor Green
    Write-Host "   This will train LoRA + Alpha Gate + SO(8) Rotation on Borea-Phi3.5" -ForegroundColor Cyan

    try {
        python scripts/training/train_soul_injection.py
        Write-Host "[OK] Soul Injection Training completed" -ForegroundColor Green
    } catch {
        Write-Host "[ERROR] Soul Injection Training failed: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "`n[SKIP] Soul Injection Training (using existing checkpoints)" -ForegroundColor Yellow
}

# STEP 2: Soul Fusion
if (-not $SkipFusion) {
    Write-Host "`n[STEP 2] Starting Soul Fusion..." -ForegroundColor Green
    Write-Host "   This will mathematically fuse Alpha+Rotation into LM Head weights" -ForegroundColor Cyan

    try {
        python scripts/training/fuse_soul_for_gguf.py
        Write-Host "[OK] Soul Fusion completed" -ForegroundColor Green
    } catch {
        Write-Host "[ERROR] Soul Fusion failed: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "`n[SKIP] Soul Fusion (using existing fused model)" -ForegroundColor Yellow
}

# STEP 3: GGUF Conversion
if (-not $SkipGGUF) {
    Write-Host "`n[STEP 3] Starting GGUF Conversion..." -ForegroundColor Green
    Write-Host "   This will convert the fused model to GGUF format for llama.cpp" -ForegroundColor Cyan

    # llama.cppパスチェック
    $llamaCppPath = "external/llama.cpp-master"
    if (-not (Test-Path $llamaCppPath)) {
        Write-Host "[ERROR] llama.cpp not found at $llamaCppPath" -ForegroundColor Red
        Write-Host "[INFO] Please run setup_llama_cpp.ps1 first" -ForegroundColor Yellow
        exit 1
    }

    # モデルパスチェック
    $modelPath = "models/AEGIS-Phi3.5-Hybrid"
    if (-not (Test-Path $modelPath)) {
        Write-Host "[ERROR] Fused model not found at $modelPath" -ForegroundColor Red
        exit 1
    }

    # GGUF出力ディレクトリ作成
    $ggufDir = "D:/webdataset/gguf_models/agiasi-phi3.5"
    New-Item -ItemType Directory -Force -Path $ggufDir > $null

    try {
        Push-Location $llamaCppPath
        python convert_hf_to_gguf.py `
            "$currentDir/$modelPath" `
            --outfile "D:/webdataset/gguf_models/agiasi-phi3.5/agiasi-phi3.5-q4_k_m.gguf" `
            --outtype q4_k_m
        Pop-Location

        Write-Host "[OK] GGUF Conversion completed" -ForegroundColor Green
        Write-Host "   Output: D:/webdataset/gguf_models/agiasi-phi3.5/agiasi-phi3.5-q4_k_m.gguf" -ForegroundColor Cyan
    } catch {
        Write-Host "[ERROR] GGUF Conversion failed: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "`n[SKIP] GGUF Conversion" -ForegroundColor Yellow
}

# STEP 4: Ollama Test
Write-Host "`n[STEP 4] Testing with Ollama..." -ForegroundColor Green

try {
    # Ollamaでモデル作成
    $modelfile = @"
FROM D:/webdataset/gguf_models/agiasi-phi3.5/agiasi-phi3.5-q4_k_m.gguf

TEMPLATE """{{ .System }}

{{ .Prompt }}"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
"@

    $modelfile | Out-File -FilePath "modelfiles/agiasi-phi3.5.modelfile" -Encoding UTF8

    ollama create agiasi-phi3.5:latest -f modelfiles/agiasi-phi3.5.modelfile

    # テスト実行
    Write-Host "[TEST] Running AEGIS with physical intelligence..." -ForegroundColor Cyan
    ollama run agiasi-phi3.5:latest "時間はなぜ不可逆なのですか？ 物理的に説明してください。"

    Write-Host "[OK] Ollama test completed" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Ollama test failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n[COMPLETE] Soul Fusion Workflow Test Finished!" -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green

# 音声通知
Write-Host "[AUDIO] Playing completion notification..." -ForegroundColor Green
try {
    Add-Type -AssemblyName System.Windows.Forms
    $player = New-Object System.Media.SoundPlayer "C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav"
    $player.PlaySync()
    Write-Host "[OK] Audio notification played" -ForegroundColor Green
} catch {
    Write-Host "[WARNING] Audio notification failed, trying beep..." -ForegroundColor Yellow
    [System.Console]::Beep(1000, 500)
}

