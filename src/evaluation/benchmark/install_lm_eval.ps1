# LM-Evaluation-Harness インストールスクリプト
# AEGIS A/Bテストで使用する評価フレームワークをインストール

Write-Host "[START] Installing LM-Evaluation-Harness..." -ForegroundColor Green

# Python環境確認
py --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Python not found" -ForegroundColor Red
    exit 1
}

# 作業ディレクトリ設定
Set-Location $PSScriptRoot\..\..

# LM-Evaluation-Harnessクローン
if (!(Test-Path "external\lm-evaluation-harness")) {
    Write-Host "[GIT] Cloning lm-evaluation-harness..." -ForegroundColor Yellow
    git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness external/lm-evaluation-harness
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to clone lm-evaluation-harness" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[SKIP] lm-evaluation-harness already exists" -ForegroundColor Cyan
}

# インストール
Set-Location external\lm-evaluation-harness

Write-Host "[PIP] Installing base package..." -ForegroundColor Yellow
py -m pip install -e .

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to install base package" -ForegroundColor Red
    exit 1
}

Write-Host "[PIP] Installing HF dependencies..." -ForegroundColor Yellow
py -m pip install "transformers" "accelerate" "datasets"

if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARNING] Failed to install HF dependencies (some features may not work)" -ForegroundColor Yellow
}

Write-Host "[PIP] Installing llama-cpp-python..." -ForegroundColor Yellow
py -m pip install "llama-cpp-python"

if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARNING] Failed to install llama-cpp-python (GGUF evaluation may not work)" -ForegroundColor Yellow
}

# 元のディレクトリに戻る
Set-Location ..\..

Write-Host "[OK] LM-Evaluation-Harness installation completed" -ForegroundColor Green

# テスト実行
Write-Host "[TEST] Testing lm_eval installation..." -ForegroundColor Yellow
py -c "import lm_eval; print('lm_eval import successful')"

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] lm_eval import test passed" -ForegroundColor Green
} else {
    Write-Host "[WARNING] lm_eval import test failed" -ForegroundColor Yellow
}

# 完了通知
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"


Write-Host "[START] Installing LM-Evaluation-Harness..." -ForegroundColor Green

# Python環境確認
py --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Python not found" -ForegroundColor Red
    exit 1
}

# 作業ディレクトリ設定
Set-Location $PSScriptRoot\..\..

# LM-Evaluation-Harnessクローン
if (!(Test-Path "external\lm-evaluation-harness")) {
    Write-Host "[GIT] Cloning lm-evaluation-harness..." -ForegroundColor Yellow
    git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness external/lm-evaluation-harness
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to clone lm-evaluation-harness" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[SKIP] lm-evaluation-harness already exists" -ForegroundColor Cyan
}

# インストール
Set-Location external\lm-evaluation-harness

Write-Host "[PIP] Installing base package..." -ForegroundColor Yellow
py -m pip install -e .

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to install base package" -ForegroundColor Red
    exit 1
}

Write-Host "[PIP] Installing HF dependencies..." -ForegroundColor Yellow
py -m pip install "transformers" "accelerate" "datasets"

if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARNING] Failed to install HF dependencies (some features may not work)" -ForegroundColor Yellow
}

Write-Host "[PIP] Installing llama-cpp-python..." -ForegroundColor Yellow
py -m pip install "llama-cpp-python"

if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARNING] Failed to install llama-cpp-python (GGUF evaluation may not work)" -ForegroundColor Yellow
}

# 元のディレクトリに戻る
Set-Location ..\..

Write-Host "[OK] LM-Evaluation-Harness installation completed" -ForegroundColor Green

# テスト実行
Write-Host "[TEST] Testing lm_eval installation..." -ForegroundColor Yellow
py -c "import lm_eval; print('lm_eval import successful')"

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] lm_eval import test passed" -ForegroundColor Green
} else {
    Write-Host "[WARNING] lm_eval import test failed" -ForegroundColor Yellow
}

# 完了通知
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"


Write-Host "[START] Installing LM-Evaluation-Harness..." -ForegroundColor Green

# Python環境確認
py --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Python not found" -ForegroundColor Red
    exit 1
}

# 作業ディレクトリ設定
Set-Location $PSScriptRoot\..\..

# LM-Evaluation-Harnessクローン
if (!(Test-Path "external\lm-evaluation-harness")) {
    Write-Host "[GIT] Cloning lm-evaluation-harness..." -ForegroundColor Yellow
    git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness external/lm-evaluation-harness
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to clone lm-evaluation-harness" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[SKIP] lm-evaluation-harness already exists" -ForegroundColor Cyan
}

# インストール
Set-Location external\lm-evaluation-harness

Write-Host "[PIP] Installing base package..." -ForegroundColor Yellow
py -m pip install -e .

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to install base package" -ForegroundColor Red
    exit 1
}

Write-Host "[PIP] Installing HF dependencies..." -ForegroundColor Yellow
py -m pip install "transformers" "accelerate" "datasets"

if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARNING] Failed to install HF dependencies (some features may not work)" -ForegroundColor Yellow
}

Write-Host "[PIP] Installing llama-cpp-python..." -ForegroundColor Yellow
py -m pip install "llama-cpp-python"

if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARNING] Failed to install llama-cpp-python (GGUF evaluation may not work)" -ForegroundColor Yellow
}

# 元のディレクトリに戻る
Set-Location ..\..

Write-Host "[OK] LM-Evaluation-Harness installation completed" -ForegroundColor Green

# テスト実行
Write-Host "[TEST] Testing lm_eval installation..." -ForegroundColor Yellow
py -c "import lm_eval; print('lm_eval import successful')"

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] lm_eval import test passed" -ForegroundColor Green
} else {
    Write-Host "[WARNING] lm_eval import test failed" -ForegroundColor Yellow
}

# 完了通知
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"
