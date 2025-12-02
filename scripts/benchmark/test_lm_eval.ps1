# LM-Evaluation-Harness テストスクリプト
# hellaswagとMMLUをPhi-3.5-mini-instructでテスト

Write-Host "[START] Testing LM-Evaluation-Harness with Phi-3.5-mini-instruct..." -ForegroundColor Green

# 作業ディレクトリ設定
Set-Location $PSScriptRoot\..\..

# 出力ディレクトリ作成
$lmEvalDir = "H:\from_D\webdataset\benchmark_results\lm_eval"
if (!(Test-Path $lmEvalDir)) {
    New-Item -ItemType Directory -Path $lmEvalDir -Force
}

# テスト実行
Write-Host "[TEST] Testing hellaswag..." -ForegroundColor Yellow
py -m lm_eval --model hf `
    --model_args pretrained=microsoft/Phi-3.5-mini-instruct,dtype=bfloat16,trust_remote_code=True `
    --tasks hellaswag `
    --device cuda `
    --batch_size auto `
    --output_path "$lmEvalDir\phi35_hellaswag_test.json" `
    --log_samples

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] hellaswag test completed" -ForegroundColor Green
} else {
    Write-Host "[NG] hellaswag test failed" -ForegroundColor Red
}

Write-Host "[TEST] Testing MMLU..." -ForegroundColor Yellow
py -m lm_eval --model hf `
    --model_args pretrained=microsoft/Phi-3.5-mini-instruct,dtype=bfloat16,trust_remote_code=True `
    --tasks mmlu `
    --device cuda `
    --batch_size auto `
    --output_path "$lmEvalDir\phi35_mmlu_test.json" `
    --log_samples

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] MMLU test completed" -ForegroundColor Green
} else {
    Write-Host "[NG] MMLU test failed" -ForegroundColor Red
}

# 結果確認
Write-Host "[RESULTS] Test results saved to: $lmEvalDir" -ForegroundColor Cyan
Get-ChildItem $lmEvalDir -Filter "*.json" | ForEach-Object {
    Write-Host "  - $($_.Name)" -ForegroundColor White
}

# 完了通知
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"


Write-Host "[START] Testing LM-Evaluation-Harness with Phi-3.5-mini-instruct..." -ForegroundColor Green

# 作業ディレクトリ設定
Set-Location $PSScriptRoot\..\..

# 出力ディレクトリ作成
$lmEvalDir = "H:\from_D\webdataset\benchmark_results\lm_eval"
if (!(Test-Path $lmEvalDir)) {
    New-Item -ItemType Directory -Path $lmEvalDir -Force
}

# テスト実行
Write-Host "[TEST] Testing hellaswag..." -ForegroundColor Yellow
py -m lm_eval --model hf `
    --model_args pretrained=microsoft/Phi-3.5-mini-instruct,dtype=bfloat16,trust_remote_code=True `
    --tasks hellaswag `
    --device cuda `
    --batch_size auto `
    --output_path "$lmEvalDir\phi35_hellaswag_test.json" `
    --log_samples

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] hellaswag test completed" -ForegroundColor Green
} else {
    Write-Host "[NG] hellaswag test failed" -ForegroundColor Red
}

Write-Host "[TEST] Testing MMLU..." -ForegroundColor Yellow
py -m lm_eval --model hf `
    --model_args pretrained=microsoft/Phi-3.5-mini-instruct,dtype=bfloat16,trust_remote_code=True `
    --tasks mmlu `
    --device cuda `
    --batch_size auto `
    --output_path "$lmEvalDir\phi35_mmlu_test.json" `
    --log_samples

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] MMLU test completed" -ForegroundColor Green
} else {
    Write-Host "[NG] MMLU test failed" -ForegroundColor Red
}

# 結果確認
Write-Host "[RESULTS] Test results saved to: $lmEvalDir" -ForegroundColor Cyan
Get-ChildItem $lmEvalDir -Filter "*.json" | ForEach-Object {
    Write-Host "  - $($_.Name)" -ForegroundColor White
}

# 完了通知
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"


Write-Host "[START] Testing LM-Evaluation-Harness with Phi-3.5-mini-instruct..." -ForegroundColor Green

# 作業ディレクトリ設定
Set-Location $PSScriptRoot\..\..

# 出力ディレクトリ作成
$lmEvalDir = "H:\from_D\webdataset\benchmark_results\lm_eval"
if (!(Test-Path $lmEvalDir)) {
    New-Item -ItemType Directory -Path $lmEvalDir -Force
}

# テスト実行
Write-Host "[TEST] Testing hellaswag..." -ForegroundColor Yellow
py -m lm_eval --model hf `
    --model_args pretrained=microsoft/Phi-3.5-mini-instruct,dtype=bfloat16,trust_remote_code=True `
    --tasks hellaswag `
    --device cuda `
    --batch_size auto `
    --output_path "$lmEvalDir\phi35_hellaswag_test.json" `
    --log_samples

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] hellaswag test completed" -ForegroundColor Green
} else {
    Write-Host "[NG] hellaswag test failed" -ForegroundColor Red
}

Write-Host "[TEST] Testing MMLU..." -ForegroundColor Yellow
py -m lm_eval --model hf `
    --model_args pretrained=microsoft/Phi-3.5-mini-instruct,dtype=bfloat16,trust_remote_code=True `
    --tasks mmlu `
    --device cuda `
    --batch_size auto `
    --output_path "$lmEvalDir\phi35_mmlu_test.json" `
    --log_samples

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] MMLU test completed" -ForegroundColor Green
} else {
    Write-Host "[NG] MMLU test failed" -ForegroundColor Red
}

# 結果確認
Write-Host "[RESULTS] Test results saved to: $lmEvalDir" -ForegroundColor Cyan
Get-ChildItem $lmEvalDir -Filter "*.json" | ForEach-Object {
    Write-Host "  - $($_.Name)" -ForegroundColor White
}

# 完了通知
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"
