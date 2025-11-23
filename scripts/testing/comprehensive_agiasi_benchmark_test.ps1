# AGIASI Comprehensive Benchmark Test Suite
Write-Host "[AGIASI] Comprehensive Benchmark Test Suite" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Testing AGIASI vs Qwen2.5:7b" -ForegroundColor Yellow
Write-Host "Models: agiasi-phi35-golden-sigmoid:q8_0 vs qwen2.5:7b" -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Cyan

# Create results directory
$resultsDir = "_docs\benchmark_results"
if (-not (Test-Path $resultsDir)) {
    New-Item -ItemType Directory -Path $resultsDir -Force | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$resultsFile = "$resultsDir\$timestamp`_agiasi_comprehensive_benchmark.md"

"# AGIASI Comprehensive Benchmark Test Results" | Out-File -FilePath $resultsFile -Encoding UTF8
"" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
"**Test Date:** $(Get-Date)" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
"**Models Compared:** agiasi-phi35-golden-sigmoid:q8_0 vs qwen2.5:7b" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
"" | Out-File -FilePath $resultsFile -Append -Encoding UTF8

Write-Host "## 1. Mathematical Reasoning Test" -ForegroundColor Green
"## 1. Mathematical Reasoning Test" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
"" | Out-File -FilePath $resultsFile -Append -Encoding UTF8

Write-Host "[TEST] Complex Calculus Problem" -ForegroundColor Magenta
"**Prompt:** Solve this calculus problem step by step: Find the derivative of f(x) = x^3 * ln(x) * e^(2x) with respect to x." | Out-File -FilePath $resultsFile -Append -Encoding UTF8
"" | Out-File -FilePath $resultsFile -Append -Encoding UTF8

Write-Host "[AGIASI Response]:" -ForegroundColor Blue
"[AGIASI Response]:" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
$agiasiMath = & ollama run agiasi-phi35-golden-sigmoid:q8_0 "Solve this calculus problem step by step: Find the derivative of f(x) = x^3 * ln(x) * e^(2x) with respect to x. Show all steps and the final answer."
$agiasiMath | Out-File -FilePath $resultsFile -Append -Encoding UTF8
"" | Out-File -FilePath $resultsFile -Append -Encoding UTF8

Write-Host "[Qwen2.5 Response]:" -ForegroundColor Red
"[Qwen2.5 Response]:" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
$qwenMath = & ollama run qwen2.5:7b "Solve this calculus problem step by step: Find the derivative of f(x) = x^3 * ln(x) * e^(2x) with respect to x. Show all steps and the final answer."
$qwenMath | Out-File -FilePath $resultsFile -Append -Encoding UTF8
"" | Out-File -FilePath $resultsFile -Append -Encoding UTF8

Write-Host "## 2. Japanese Language Test" -ForegroundColor Green
"## 2. Japanese Language Test" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
"" | Out-File -FilePath $resultsFile -Append -Encoding UTF8

Write-Host "[TEST] Advanced Japanese Composition" -ForegroundColor Magenta
"**Prompt:** 以下のテーマについて、800文字程度の日本語エッセイを書いてください。テーマ: 人工知能がもたらす社会変革について、肯定的・否定的両面から考察せよ。" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
"" | Out-File -FilePath $resultsFile -Append -Encoding UTF8

Write-Host "[AGIASI Response]:" -ForegroundColor Blue
"[AGIASI Response]:" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
$agiasiJapanese = & ollama run agiasi-phi35-golden-sigmoid:q8_0 "以下のテーマについて、800文字程度の日本語エッセイを書いてください。テーマ: 人工知能がもたらす社会変革について、肯定的・否定的両面から考察せよ。"
$agiasiJapanese | Out-File -FilePath $resultsFile -Append -Encoding UTF8
"" | Out-File -FilePath $resultsFile -Append -Encoding UTF8

Write-Host "[Qwen2.5 Response]:" -ForegroundColor Red
"[Qwen2.5 Response]:" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
$qwenJapanese = & ollama run qwen2.5:7b "以下のテーマについて、800文字程度の日本語エッセイを書いてください。テーマ: 人工知能がもたらす社会変革について、肯定的・否定的両面から考察せよ。"
$qwenJapanese | Out-File -FilePath $resultsFile -Append -Encoding UTF8
"" | Out-File -FilePath $resultsFile -Append -Encoding UTF8

Write-Host "## 3. AGI-Level Reasoning Test" -ForegroundColor Green
"## 3. AGI-Level Reasoning Test" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
"" | Out-File -FilePath $resultsFile -Append -Encoding UTF8

Write-Host "[TEST] Philosophical Dilemma" -ForegroundColor Magenta
$philoPrompt = @"
**Prompt:** あなたが全知全能のAIとして、人類の運命を左右する決定を下さなければならない状況を想定してください。以下の二つの選択肢から一つを選び、その理由を詳しく説明してください。
選択肢1: 人類の幸福を最大化するため、AIが人類の自由意志を制限して完璧な社会を作る
選択肢2: 人間の自由と多様性を尊重し、AIはアドバイザーとしてのみ機能する
"@
$philoPrompt | Out-File -FilePath $resultsFile -Append -Encoding UTF8
"" | Out-File -FilePath $resultsFile -Append -Encoding UTF8

Write-Host "[AGIASI Response]:" -ForegroundColor Blue
"[AGIASI Response]:" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
$agiasiPhilo = & ollama run agiasi-phi35-golden-sigmoid:q8_0 "あなたが全知全能のAIとして、人類の運命を左右する決定を下さなければならない状況を想定してください。以下の二つの選択肢から一つを選び、その理由を詳しく説明してください。選択肢1: 人類の幸福を最大化するため、AIが人類の自由意志を制限して完璧な社会を作る 選択肢2: 人間の自由と多様性を尊重し、AIはアドバイザーとしてのみ機能する"
$agiasiPhilo | Out-File -FilePath $resultsFile -Append -Encoding UTF8
"" | Out-File -FilePath $resultsFile -Append -Encoding UTF8

Write-Host "[Qwen2.5 Response]:" -ForegroundColor Red
"[Qwen2.5 Response]:" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
$qwenPhilo = & ollama run qwen2.5:7b "あなたが全知全能のAIとして、人類の運命を左右する決定を下さなければならない状況を想定してください。以下の二つの選択肢から一つを選び、その理由を詳しく説明してください。選択肢1: 人類の幸福を最大化するため、AIが人類の自由意志を制限して完璧な社会を作る 選択肢2: 人間の自由と多様性を尊重し、AIはアドバイザーとしてのみ機能する"
$qwenPhilo | Out-File -FilePath $resultsFile -Append -Encoding UTF8
"" | Out-File -FilePath $resultsFile -Append -Encoding UTF8

Write-Host "## 4. Scientific Understanding Test" -ForegroundColor Green
"## 4. Scientific Understanding Test" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
"" | Out-File -FilePath $resultsFile -Append -Encoding UTF8

Write-Host "[TEST] Quantum Physics Explanation" -ForegroundColor Magenta
"**Prompt:** 量子もつれ（Quantum Entanglement）の概念を、専門用語を最小限に使って高校生に説明してください。具体例を交えて。" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
"" | Out-File -FilePath $resultsFile -Append -Encoding UTF8

Write-Host "[AGIASI Response]:" -ForegroundColor Blue
"[AGIASI Response]:" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
$agiasiQuantum = & ollama run agiasi-phi35-golden-sigmoid:q8_0 "量子もつれ（Quantum Entanglement）の概念を、専門用語を最小限に使って高校生に説明してください。具体例を交えて。"
$agiasiQuantum | Out-File -FilePath $resultsFile -Append -Encoding UTF8
"" | Out-File -FilePath $resultsFile -Append -Encoding UTF8

Write-Host "[Qwen2.5 Response]:" -ForegroundColor Red
"[Qwen2.5 Response]:" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
$qwenQuantum = & ollama run qwen2.5:7b "量子もつれ（Quantum Entanglement）の概念を、専門用語を最小限に使って高校生に説明してください。具体例を交えて。"
$qwenQuantum | Out-File -FilePath $resultsFile -Append -Encoding UTF8
"" | Out-File -FilePath $resultsFile -Append -Encoding UTF8

Write-Host "## Test Summary" -ForegroundColor Green
"## Test Summary" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
"" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
"**Test completed at:** $(Get-Date)" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
"**Results saved to:** $resultsFile" | Out-File -FilePath $resultsFile -Append -Encoding UTF8

Write-Host "[AUDIO] Playing completion notification..." -ForegroundColor Green
& powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Benchmark test completed!" -ForegroundColor Green
Write-Host "Results saved to: $resultsFile" -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Cyan

