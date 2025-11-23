# Quick Benchmark Test for AGIASI vs Qwen2.5:7b
Write-Host "[QUICK TEST] AGIASI vs Qwen2.5 Benchmark" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# Create results directory
$resultsDir = "_docs\benchmark_results"
if (-not (Test-Path $resultsDir)) {
    New-Item -ItemType Directory -Path $resultsDir -Force | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$resultsFile = "$resultsDir\$timestamp`_quick_benchmark.md"

"# Quick Benchmark Test Results - AGIASI vs Qwen2.5:7b" | Out-File -FilePath $resultsFile -Encoding UTF8
"**Test Date:** $(Get-Date)" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
"" | Out-File -FilePath $resultsFile -Append -Encoding UTF8

# Test 1: Simple Math
Write-Host "[TEST 1] Simple Math: 15 + 27 = ?" -ForegroundColor Yellow
"## Test 1: Simple Math" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
"**Question:** What is 15 + 27?" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
"" | Out-File -FilePath $resultsFile -Append -Encoding UTF8

Write-Host "AGIASI:" -ForegroundColor Blue
"[AGIASI Response]:" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
try {
    $result = & ollama run agiasi-phi35-golden-sigmoid:q8_0 "What is 15 + 27? Answer with just the number." 2>$null
    Write-Host $result -ForegroundColor White
    $result | Out-File -FilePath $resultsFile -Append -Encoding UTF8
} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    "ERROR: $($_.Exception.Message)" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
}
"" | Out-File -FilePath $resultsFile -Append -Encoding UTF8

Write-Host "Qwen2.5:" -ForegroundColor Red
"[Qwen2.5 Response]:" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
try {
    $result = & ollama run qwen2.5:7b "What is 15 + 27? Answer with just the number." 2>$null
    Write-Host $result -ForegroundColor White
    $result | Out-File -FilePath $resultsFile -Append -Encoding UTF8
} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    "ERROR: $($_.Exception.Message)" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
}
"" | Out-File -FilePath $resultsFile -Append -Encoding UTF8

# Test 2: Simple Reasoning
Write-Host "[TEST 2] Simple Reasoning" -ForegroundColor Yellow
"## Test 2: Simple Reasoning" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
"**Question:** If all cats are mammals and some mammals are pets, are all cats pets? Explain briefly." | Out-File -FilePath $resultsFile -Append -Encoding UTF8
"" | Out-File -FilePath $resultsFile -Append -Encoding UTF8

Write-Host "AGIASI:" -ForegroundColor Blue
"[AGIASI Response]:" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
try {
    $result = & ollama run agiasi-phi35-golden-sigmoid:q8_0 "If all cats are mammals and some mammals are pets, are all cats pets? Explain briefly." 2>$null
    Write-Host $result -ForegroundColor White
    $result | Out-File -FilePath $resultsFile -Append -Encoding UTF8
} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    "ERROR: $($_.Exception.Message)" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
}
"" | Out-File -FilePath $resultsFile -Append -Encoding UTF8

Write-Host "Qwen2.5:" -ForegroundColor Red
"[Qwen2.5 Response]:" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
try {
    $result = & ollama run qwen2.5:7b "If all cats are mammals and some mammals are pets, are all cats pets? Explain briefly." 2>$null
    Write-Host $result -ForegroundColor White
    $result | Out-File -FilePath $resultsFile -Append -Encoding UTF8
} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    "ERROR: $($_.Exception.Message)" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
}
"" | Out-File -FilePath $resultsFile -Append -Encoding UTF8

# Test 3: Creative Task
Write-Host "[TEST 3] Creative Task" -ForegroundColor Yellow
"## Test 3: Creative Task" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
"**Question:** Name three unusual uses for a paperclip." | Out-File -FilePath $resultsFile -Append -Encoding UTF8
"" | Out-File -FilePath $resultsFile -Append -Encoding UTF8

Write-Host "AGIASI:" -ForegroundColor Blue
"[AGIASI Response]:" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
try {
    $result = & ollama run agiasi-phi35-golden-sigmoid:q8_0 "Name three unusual uses for a paperclip." 2>$null
    Write-Host $result -ForegroundColor White
    $result | Out-File -FilePath $resultsFile -Append -Encoding UTF8
} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    "ERROR: $($_.Exception.Message)" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
}
"" | Out-File -FilePath $resultsFile -Append -Encoding UTF8

Write-Host "Qwen2.5:" -ForegroundColor Red
"[Qwen2.5 Response]:" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
try {
    $result = & ollama run qwen2.5:7b "Name three unusual uses for a paperclip." 2>$null
    Write-Host $result -ForegroundColor White
    $result | Out-File -FilePath $resultsFile -Append -Encoding UTF8
} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    "ERROR: $($_.Exception.Message)" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
}
"" | Out-File -FilePath $resultsFile -Append -Encoding UTF8

# Summary
Write-Host "[SUMMARY]" -ForegroundColor Green
"## Summary" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
"**Test completed at:** $(Get-Date)" | Out-File -FilePath $resultsFile -Append -Encoding UTF8
"**Results saved to:** $resultsFile" | Out-File -FilePath $resultsFile -Append -Encoding UTF8

Write-Host "Results saved to: $resultsFile" -ForegroundColor Yellow
Write-Host "======================================" -ForegroundColor Cyan

# Play notification
Write-Host "[AUDIO] Playing completion notification..." -ForegroundColor Green
& powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"
