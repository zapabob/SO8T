# SO8T包括的テストスイート実行 PowerShell版
# UTF-8エンコーディングで実行

param(
    [string]$TestType = "all",
    [switch]$Verbose = $false,
    [switch]$GenerateReport = $true,
    [string]$OutputDir = "_docs\test_logs"
)

# エンコーディング設定
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "SO8T包括的テストスイート実行 (PowerShell版)" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# 環境設定
$env:PYTHONPATH = Get-Location
$env:CUDA_VISIBLE_DEVICES = "0"

# ログディレクトリの作成
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

# タイムスタンプの取得
$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
Write-Host "[INFO] テスト開始時刻: $timestamp" -ForegroundColor Green
Write-Host ""

# テスト結果ファイルの設定
$testResults = Join-Path $OutputDir "comprehensive_test_results_$timestamp.json"
$testLog = Join-Path $OutputDir "comprehensive_test_log_$timestamp.log"
$testSummary = Join-Path $OutputDir "test_summary_$timestamp.txt"

# テスト実行関数
function Invoke-Test {
    param(
        [string]$TestName,
        [string]$TestFile,
        [string]$TestDescription
    )
    
    Write-Host "[TEST] $TestName 開始..." -ForegroundColor Yellow
    Write-Host "[DESC] $TestDescription" -ForegroundColor Gray
    Write-Host ""
    
    # テスト実行
    $testCommand = "python -m pytest $TestFile -v --tb=short --json-report --json-report-file=$testResults"
    
    if ($Verbose) {
        $testCommand += " -s"
    }
    
    try {
        # テスト実行とログ出力
        Invoke-Expression $testCommand 2>&1 | Tee-Object -FilePath $testLog -Append
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] $TestName 成功" -ForegroundColor Green
            "$TestName`:SUCCESS" | Add-Content -Path $testSummary
            return $true
        } else {
            Write-Host "[NG] $TestName 失敗" -ForegroundColor Red
            "$TestName`:FAILED" | Add-Content -Path $testSummary
            return $false
        }
    } catch {
        Write-Host "[ERROR] $TestName 実行エラー: $($_.Exception.Message)" -ForegroundColor Red
        "$TestName`:ERROR" | Add-Content -Path $testSummary
        return $false
    }
    
    Write-Host ""
}

# テスト定義
$testSuites = @{
    "SO8_Operations" = @{
        "File" = "tests\test_so8_operations_comprehensive.py"
        "Description" = "SO(8)群構造の数学的性質検証"
        "Category" = "unit"
    }
    "PyTorch_Comparison" = @{
        "File" = "tests\test_pytorch_comparison.py"
        "Description" = "PyTorchモデルとの精度比較"
        "Category" = "comparison"
    }
    "Quantization" = @{
        "File" = "tests\test_so8t_quantization.py"
        "Description" = "SO8Tモデル量子化機能検証"
        "Category" = "quantization"
    }
    "Existing_Tests" = @{
        "File" = "tests\"
        "Description" = "既存のテストスイート実行"
        "Category" = "integration"
    }
}

# テスト実行
$testResults = @{}
$totalTests = 0
$successfulTests = 0
$failedTests = 0

Write-Host "[PHASE 1] SO(8)演算ユニットテスト" -ForegroundColor Magenta
if ($TestType -eq "all" -or $TestType -eq "unit") {
    $result = Invoke-Test -TestName "SO8_Operations" -TestFile $testSuites["SO8_Operations"]["File"] -TestDescription $testSuites["SO8_Operations"]["Description"]
    $testResults["SO8_Operations"] = $result
    $totalTests++
    if ($result) { $successfulTests++ } else { $failedTests++ }
}

Write-Host "[PHASE 2] PyTorch比較テスト" -ForegroundColor Magenta
if ($TestType -eq "all" -or $TestType -eq "comparison") {
    $result = Invoke-Test -TestName "PyTorch_Comparison" -TestFile $testSuites["PyTorch_Comparison"]["File"] -TestDescription $testSuites["PyTorch_Comparison"]["Description"]
    $testResults["PyTorch_Comparison"] = $result
    $totalTests++
    if ($result) { $successfulTests++ } else { $failedTests++ }
}

Write-Host "[PHASE 3] 量子化テスト" -ForegroundColor Magenta
if ($TestType -eq "all" -or $TestType -eq "quantization") {
    $result = Invoke-Test -TestName "Quantization" -TestFile $testSuites["Quantization"]["File"] -TestDescription $testSuites["Quantization"]["Description"]
    $testResults["Quantization"] = $result
    $totalTests++
    if ($result) { $successfulTests++ } else { $failedTests++ }
}

Write-Host "[PHASE 4] 既存テストスイート" -ForegroundColor Magenta
if ($TestType -eq "all" -or $TestType -eq "integration") {
    $result = Invoke-Test -TestName "Existing_Tests" -TestFile $testSuites["Existing_Tests"]["File"] -TestDescription $testSuites["Existing_Tests"]["Description"]
    $testResults["Existing_Tests"] = $result
    $totalTests++
    if ($result) { $successfulTests++ } else { $failedTests++ }
}

# テスト結果の集計
Write-Host "[SUMMARY] テスト結果集計中..." -ForegroundColor Cyan
Write-Host ""

$successRate = if ($totalTests -gt 0) { [math]::Round(($successfulTests / $totalTests) * 100, 1) } else { 0 }

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "テスト結果サマリー" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "総テスト数: $totalTests" -ForegroundColor White
Write-Host "成功: $successfulTests" -ForegroundColor Green
Write-Host "失敗: $failedTests" -ForegroundColor Red
Write-Host "成功率: $successRate%" -ForegroundColor Yellow
Write-Host ""

# 詳細結果の表示
Write-Host "詳細結果:" -ForegroundColor Cyan
foreach ($testName in $testResults.Keys) {
    $status = if ($testResults[$testName]) { "SUCCESS" } else { "FAILED" }
    $color = if ($testResults[$testName]) { "Green" } else { "Red" }
    Write-Host "  $testName`: $status" -ForegroundColor $color
}
Write-Host ""

# 結果ファイルの作成
if ($GenerateReport) {
    $finalResults = @{
        "timestamp" = $timestamp
        "total_tests" = $totalTests
        "successful_tests" = $successfulTests
        "failed_tests" = $failedTests
        "success_rate" = $successRate
        "test_log_file" = $testLog
        "test_results_file" = $testResults
        "test_summary_file" = $testSummary
        "test_type" = $TestType
        "environment" = @{
            "python_version" = (python --version 2>&1)
            "pytorch_version" = (python -c "import torch; print(torch.__version__)" 2>&1)
            "cuda_available" = (python -c "import torch; print(torch.cuda.is_available())" 2>&1)
        }
    }
    
    $finalResultsJson = $finalResults | ConvertTo-Json -Depth 10
    $finalResultsFile = Join-Path $OutputDir "final_results_$timestamp.json"
    $finalResultsJson | Out-File -FilePath $finalResultsFile -Encoding UTF8
    
    Write-Host "結果ファイル: $finalResultsFile" -ForegroundColor Cyan
}

Write-Host "ログファイル: $testLog" -ForegroundColor Cyan
Write-Host "サマリーファイル: $testSummary" -ForegroundColor Cyan

# 最終結果の表示
if ($failedTests -eq 0) {
    Write-Host "[SUCCESS] 全てのテストが成功しました！" -ForegroundColor Green
    $finalStatus = "SUCCESS"
} else {
    Write-Host "[WARNING] $failedTests 個のテストが失敗しました" -ForegroundColor Yellow
    $finalStatus = "WARNING"
}

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "テスト完了" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "最終ステータス: $finalStatus" -ForegroundColor $(if ($finalStatus -eq "SUCCESS") { "Green" } else { "Yellow" })
Write-Host ""

# 音声通知の再生
Write-Host "[AUDIO] 完了通知を再生中..." -ForegroundColor Magenta
$audioFile = "C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav"

if (Test-Path $audioFile) {
    try {
        Add-Type -AssemblyName System.Windows.Forms
        $player = New-Object System.Media.SoundPlayer $audioFile
        $player.Play()
        Write-Host "[OK] 音声通知再生完了" -ForegroundColor Green
    } catch {
        Write-Host "[WARNING] 音声再生に失敗しました: $($_.Exception.Message)" -ForegroundColor Yellow
    }
} else {
    Write-Host "[WARNING] 音声ファイルが見つかりません: $audioFile" -ForegroundColor Yellow
}

# 終了コードの設定
if ($failedTests -eq 0) {
    exit 0
} else {
    exit 1
}
