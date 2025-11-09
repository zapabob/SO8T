# Wikipediaクローラー バックグラウンド実行スクリプト（改善版）
# 
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\data\start_crawler_background.ps1

param(
    [int]$Target = 1000,
    [string]$Output = "D:\webdataset",
    [int]$Seed = 42
)

# プロジェクトルートに移動
$scriptPath = $PSScriptRoot
$projectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $scriptPath))
Set-Location $projectRoot

# ログディレクトリ作成
$logDir = Join-Path $projectRoot "logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

# 出力ディレクトリ作成
if (-not (Test-Path $Output)) {
    New-Item -ItemType Directory -Path $Output -Force | Out-Null
}

# ログファイル
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logFile = Join-Path $logDir "wikipedia_crawler_$timestamp.log"
$errorLogFile = Join-Path $logDir "wikipedia_crawler_errors_$timestamp.log"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Wikipedia Chromium Crawler" -ForegroundColor Cyan
Write-Host "Background Execution" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Target samples per domain: $Target" -ForegroundColor Yellow
Write-Host "Output directory: $Output" -ForegroundColor Yellow
Write-Host "Log file: $logFile" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan

# Pythonスクリプトパス
$pythonScript = Join-Path $projectRoot "scripts\data\wikipedia_chromium_crawler.py"

# バックグラウンドプロセスとして実行
$processArgs = @(
    "-3",
    $pythonScript,
    "--output", $Output,
    "--target", $Target,
    "--seed", $Seed
)

Write-Host "[INFO] Starting crawler in background..." -ForegroundColor Green

# Start-Processでバックグラウンド実行
$process = Start-Process -FilePath "py" `
    -ArgumentList $processArgs `
    -NoNewWindow `
    -PassThru `
    -RedirectStandardOutput $logFile `
    -RedirectStandardError $errorLogFile

Write-Host "[OK] Crawler started (PID: $($process.Id))" -ForegroundColor Green
Write-Host "[INFO] Monitor log file: $logFile" -ForegroundColor Yellow
Write-Host "[INFO] Monitor error log: $errorLogFile" -ForegroundColor Yellow
Write-Host ""
Write-Host "To check process status:" -ForegroundColor Cyan
Write-Host "  Get-Process -Id $($process.Id)" -ForegroundColor White
Write-Host ""
Write-Host "To view log (tail):" -ForegroundColor Cyan
Write-Host "  Get-Content $logFile -Tail 50 -Wait" -ForegroundColor White
Write-Host ""
Write-Host "To stop:" -ForegroundColor Cyan
Write-Host "  Stop-Process -Id $($process.Id)" -ForegroundColor White
Write-Host ""

# プロセス情報を保存
$processInfoFile = Join-Path $Output "crawler_process_info.json"
$processInfo = @{
    process_id = $process.Id
    start_time = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    target_samples = $Target
    output_dir = $Output
    log_file = $logFile
    error_log_file = $errorLogFile
    python_script = $pythonScript
} | ConvertTo-Json

$processInfo | Out-File -FilePath $processInfoFile -Encoding UTF8

Write-Host "[OK] Process info saved to: $processInfoFile" -ForegroundColor Green





















