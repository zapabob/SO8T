# Wikipediaクローラー バックグラウンド実行スクリプト
# 
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\data\run_wikipedia_crawler_background.ps1 -Target 1000 -Output "D:\webdataset"

param(
    [int]$Target = 1000,
    [string]$Output = "D:\webdataset",
    [int]$Seed = 42
)

# プロジェクトルートに移動
$projectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
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
$logFile = Join-Path $logDir "wikipedia_crawler_background_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
$errorLogFile = Join-Path $logDir "wikipedia_crawler_background_errors_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Wikipedia Chromium Crawler" -ForegroundColor Cyan
Write-Host "Background Execution" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Target samples per domain: $Target" -ForegroundColor Yellow
Write-Host "Output directory: $Output" -ForegroundColor Yellow
Write-Host "Log file: $logFile" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan

# Pythonスクリプトをバックグラウンドで実行
$pythonScript = Join-Path $projectRoot "scripts\data\wikipedia_chromium_crawler.py"
$pythonArgs = @(
    "--output", $Output,
    "--target", $Target,
    "--seed", $Seed
)

# バックグラウンドジョブとして実行
$job = Start-Job -ScriptBlock {
    param($script, $args, $logFile, $errorLogFile)
    
    # プロジェクトルートに移動
    $projectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $script)))
    Set-Location $projectRoot
    
    # Python実行
    $process = Start-Process -FilePath "py" -ArgumentList @("-3", $script) + $args -NoNewWindow -PassThru -RedirectStandardOutput $logFile -RedirectStandardError $errorLogFile -Wait
    
    return $process.ExitCode
} -ArgumentList $pythonScript, $pythonArgs, $logFile, $errorLogFile

Write-Host "[INFO] Crawler started in background (Job ID: $($job.Id))" -ForegroundColor Green
Write-Host "[INFO] Monitor progress with: Get-Job -Id $($job.Id) | Receive-Job" -ForegroundColor Yellow
Write-Host "[INFO] View log file: $logFile" -ForegroundColor Yellow
Write-Host "[INFO] View error log: $errorLogFile" -ForegroundColor Yellow
Write-Host ""
Write-Host "To check status:" -ForegroundColor Cyan
Write-Host "  Get-Job -Id $($job.Id)" -ForegroundColor White
Write-Host ""
Write-Host "To view output:" -ForegroundColor Cyan
Write-Host "  Get-Job -Id $($job.Id) | Receive-Job" -ForegroundColor White
Write-Host ""
Write-Host "To stop:" -ForegroundColor Cyan
Write-Host "  Stop-Job -Id $($job.Id)" -ForegroundColor White
Write-Host "  Remove-Job -Id $($job.Id)" -ForegroundColor White

# ジョブ情報を保存
$jobInfoFile = Join-Path $Output "crawler_job_info.json"
$jobInfo = @{
    job_id = $job.Id
    start_time = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    target_samples = $Target
    output_dir = $Output
    log_file = $logFile
    error_log_file = $errorLogFile
} | ConvertTo-Json

$jobInfo | Out-File -FilePath $jobInfoFile -Encoding UTF8

Write-Host "[OK] Job info saved to: $jobInfoFile" -ForegroundColor Green





















