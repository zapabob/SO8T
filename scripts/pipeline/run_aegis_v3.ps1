<#
.SYNOPSIS
    AEGIS-v3.0 Main Pipeline Controller
.DESCRIPTION
    パイプラインを起動し、同時に進捗監視モニターを表示します。
#>

$projectRoot = Get-Location

# ログディレクトリの準備
if (-not (Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" }
$pipelineLog = "logs/aegis_v3_pipeline.log"
$sftProgressLog = "logs/sft_progress.log"

# 古いログをクリア
Clear-Content $pipelineLog -ErrorAction SilentlyContinue
Clear-Content $sftProgressLog -ErrorAction SilentlyContinue

Write-Host "Starting AEGIS-v3.0 Pipeline..." -ForegroundColor Green

# パイプラインをバックグラウンドで起動
$pipelineJob = Start-Job -ScriptBlock {
    param($root)
    Set-Location $root
    $env:SO8T_USE_UNSLOTH = "1"
    $env:SO8T_DRYRUN = "0"
    $env:SO8T_HF_UPLOAD = "1"
    $env:SO8T_COLLECT_ARXIV = "1"
    $env:SO8T_COLLECT_OSINT = "1"
    $env:SO8T_GRAPE_VARIANT = "multiplicative"
    $env:SO8T_CHECKPOINT_INTERVAL = "300"
    $env:SO8T_CHECKPOINT_ROLLING = "3"
    $env:SO8T_RESEARCH_TOPIC = "Advanced Mathematical Reasoning, Pharmacology, and Defense Intelligence"
    
    py -3 scripts/pipeline/auto_resume_aegis.py 2>&1 | Tee-Object -FilePath "logs/aegis_v3_pipeline.log"
} -ArgumentList $projectRoot

Write-Host "Pipeline Job ID: $($pipelineJob.Id)" -ForegroundColor Gray
Write-Host "Starting Progress Monitor..." -ForegroundColor Yellow
Write-Host ""

# 進捗監視スクリプトを実行 (現在のウィンドウで)
& "scripts/pipeline/monitor_training_progress.ps1" -LogPath $sftProgressLog

# パイプラインが終了するまで待機（モニターが終了された場合）
$jobStatus = Get-Job -Id $pipelineJob.Id
if ($jobStatus.State -eq "Running") {
    Write-Host "`nMonitoring stopped, but pipeline is still running in background." -ForegroundColor Cyan
    Write-Host "Use 'Receive-Job -Id $($pipelineJob.Id)' to see output." -ForegroundColor Gray
}
