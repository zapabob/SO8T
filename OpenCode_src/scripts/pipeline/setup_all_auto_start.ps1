# SO8T全自動パイプライン 統合セットアップスクリプト (PowerShell版)
# master_automated_pipeline と parallel_pipeline_manager の両方をセットアップ

# 管理者権限チェック
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "[INFO] This script requires administrator privileges." -ForegroundColor Yellow
    Write-Host "[INFO] Restarting with administrator privileges..." -ForegroundColor Yellow
    Write-Host ""
    
    # 管理者権限で再起動
    $scriptPath = $MyInvocation.MyCommand.Path
    Start-Process powershell.exe -Verb RunAs -ArgumentList "-ExecutionPolicy Bypass -File `"$scriptPath`""
    exit
}

# プロジェクトルートパス
$PROJECT_ROOT = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SO8T All Auto-Start Pipeline Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Project Root: $PROJECT_ROOT" -ForegroundColor Green
Write-Host ""

# Python実行ファイルの検出
$pythonCmd = Get-Command py -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[ERROR] Python not found in PATH" -ForegroundColor Red
    Write-Host "[ERROR] Please install Python or add it to PATH" -ForegroundColor Red
    pause
    exit 1
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "[STEP 1] Setting up Master Automated Pipeline" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$masterSetupScript = Join-Path $PROJECT_ROOT "scripts\pipelines\setup_master_automated_pipeline.py"
& py -3 $masterSetupScript

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Master Automated Pipeline setup failed" -ForegroundColor Red
    pause
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "[STEP 2] Setting up Parallel Pipeline Manager" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$parallelSetupScript = Join-Path $PROJECT_ROOT "scripts\data\setup_parallel_pipeline_manager.py"
& py -3 $parallelSetupScript

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Parallel Pipeline Manager setup failed" -ForegroundColor Red
    pause
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "[STEP 3] Verifying All Tasks" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[INFO] Checking Master Automated Pipeline task..." -ForegroundColor Yellow
$masterTask = schtasks /query /tn "SO8T-MasterAutomatedPipeline-AutoStart" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARNING] Master Automated Pipeline task not found" -ForegroundColor Yellow
} else {
    Write-Host "[OK] Master Automated Pipeline task is registered" -ForegroundColor Green
}

Write-Host "[INFO] Checking Parallel Pipeline Manager task..." -ForegroundColor Yellow
$parallelTask = schtasks /query /tn "SO8T-ParallelPipelineManager-AutoStart" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARNING] Parallel Pipeline Manager task not found" -ForegroundColor Yellow
} else {
    Write-Host "[OK] Parallel Pipeline Manager task is registered" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "[SUCCESS] All Auto-Start Setup Completed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Both pipelines will automatically run on system startup:" -ForegroundColor Cyan
Write-Host "  - Master Automated Pipeline (SO8T-MasterAutomatedPipeline-AutoStart)" -ForegroundColor Cyan
Write-Host "  - Parallel Pipeline Manager (SO8T-ParallelPipelineManager-AutoStart)" -ForegroundColor Cyan
Write-Host ""
Write-Host "To test manually:" -ForegroundColor Cyan
Write-Host "  Master Pipeline: py -3 `"$masterSetupScript`" --run" -ForegroundColor Cyan
Write-Host "  Parallel Manager: py -3 `"$parallelSetupScript`" --run --daemon" -ForegroundColor Cyan
Write-Host ""
pause

