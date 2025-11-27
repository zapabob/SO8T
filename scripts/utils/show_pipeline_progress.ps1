#!/usr/bin/env powershell
# -*- coding: utf-8 -*-
<#
.SYNOPSIS
    AEGIS v2.0パイプラインとベイズ最適化の進捗状況を表示

.DESCRIPTION
    パイプラインの実行状況、ベイズ最適化の進捗、ログの最新情報を表示します。
#>

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "AEGIS v2.0 Pipeline & Bayesian Optimization Progress" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# 1. パイプラインの実行状況
Write-Host "[1] Pipeline Status" -ForegroundColor Yellow
Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Gray

$pipelineLog = "logs\aegis_v2_pipeline.log"
if (Test-Path $pipelineLog) {
    $pipelineLines = Get-Content $pipelineLog -Tail 20
    Write-Host "Latest Pipeline Log (last 20 lines):" -ForegroundColor Green
    $pipelineLines | ForEach-Object { Write-Host $_ }
} else {
    Write-Host "[WARNING] Pipeline log not found: $pipelineLog" -ForegroundColor Yellow
}

Write-Host ""

# 2. トレーニングログの最新情報
Write-Host "[2] Training Status" -ForegroundColor Yellow
Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Gray

$trainingLog = "logs\train_so8t_quadruple_ppo.log"
if (Test-Path $trainingLog) {
    $trainingLines = Get-Content $trainingLog -Tail 30
    Write-Host "Latest Training Log (last 30 lines):" -ForegroundColor Green
    
    # 重要な情報をハイライト
    $trainingLines | ForEach-Object {
        $line = $_
        if ($line -match "ERROR|FAILED|CRITICAL") {
            Write-Host $line -ForegroundColor Red
        } elseif ($line -match "WARNING|WARN") {
            Write-Host $line -ForegroundColor Yellow
        } elseif ($line -match "SUCCESS|OK|completed|Initialized") {
            Write-Host $line -ForegroundColor Green
        } elseif ($line -match "Alpha Gate|ALPHA|orthogonal|intermediate|rotation gate|PET") {
            Write-Host $line -ForegroundColor Cyan
        } elseif ($line -match "Step|STEP") {
            Write-Host $line -ForegroundColor Magenta
        } else {
            Write-Host $line
        }
    }
} else {
    Write-Host "[WARNING] Training log not found: $trainingLog" -ForegroundColor Yellow
}

Write-Host ""

# 3. ベイズ最適化の進捗
Write-Host "[3] Bayesian Optimization Status" -ForegroundColor Yellow
Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Gray

$bayesOptResult = "D:\webdataset\alpha_gate_bayes_opt\optimal_alpha_gate_orthogonal.json"
if (Test-Path $bayesOptResult) {
    Write-Host "[OK] Optimization completed!" -ForegroundColor Green
    try {
        $result = Get-Content $bayesOptResult -Raw | ConvertFrom-Json
        Write-Host "Best Objective Value: $($result.best_objective)" -ForegroundColor Cyan
        Write-Host "Best Parameters:" -ForegroundColor Cyan
        $result.best_params.PSObject.Properties | ForEach-Object {
            Write-Host "  $($_.Name): $($_.Value)" -ForegroundColor White
        }
        Write-Host ""
        Write-Host "Best Trial Metrics:" -ForegroundColor Cyan
        if ($result.best_trial_metrics) {
            $result.best_trial_metrics.PSObject.Properties | ForEach-Object {
                Write-Host "  $($_.Name): $($_.Value)" -ForegroundColor White
            }
        }
    } catch {
        Write-Host "[ERROR] Failed to parse optimization results: $_" -ForegroundColor Red
    }
} else {
    Write-Host "[INFO] Optimization still running..." -ForegroundColor Yellow
    
    # Optuna studyの進捗を確認（study.dbがある場合）
    $studyDbPath = "D:\webdataset\alpha_gate_bayes_opt"
    if (Test-Path $studyDbPath) {
        $studyFiles = Get-ChildItem -Path $studyDbPath -Filter "*.db" -ErrorAction SilentlyContinue
        if ($studyFiles) {
            Write-Host "[INFO] Found Optuna study database files" -ForegroundColor Yellow
            $studyFiles | ForEach-Object {
                $sizeKB = [math]::Round($_.Length / 1024, 2)
                Write-Host "  Study DB: $($_.Name) ($sizeKB KB)" -ForegroundColor Gray
            }
        }
    }
}

Write-Host ""

# 4. プロセス状況
Write-Host "[4] Running Processes" -ForegroundColor Yellow
Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Gray

$pythonProcesses = Get-Process python -ErrorAction SilentlyContinue | Where-Object {
    $_.Path -like "*python*"
}

if ($pythonProcesses) {
    Write-Host "[OK] Found $($pythonProcesses.Count) Python process(es):" -ForegroundColor Green
    $pythonProcesses | Select-Object -First 5 | ForEach-Object {
        $runtime = (Get-Date) - $_.StartTime
        Write-Host "  PID: $($_.Id) | Runtime: $($runtime.ToString('hh\:mm\:ss')) | CPU: $([math]::Round($_.CPU, 2))s" -ForegroundColor White
    }
} else {
    Write-Host "[INFO] No Python processes found" -ForegroundColor Yellow
}

Write-Host ""

# 5. チェックポイント状況
Write-Host "[5] Checkpoint Status" -ForegroundColor Yellow
Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Gray

$checkpointDir = "D:\webdataset\aegis_v2.0\so8t_ppo_model"
if (Test-Path $checkpointDir) {
    $checkpoints = Get-ChildItem -Path $checkpointDir -Recurse -Filter "*.pt" -ErrorAction SilentlyContinue
    if ($checkpoints) {
        Write-Host "[OK] Found $($checkpoints.Count) checkpoint(s):" -ForegroundColor Green
        $checkpoints | Sort-Object LastWriteTime -Descending | Select-Object -First 5 | ForEach-Object {
            $sizeMB = [math]::Round($_.Length / 1MB, 2)
            Write-Host "  $($_.Name) | Size: $sizeMB MB | Modified: $($_.LastWriteTime)" -ForegroundColor White
        }
    } else {
        Write-Host "[INFO] No checkpoints found yet" -ForegroundColor Yellow
    }
} else {
    Write-Host "[INFO] Checkpoint directory not found: $checkpointDir" -ForegroundColor Yellow
}

Write-Host ""

# 6. GPU/メモリ使用状況
Write-Host "[6] System Resources" -ForegroundColor Yellow
Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Gray

# CPU使用率
try {
    $cpu = Get-Counter '\Processor(_Total)\% Processor Time' -ErrorAction SilentlyContinue
    if ($cpu) {
        $cpuUsage = [math]::Round($cpu.CounterSamples[0].CookedValue, 2)
        Write-Host "CPU Usage: $cpuUsage%" -ForegroundColor White
    }
} catch {
    Write-Host "CPU Usage: Unable to retrieve" -ForegroundColor Gray
}

# メモリ使用状況
try {
    $mem = Get-CimInstance Win32_OperatingSystem
    $totalMem = [math]::Round($mem.TotalVisibleMemorySize / 1MB, 2)
    $freeMem = [math]::Round($mem.FreePhysicalMemory / 1MB, 2)
    $usedMem = $totalMem - $freeMem
    $memPercent = [math]::Round(($usedMem / $totalMem) * 100, 2)
    Write-Host "Memory: $usedMem GB / $totalMem GB ($memPercent%)" -ForegroundColor White
} catch {
    Write-Host "Memory: Unable to retrieve" -ForegroundColor Gray
}

# GPU使用状況（nvidia-smiがある場合）
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    Write-Host ""
    Write-Host "GPU Status:" -ForegroundColor Cyan
    try {
        $gpuInfo = nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>$null
        if ($gpuInfo) {
            $gpuInfo | ForEach-Object {
                $parts = $_ -split ','
                if ($parts.Count -ge 4) {
                    Write-Host "  $($parts[0].Trim()): $($parts[1].Trim()) / $($parts[2].Trim()) | GPU: $($parts[3].Trim())" -ForegroundColor White
                }
            }
        }
    } catch {
        Write-Host "  Unable to retrieve GPU info" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Write-Host "Progress check completed at $timestamp" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
