<# 
.SYNOPSIS
    AEGIS パイプライン自動実行・進捗管理スクリプト（PowerShell版）

.DESCRIPTION
    Phase 4-6 パイプラインを5分間隔ローリングチェックポイント（3個）で実行し、
    電源投入時に自動再開する機能を提供します。

.NOTES
    - 5分（300秒）ごとにチェックポイントを保存
    - 最大3個のローリングストック
    - Windows スタートアップへの登録で電源投入時自動再開
#>

param(
    [switch]$Install,
    [switch]$Uninstall,
    [switch]$Resume,
    [string]$Phase = "all"
)

$ErrorActionPreference = "Continue"

# Configuration
$CHECKPOINT_INTERVAL_SEC = 300  # 5 minutes
$ROLLING_COUNT = 3
$PROJECT_ROOT = Split-Path -Parent $PSScriptRoot
if (-not $PROJECT_ROOT) { $PROJECT_ROOT = "C:\Users\downl\Desktop\SO8T" }
$CHECKPOINT_DIR = Join-Path $PROJECT_ROOT "checkpoints\powershell"
$LOG_DIR = Join-Path $PROJECT_ROOT "logs"
$PID_FILE = Join-Path $CHECKPOINT_DIR "pipeline.pid"
$STATE_FILE = Join-Path $CHECKPOINT_DIR "pipeline_state.json"
$STARTUP_SCRIPT = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Startup\aegis_auto_resume.bat"

# Ensure directories exist
if (-not (Test-Path $CHECKPOINT_DIR)) { New-Item -ItemType Directory -Path $CHECKPOINT_DIR -Force | Out-Null }
if (-not (Test-Path $LOG_DIR)) { New-Item -ItemType Directory -Path $LOG_DIR -Force | Out-Null }

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logLine = "$timestamp [$Level] $Message"
    Write-Host $logLine
    Add-Content -Path (Join-Path $LOG_DIR "aegis_pipeline.log") -Value $logLine -Encoding UTF8
}

function Save-Checkpoint {
    param([hashtable]$State)
    
    # Determine checkpoint index
    $pointerFile = Join-Path $CHECKPOINT_DIR "latest_checkpoint.ptr"
    $currentIndex = 0
    if (Test-Path $pointerFile) {
        $currentIndex = [int](Get-Content $pointerFile -Raw).Trim()
    }
    $nextIndex = ($currentIndex + 1) % $ROLLING_COUNT
    
    # Save state
    $checkpointPath = Join-Path $CHECKPOINT_DIR "checkpoint_$nextIndex.json"
    $State["timestamp"] = (Get-Date).ToString("o")
    $State["checkpoint_index"] = $nextIndex
    $State | ConvertTo-Json -Depth 10 | Set-Content -Path $checkpointPath -Encoding UTF8
    
    # Update pointer
    $nextIndex | Out-File -FilePath $pointerFile -Encoding UTF8 -NoNewline
    
    Write-Log "Saved checkpoint $nextIndex at step $($State.step)"
    return $checkpointPath
}

function Load-Checkpoint {
    $pointerFile = Join-Path $CHECKPOINT_DIR "latest_checkpoint.ptr"
    if (-not (Test-Path $pointerFile)) {
        Write-Log "No checkpoint found. Starting fresh." "WARN"
        return $null
    }
    
    $index = [int](Get-Content $pointerFile -Raw).Trim()
    $checkpointPath = Join-Path $CHECKPOINT_DIR "checkpoint_$index.json"
    
    if (Test-Path $checkpointPath) {
        $state = Get-Content $checkpointPath -Raw | ConvertFrom-Json -AsHashtable
        Write-Log "Loaded checkpoint $index from step $($state.step)"
        return $state
    }
    
    Write-Log "Checkpoint file not found: $checkpointPath" "WARN"
    return $null
}

function Show-Progress {
    param(
        [int]$Current,
        [int]$Total,
        [string]$Activity,
        [string]$Status
    )
    
    $percent = [math]::Min(100, [math]::Floor(($Current / [math]::Max(1, $Total)) * 100))
    $barLength = 50
    $filled = [math]::Floor($barLength * $percent / 100)
    $empty = $barLength - $filled
    $bar = ("█" * $filled) + ("░" * $empty)
    
    Write-Host -NoNewline "`r[$bar] $percent% | $Activity | $Status    "
}

function Run-Phase4 {
    param([hashtable]$State)
    
    Write-Log "Starting Phase 4: Data Enrichment"
    Show-Progress -Current 1 -Total 6 -Activity "Phase 4" -Status "Running data enrichment..."
    
    try {
        $pythonCmd = "py -3 -c `"from src.data.phase4_data_enrichment_pipeline import Phase4DataEnrichmentPipeline; p = Phase4DataEnrichmentPipeline(); p.run()`""
        $result = Invoke-Expression $pythonCmd 2>&1
        $State["phase4_complete"] = $true
        $State["phase4_result"] = "Success"
        Write-Log "Phase 4 completed successfully"
    }
    catch {
        $State["phase4_complete"] = $false
        $State["phase4_error"] = $_.Exception.Message
        Write-Log "Phase 4 failed: $($_.Exception.Message)" "ERROR"
    }
    
    return $State
}

function Run-WorldEvents {
    param([hashtable]$State)
    
    Write-Log "Collecting 2024-2026 World Events Data"
    Show-Progress -Current 2 -Total 6 -Activity "World Events" -Status "Collecting geopolitics, tech, culture..."
    
    try {
        $pythonCmd = "py -3 -c `"from src.data.collect_world_events_2024_2026 import WorldEvents2024_2026Collector; c = WorldEvents2024_2026Collector(); c.run()`""
        $result = Invoke-Expression $pythonCmd 2>&1
        $State["world_events_complete"] = $true
        Write-Log "World Events collection completed"
    }
    catch {
        $State["world_events_complete"] = $false
        $State["world_events_error"] = $_.Exception.Message
        Write-Log "World Events collection failed: $($_.Exception.Message)" "ERROR"
    }
    
    return $State
}

function Run-Phase5 {
    param([hashtable]$State)
    
    Write-Log "Starting Phase 5: Auto-Retraining"
    Show-Progress -Current 3 -Total 6 -Activity "Phase 5" -Status "Running SFT + GRPO training..."
    
    try {
        $pythonCmd = "py -3 -c `"from src.training.phase5_auto_retraining_pipeline import Phase5AutoRetrainingPipeline; p = Phase5AutoRetrainingPipeline(); p.run()`""
        $result = Invoke-Expression $pythonCmd 2>&1
        $State["phase5_complete"] = $true
        $State["phase5_result"] = "Success"
        Write-Log "Phase 5 completed successfully"
    }
    catch {
        $State["phase5_complete"] = $false
        $State["phase5_error"] = $_.Exception.Message
        Write-Log "Phase 5 failed: $($_.Exception.Message)" "ERROR"
    }
    
    return $State
}

function Run-Phase6 {
    param([hashtable]$State)
    
    Write-Log "Starting Phase 6: Statistical Benchmark"
    Show-Progress -Current 5 -Total 6 -Activity "Phase 6" -Status "Running ANOVA, Cohen's d analysis..."
    
    try {
        $pythonCmd = "py -3 -c `"from src.evaluation.phase6_statistical_benchmark import Phase6StatisticalBenchmark; p = Phase6StatisticalBenchmark(); p.run()`""
        $result = Invoke-Expression $pythonCmd 2>&1
        $State["phase6_complete"] = $true
        $State["phase6_result"] = "Success"
        Write-Log "Phase 6 completed successfully"
    }
    catch {
        $State["phase6_complete"] = $false
        $State["phase6_error"] = $_.Exception.Message
        Write-Log "Phase 6 failed: $($_.Exception.Message)" "ERROR"
    }
    
    return $State
}

function Start-CheckpointTimer {
    param([hashtable]$State)
    
    $timer = New-Object Timers.Timer
    $timer.Interval = $CHECKPOINT_INTERVAL_SEC * 1000
    $timer.AutoReset = $true
    
    $action = {
        param($sender, $e)
        $State["step"] = $State["step"] + 1
        Save-Checkpoint -State $State
    }
    
    Register-ObjectEvent -InputObject $timer -EventName Elapsed -Action $action | Out-Null
    $timer.Start()
    
    return $timer
}

function Install-AutoStart {
    Write-Log "Installing auto-start script..."
    
    $batContent = @"
@echo off
cd /d "$PROJECT_ROOT"
powershell -ExecutionPolicy Bypass -File "$PSCommandPath" -Resume
"@
    
    $batContent | Out-File -FilePath $STARTUP_SCRIPT -Encoding ASCII
    Write-Log "Auto-start installed: $STARTUP_SCRIPT"
    Write-Host "✓ Auto-start script installed. Pipeline will resume on next boot."
}

function Uninstall-AutoStart {
    if (Test-Path $STARTUP_SCRIPT) {
        Remove-Item $STARTUP_SCRIPT -Force
        Write-Log "Auto-start script removed."
        Write-Host "✓ Auto-start script uninstalled."
    }
    else {
        Write-Host "Auto-start script not found."
    }
}

function Run-Pipeline {
    param([switch]$Resume, [string]$Phase)
    
    Write-Log "=" * 60
    Write-Log "AEGIS Pipeline Orchestrator - PowerShell"
    Write-Log "=" * 60
    
    # Initialize or load state
    $state = @{
        "step" = 0
        "phase" = $Phase
        "start_time" = (Get-Date).ToString("o")
        "phase4_complete" = $false
        "world_events_complete" = $false
        "phase5_complete" = $false
        "phase6_complete" = $false
    }
    
    if ($Resume) {
        $loadedState = Load-Checkpoint
        if ($loadedState) {
            $state = $loadedState
            Write-Log "Resuming from step $($state.step)"
        }
    }
    
    # Start checkpoint timer
    $timer = Start-CheckpointTimer -State $state
    
    try {
        # Save PID
        $PID | Out-File -FilePath $PID_FILE -Encoding UTF8 -NoNewline
        
        # Execute phases
        if ($Phase -eq "all" -or $Phase -eq "4") {
            if (-not $state["phase4_complete"]) {
                $state = Run-Phase4 -State $state
                Save-Checkpoint -State $state
            }
            
            if (-not $state["world_events_complete"]) {
                $state = Run-WorldEvents -State $state
                Save-Checkpoint -State $state
            }
        }
        
        if ($Phase -eq "all" -or $Phase -eq "5") {
            if (-not $state["phase5_complete"]) {
                $state = Run-Phase5 -State $state
                Save-Checkpoint -State $state
            }
        }
        
        if ($Phase -eq "all" -or $Phase -eq "6") {
            if (-not $state["phase6_complete"]) {
                $state = Run-Phase6 -State $state
                Save-Checkpoint -State $state
            }
        }
        
        Show-Progress -Current 6 -Total 6 -Activity "Complete" -Status "Pipeline finished!"
        Write-Host ""
        Write-Log "=" * 60
        Write-Log "AEGIS Pipeline Complete!"
        Write-Log "=" * 60
    }
    finally {
        $timer.Stop()
        $timer.Dispose()
        if (Test-Path $PID_FILE) { Remove-Item $PID_FILE -Force }
    }
}

# Main entry point
if ($Install) {
    Install-AutoStart
}
elseif ($Uninstall) {
    Uninstall-AutoStart
}
else {
    Run-Pipeline -Resume:$Resume -Phase $Phase
}
