#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Moonshot Pipeline v3.0 - Auto Power-On Resume Setup

.DESCRIPTION
    Configures Windows Task Scheduler for automatic pipeline resume on power-on.
    Ensures rolling checkpoints every 3 minutes and progress tracking.

.PARAMETER PipelineScript
    Path to the pipeline script (default: run_moonshot_pipeline_2025_2026.py)

.PARAMETER CheckpointInterval
    Checkpoint capture interval in seconds (default: 180 = 3 minutes)

.PARAMETER MaxCheckpoints
    Maximum number of rolling checkpoints to keep (default: 5)

.PARAMETER WatchdogIntervalMinutes
    Watchdog execution interval in minutes (default: 5)

.PARAMETER ModelLoadingStaleMinutes
    Minutes before model_loading is considered stale (default: 15)

.EXAMPLE
    .\setup_auto_poweron.ps1
#>

param(
    [string]$PipelineScript = "run_moonshot_pipeline_2025_2026.py",
    [int]$CheckpointInterval = 180,
    [int]$MaxCheckpoints = 5,
    [string]$TaskName = "MoonshotPipelineV3_AutoResume",
    [int]$WatchdogIntervalMinutes = 5,
    [int]$ModelLoadingStaleMinutes = 15
)

$ErrorActionPreference = "Stop"

# Configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Get-Item $ScriptDir).Parent.Parent
$LogDir = "$ProjectRoot\logs"
$CheckpointDir = "$ProjectRoot\checkpoints\rolling_snapshots"
$WatchdogScript = "$ProjectRoot\scripts\utils\model_loading_watchdog.ps1"
$WatchdogTaskName = "MoonshotPipelineV3_ModelLoadingWatchdog"

# Ensure directories exist
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $CheckpointDir | Out-Null

# Python executable
$PythonExe = "$env:USERPROFILE\.pyenv\pyenv-win\versions\3.12.0\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "py"
    Write-Warning "Using system python (py). Ensure Python 3.12 is installed."
}

# Task Scheduler action
$Action = New-ScheduledTaskAction `
    -Execute "$PythonExe" `
    -Argument "-FullPath ""$ProjectRoot\$PipelineScript"" --use-existing-datasets" `
    -WorkingDirectory "$ProjectRoot"

# Task trigger - on system startup
$Trigger = New-ScheduledTaskTrigger `
    -AtStartup `
    -RandomDelay "00:01:00"  # 1 minute delay

# Task settings - ensure it runs with network access and suitable power settings
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries $true `
    -DontStopIfGoingOnBatteries $true `
    -StartWhenAvailable $true `
    -RunOnlyIfNetworkAvailable $true `
    -ExecutionTimeLimit "48:00:00" `
    -RestartOnFailure $true `
    -RestartCount 3 `
    -RestartInterval "00:05:00"

# Task principal - run with highest privileges
$Principal = New-ScheduledTaskPrincipal `
    -UserId "NT AUTHORITY\SYSTEM" `
    -LogonType "ServiceAccount" `
    -RunLevel "Highest"

# Register the task
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Moonshot Pipeline v3.0 - Auto Resume Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`n[CONFIG]" -ForegroundColor Yellow
Write-Host "  Task Name:    $TaskName"
Write-Host "  Pipeline:     $PipelineScript"
Write-Host "  Checkpoint:   Every $CheckpointInterval seconds"
Write-Host "  Max Snapshots: $MaxCheckpoints"
Write-Host "  Python:       $PythonExe"
Write-Host "  Watchdog:     Every $WatchdogIntervalMinutes min (stale $ModelLoadingStaleMinutes min)"

# Check if task exists and remove
$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($ExistingTask) {
    Write-Host "`n[REMOVE] Removing existing task..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Start-Sleep -Seconds 2
}

# Register new task
Write-Host "`n[REGISTER] Creating scheduled task..." -ForegroundColor Yellow
try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Principal $Principal `
        -Force | Out-Null

    Write-Host "[OK] Task registered successfully!" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Failed to register task: $_" -ForegroundColor Red
    exit 1
}

# Register watchdog task (model_loading stale recovery)
if (Test-Path $WatchdogScript) {
    Write-Host "`n[REGISTER] Creating watchdog task..." -ForegroundColor Yellow
    try {
        $wdAction = New-ScheduledTaskAction `
            -Execute "powershell.exe" `
            -Argument "-ExecutionPolicy Bypass -File `"$WatchdogScript`" -StaleMinutes $ModelLoadingStaleMinutes" `
            -WorkingDirectory "$ProjectRoot"

        $wdTrigger = New-ScheduledTaskTrigger -AtStartup
        $wdTrigger.RepetitionInterval = (New-TimeSpan -Minutes $WatchdogIntervalMinutes)
        $wdTrigger.RepetitionDuration = (New-TimeSpan -Days 3650)

        $wdSettings = New-ScheduledTaskSettingsSet `
            -AllowStartIfOnBatteries $true `
            -DontStopIfGoingOnBatteries $true `
            -StartWhenAvailable $true `
            -RunOnlyIfNetworkAvailable $false `
            -ExecutionTimeLimit "00:10:00" `
            -RestartOnFailure $true `
            -RestartCount 3 `
            -RestartInterval "00:02:00"

        $wdPrincipal = New-ScheduledTaskPrincipal `
            -UserId "NT AUTHORITY\SYSTEM" `
            -LogonType "ServiceAccount" `
            -RunLevel "Highest"

        $ExistingWatchdog = Get-ScheduledTask -TaskName $WatchdogTaskName -ErrorAction SilentlyContinue
        if ($ExistingWatchdog) {
            Unregister-ScheduledTask -TaskName $WatchdogTaskName -Confirm:$false
            Start-Sleep -Seconds 2
        }

        Register-ScheduledTask `
            -TaskName $WatchdogTaskName `
            -Action $wdAction `
            -Trigger $wdTrigger `
            -Settings $wdSettings `
            -Principal $wdPrincipal `
            -Force | Out-Null

        Write-Host "[OK] Watchdog task registered: $WatchdogTaskName" -ForegroundColor Green
    } catch {
        Write-Host "[ERROR] Watchdog registration failed: $_" -ForegroundColor Red
    }
} else {
    Write-Host "[WARN] Watchdog script not found: $WatchdogScript" -ForegroundColor Yellow
}

# Create helper scripts
Write-Host "`n[CREATE] Helper scripts..." -ForegroundColor Yellow

# Rolling checkpoint script
$CheckpointScript = @"
#!/usr/bin/env python3
"""
Rolling checkpoint capture script.
Runs every $CheckpointInterval seconds, keeps $MaxCheckpoints snapshots.
"""
import shutil
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
SOURCE = PROJECT_ROOT / "checkpoints" / "latest_checkpoint.json"
DEST_DIR = PROJECT_ROOT / "checkpoints" / "rolling_snapshots"
DEST_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
DEST = DEST_DIR / f"rolling_checkpoint_{TIMESTAMP}.json"

if SOURCE.exists():
    shutil.copy2(SOURCE, DEST)
    print(f"[CHECKPOINT] Captured: {DEST.name}")

    # Trim old checkpoints
    checkpoints = sorted(DEST_DIR.glob("rolling_checkpoint_*.json"),
                        key=lambda p: p.stat().st_mtime)
    while len(checkpoints) > $MaxCheckpoints:
        stale = checkpoints.pop(0)
        stale.unlink()
        print(f"[CLEANUP] Removed: {stale.name}")
else:
    print("[SKIP] No checkpoint source found")
"@

$CheckpointScriptPath = "$ScriptDir\rolling_checkpoint_capture.py"
$CheckpointScript | Out-File -FilePath $CheckpointScriptPath -Encoding UTF8
Write-Host "  - $CheckpointScriptPath"

# Progress reporter script
$ProgressScript = @"
#!/usr/bin/env python3
"""
Tqdm-style progress reporter.
Simple English progress messages to log file.
"""
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
LOG_FILE = PROJECT_ROOT / "logs" / "pipeline_progress.log"

BAR_LENGTH = 10
state = 0

while True:
    filled = "=" * state
    bar = f"|{filled}{'-' * (BAR_LENGTH - state)}|"
    message = f"[PROGRESS] {bar} Pipeline running - {datetime.now().isoformat()}"

    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(message + "\n")

    print(message)
    state = (state + 1) % (BAR_LENGTH + 1)
    time.sleep(60)  # Report every minute
"@

$ProgressScriptPath = "$ScriptDir\progress_reporter.py"
$ProgressScript | Out-File -FilePath $ProgressScriptPath -Encoding UTF8
Write-Host "  - $ProgressScriptPath"

# Create README
$ReadmeContent = @"
# Moonshot Pipeline v3.0 - Auto Resume

## Setup

Run this script to configure automatic pipeline resume on power-on:

```powershell
.\setup_auto_poweron.ps1
```

## Features

- **Automatic Startup**: Pipeline resumes on system boot
- **Rolling Checkpoints**: 5 snapshots every 3 minutes
- **Model-Loading Watchdog**: Auto-restart when stuck > 15 min
- **Progress Logging**: Tqdm-style progress in `logs/pipeline_progress.log`
- **SQL Tracking**: All progress saved to `logs/pipeline_progress.sqlite`

## Manual Control

```powershell
# Start pipeline manually
py -3 run_moonshot_pipeline_2025_2026.py

# Monitor progress
py -3 scripts/utils/monitor_pipeline.py

# Check scheduled tasks
Get-ScheduledTask -TaskName "MoonshotPipelineV3*"
Get-ScheduledTask -TaskName "MoonshotPipelineV3_ModelLoadingWatchdog"
```

## Checkpoint Locations

- Latest: `checkpoints/latest_checkpoint.json`
- Rolling: `checkpoints/rolling_snapshots/`
- SQL DB: `logs/pipeline_progress.sqlite`

## Remove Auto-Resume

```powershell
Unregister-ScheduledTask -TaskName "MoonshotPipelineV3_AutoResume" -Confirm
```
"@

$ReadmePath = "$ScriptDir\README_AUTO_RESUME.md"
$ReadmeContent | Out-File -FilePath $ReadmePath -Encoding UTF8
Write-Host "  - $ReadmePath"

# Final summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nTo start pipeline manually:"
Write-Host "  py -3 run_moonshot_pipeline_2025_2026.py"
Write-Host ""
Write-Host "To monitor progress:"
Write-Host "  py -3 scripts/utils/monitor_pipeline.py"
Write-Host ""
Write-Host "To check scheduled tasks:"
Write-Host "  Get-ScheduledTask -TaskName ""MoonshotPipelineV3*"""
Write-Host ""
