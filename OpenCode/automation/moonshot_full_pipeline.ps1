#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Moonshot Pipeline v3.0 - Full Automation Launcher (PowerShell)

.DESCRIPTION
    English PowerShell entrypoint for the improved Moonshot pipeline.
    Provides dry-run support, progress-style output, logging, and startup
    auto-resume integration using existing Task Scheduler settings.

.PARAMETER DryRun
    Run setup + data validation only (no training/benchmark).

.PARAMETER Resume
    Resume from latest checkpoint (used for auto-resume on power-on).

.PARAMETER SetupStartup
    Register startup task (reuse current startup handler).

.PARAMETER RemoveStartup
    Remove startup task.

.PARAMETER Status
    Show status (task scheduler + logs + checkpoints).
#>

param(
    [switch]$DryRun,
    [switch]$Resume,
    [switch]$SetupStartup,
    [switch]$RemoveStartup,
    [switch]$Status
)

$ErrorActionPreference = "Stop"

function Write-Stage {
    param([string]$Message, [string]$Color = "Cyan")
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$ts] $Message" -ForegroundColor $Color
}

function Write-ProgressStage {
    param([string]$Activity, [int]$Step, [int]$Total)
    $percent = [int](($Step / [double]$Total) * 100)
    Write-Progress -Activity $Activity -Status "$Step / $Total" -PercentComplete $percent
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Get-Item $ScriptDir).Parent.FullName
$LogsDir = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null

$logPath = Join-Path $LogsDir ("moonshot_full_ps_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
Start-Transcript -Path $logPath | Out-Null

Write-Stage "Moonshot Pipeline v3.0 - PowerShell Launcher"
Write-Stage "Project Root: $ProjectRoot" "Yellow"

Write-ProgressStage -Activity "Initialize" -Step 1 -Total 3

# Resolve Python (prefer py)
$PythonExe = (Get-Command py -ErrorAction SilentlyContinue).Source
if (-not $PythonExe) {
    $PythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
}
if (-not $PythonExe) {
    Write-Stage "Python not found" "Red"
    Stop-Transcript | Out-Null
    exit 1
}
Write-Stage "Python: $PythonExe" "Yellow"

Write-ProgressStage -Activity "Initialize" -Step 2 -Total 3

# Startup management (reuse existing startup handler)
if ($SetupStartup) {
    Write-Stage "Configuring startup auto-resume using existing handler..." "Yellow"
    & $PythonExe (Join-Path $ProjectRoot "scripts\\utils\\boot_pipeline_launcher.py") --setup-startup
    Stop-Transcript | Out-Null
    exit $LASTEXITCODE
}
if ($RemoveStartup) {
    Write-Stage "Removing startup auto-resume task..." "Yellow"
    & $PythonExe (Join-Path $ProjectRoot "scripts\\utils\\boot_pipeline_launcher.py") --remove-startup
    Stop-Transcript | Out-Null
    exit $LASTEXITCODE
}
if ($Status) {
    Write-Stage "Status" "Yellow"
    & $PythonExe (Join-Path $ProjectRoot "run_moonshot_full_pipeline.py") --status
    Stop-Transcript | Out-Null
    exit $LASTEXITCODE
}

Write-ProgressStage -Activity "Initialize" -Step 3 -Total 3

Write-Stage "Starting Moonshot Full Pipeline..." "Yellow"
Write-Stage ("Mode: " + ($(if ($DryRun) { "DRY-RUN" } elseif ($Resume) { "RESUME" } else { "FULL" }))) "Yellow"

$scriptPath = (Join-Path $ProjectRoot "run_moonshot_full_pipeline.py")
$pyArgs = @()
if ($Resume) { $pyArgs += "--resume" }
if ($DryRun) { $pyArgs += "--dry-run" }

& $PythonExe $scriptPath @pyArgs

Write-Stage "Pipeline finished. Exit code: $LASTEXITCODE" $(if ($LASTEXITCODE -eq 0) { "Green" } else { "Red" })
Stop-Transcript | Out-Null
exit $LASTEXITCODE
