#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Moonshot Pipeline v3.0 - Startup Launcher

.DESCRIPTION
    Main entry point for Moonshot Pipeline v3.0 with power failure protection.
    Supports auto-start configuration and status checking.

.PARAMETER SetupStartup
    Register Windows Task Scheduler for auto-start on power-on

.PARAMETER RemoveStartup
    Remove Windows Task Scheduler entry

.PARAMETER Status
    Check current startup status

.EXAMPLE
    .\startup.ps1                         # Run pipeline normally
    .\startup.ps1 -SetupStartup           # Configure auto-start
    .\startup.ps1 -Status                 # Check status
#>

param(
    [switch]$SetupStartup,
    [switch]$RemoveStartup,
    [switch]$Status
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Get-Item $ScriptDir).Parent.Parent

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "Moonshot Pipeline v3.0 - Startup Launcher" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

# Find Python
$PythonExe = "$env:USERPROFILE\.pyenv\pyenv-win\versions\3.12.0\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = (Get-Command py -ErrorAction SilentlyContinue).Source
    if (-not $PythonExe) {
        $PythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
    }
}

if (-not $PythonExe) {
    Write-Host "[ERROR] Python not found" -ForegroundColor Red
    exit 1
}

Write-Host "[INFO] Python: $PythonExe" -ForegroundColor Yellow
Write-Host ""

# Handle switches
if ($SetupStartup) {
    Write-Host "[SETUP] Configuring Windows Task Scheduler..." -ForegroundColor Yellow
    & $PythonExe -FullPath "$ProjectRoot\scripts\utils\boot_pipeline_launcher.py" --setup-startup
    exit $LASTEXITCODE
}

if ($RemoveStartup) {
    Write-Host "[SETUP] Removing Task Scheduler entry..." -ForegroundColor Yellow
    & $PythonExe -FullPath "$ProjectRoot\scripts\utils\boot_pipeline_launcher.py" --remove-startup
    exit $LASTEXITCODE
}

if ($Status) {
    Write-Host "[STATUS] Checking Task Scheduler..." -ForegroundColor Yellow
    Write-Host ""

    try {
        $Task = Get-ScheduledTask -TaskName "MoonshotPipelineV3*" -ErrorAction Stop
        $Task | Format-Table TaskName, State, Date -AutoSize
    } catch {
        Write-Host "No Moonshot task found" -ForegroundColor Gray
    }

    Write-Host ""
    Write-Host "[STATUS] Recent logs:" -ForegroundColor Yellow
    Get-ChildItem -Path "$ProjectRoot\logs" -Filter "*.log" -ErrorAction SilentlyContinue |
        Sort-Object -Property LastWriteTime -Descending |
        Select-Object -First 5 |
        Format-Table Name, LastWriteTime -AutoSize

    Write-Host ""
    Write-Host "[STATUS] Rolling checkpoints:" -ForegroundColor Yellow
    Get-ChildItem -Path "$ProjectRoot\checkpoints\rolling_snapshots" -Filter "*.json" -ErrorAction SilentlyContinue |
        Sort-Object -Property LastWriteTime -Descending |
        Select-Object -First 3 |
        Format-Table Name, LastWriteTime -AutoSize

    exit 0
}

# Run pipeline normally
Write-Host "[RUN] Starting Moonshot Pipeline v3.0..." -ForegroundColor Yellow
Write-Host "[RUN] Checkpoint interval: 180 seconds (3 min)" -ForegroundColor Yellow
Write-Host "[RUN] Max rolling checkpoints: 5" -ForegroundColor Yellow
Write-Host "[RUN] Power failure recovery: enabled" -ForegroundColor Yellow
Write-Host ""

& $PythonExe -FullPath "$ProjectRoot\scripts\utils\boot_pipeline_launcher.py" --use-existing-datasets

Write-Host ""
Write-Host "[DONE] Pipeline finished. Exit code: $LASTEXITCODE" -ForegroundColor $(if ($LASTEXITCODE -eq 0) { "Green" } else { "Red" })
exit $LASTEXITCODE
