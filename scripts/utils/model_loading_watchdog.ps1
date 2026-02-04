#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Moonshot Pipeline v3.0 - Model Loading Watchdog

.DESCRIPTION
    Detects stale "model_loading" checkpoints and auto-resumes the pipeline.
    Intended to run on a schedule (e.g., every 5 minutes).

.PARAMETER StaleMinutes
    Minutes threshold to consider model_loading stale (default: 15)

.PARAMETER ProjectRoot
    Optional override for project root

.PARAMETER CheckpointPath
    Optional override for checkpoint JSON path

.PARAMETER LogPath
    Optional override for watchdog log path
#>

param(
    [int]$StaleMinutes = 15,
    [string]$ProjectRoot = "",
    [string]$CheckpointPath = "",
    [string]$LogPath = ""
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $ProjectRoot) {
    $ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..\\..")
}

if (-not $CheckpointPath) {
    $CheckpointPath = Join-Path $ProjectRoot "checkpoints\\latest_checkpoint.json"
}

if (-not $LogPath) {
    $LogPath = Join-Path $ProjectRoot "logs\\model_loading_watchdog.log"
}

$LogDir = Split-Path -Parent $LogPath
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$PipelineScript = Join-Path $ProjectRoot "run_moonshot_full_pipeline.py"

function Write-Log {
    param([string]$Message)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $LogPath -Value "$ts $Message"
}

try {
    # If pipeline already running, do nothing
    $running = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='py.exe'" |
        Where-Object { $_.CommandLine -match "run_moonshot_full_pipeline\.py" }
    if ($running) {
        Write-Log "[SKIP] Pipeline already running"
        exit 0
    }

    if (-not (Test-Path $CheckpointPath)) {
        Write-Log "[SKIP] Checkpoint not found: $CheckpointPath"
        exit 0
    }

    $json = Get-Content -Path $CheckpointPath -Raw | ConvertFrom-Json
    if ($null -eq $json) {
        Write-Log "[WARN] Failed to parse checkpoint JSON"
        exit 0
    }

    if ($json.current_phase -ne "model_loading") {
        Write-Log "[SKIP] Current phase: $($json.current_phase)"
        exit 0
    }

    $ts = [datetime]::Parse($json.timestamp)
    $age = (Get-Date) - $ts
    if ($age.TotalMinutes -lt $StaleMinutes) {
        Write-Log "[SKIP] model_loading age ${age.TotalMinutes}min < $StaleMinutes"
        exit 0
    }

    # Mark checkpoint as stale
    $stale = Join-Path (Split-Path $CheckpointPath) ("stale_model_loading_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".json")
    Move-Item -Path $CheckpointPath -Destination $stale -Force
    Write-Log "[ACTION] Stale checkpoint moved: $stale"

    # Resolve python
    $pythonExe = "$env:USERPROFILE\\.pyenv\\pyenv-win\\versions\\3.12.0\\python.exe"
    $usePyLauncher = $false
    if (-not (Test-Path $pythonExe)) {
        $pyCmd = Get-Command py -ErrorAction SilentlyContinue
        if ($pyCmd) {
            $pythonExe = "py"
            $usePyLauncher = $true
        } else {
            $pyCmd = Get-Command python -ErrorAction SilentlyContinue
            if ($pyCmd) {
                $pythonExe = $pyCmd.Source
            }
        }
    }

    if (-not $pythonExe) {
        Write-Log "[ERROR] Python not found"
        exit 1
    }

    if ($usePyLauncher) {
        $args = @("-3", $PipelineScript, "--resume")
    } else {
        $args = @($PipelineScript, "--resume")
    }

    Write-Log "[ACTION] Restarting pipeline: $pythonExe $($args -join ' ')"
    Start-Process -FilePath $pythonExe -ArgumentList $args -WorkingDirectory $ProjectRoot -WindowStyle Hidden
    Write-Log "[OK] Restart triggered"
} catch {
    Write-Log "[ERROR] $($_.Exception.Message)"
    exit 1
}
