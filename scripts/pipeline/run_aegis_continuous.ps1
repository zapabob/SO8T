# AEGIS-v3.0 Full Automation & Continuous Operation
# 2026-02-06 Enhanced with Rolling Checkpoint & Progress Monitor

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent -Path (Split-Path -Parent -Path $PSScriptRoot)
$LogDir = Join-Path $ProjectRoot "logs"
$CheckpointDir = Join-Path (Join-Path $ProjectRoot "data") "pipeline_checkpoints"
$SessionId = Get-Date -Format 'yyyyMMdd_HHmmss'
$LogFile = Join-Path $LogDir "pipeline_continuous_$SessionId.log"
$ErrorLogFile = Join-Path $LogDir "pipeline_errors_$SessionId.log"

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }
if (-not (Test-Path $CheckpointDir)) { New-Item -ItemType Directory -Path $CheckpointDir | Out-Null }

# === Banner ===
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "       AEGIS-v3.0 FULL AUTOMATION & CONTINUOUS OPERATION    " -ForegroundColor Cyan
Write-Host "     5-Minute Rolling Checkpoint (3 Generations) Enabled    " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# === 0. Environment Configuration ===
$env:SO8T_USE_UNSLOTH = "1"
$env:SO8T_DRYRUN = "0"
$env:SO8T_HF_UPLOAD = "1"
$env:SO8T_HF_INCLUDE_LARGE = "0"
$env:SO8T_GRAPE_VARIANT = "multiplicative"
$env:SO8T_RESEARCH_TOPIC = "Advanced Mathematical Reasoning and SO8T Quadrality Optimization"
$env:SO8T_RECOVER = "0"
$env:SO8T_TRAINING_CONFIG = "src/infrastructure/config/borea_training.json"
$env:SO8T_CHECKPOINT_INTERVAL = "300"
$env:SO8T_CHECKPOINT_ROLLING = "3"

Write-Host "[CONFIG] Environment variables set:" -ForegroundColor Yellow
Write-Host "  SO8T_USE_UNSLOTH         = $($env:SO8T_USE_UNSLOTH)"
Write-Host "  SO8T_CHECKPOINT_INTERVAL = $($env:SO8T_CHECKPOINT_INTERVAL) seconds"
Write-Host "  SO8T_CHECKPOINT_ROLLING  = $($env:SO8T_CHECKPOINT_ROLLING) generations"
Write-Host ""

# === 1. Python Environment Check ===
Write-Host "[CHECK] Verifying Python installation..." -NoNewline
try {
    $pythonVersion = & py -3 --version 2>&1
    Write-Host " OK ($pythonVersion)" -ForegroundColor Green
}
catch {
    Write-Host " FAILED" -ForegroundColor Red
    Write-Host "[ERROR] Python 3 not found via 'py -3'. Please install Python."
    exit 1
}

# === 2. Checkpoint Detection ===
Write-Host "[RESUME] Checking for existing checkpoints..." -NoNewline
$checkpoints = Get-ChildItem -Path $CheckpointDir -Filter "pipeline_checkpoint_*.json" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
if ($checkpoints.Count -gt 0) {
    $latestCheckpoint = $checkpoints[0]
    Write-Host " FOUND ($($latestCheckpoint.Name))" -ForegroundColor Green
    Write-Host "[RESUME] Last modified: $($latestCheckpoint.LastWriteTime)"
}
else {
    Write-Host " None found (starting fresh)" -ForegroundColor Yellow
}
Write-Host ""

# === Error Display Function ===
function Show-RecentErrors {
    if (Test-Path $ErrorLogFile) {
        $errors = Get-Content $ErrorLogFile -Tail 5 -ErrorAction SilentlyContinue
        if ($errors -and $errors.Count -gt 0) {
            Write-Host ""
            Write-Host "========== Recent Errors ==========" -ForegroundColor Red
            $errors | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
            Write-Host "===================================" -ForegroundColor Red
        }
    }
}

# === 3. Auto-Resume Loop ===
$MaxRetries = 10
$RetryCount = 0

while ($RetryCount -lt $MaxRetries) {
    Write-Host ""
    Write-Host "------------------------------------------------------------" -ForegroundColor Yellow
    Write-Host "  [RUN] Executing Auto-Resume Pipeline (Attempt $($RetryCount + 1)/$MaxRetries)" -ForegroundColor Yellow
    Write-Host "------------------------------------------------------------" -ForegroundColor Yellow
    $startTime = Get-Date
    
    $host.ui.RawUI.WindowTitle = "AEGIS-v3.0 Pipeline [RUNNING] - Attempt $($RetryCount + 1)"

    try {
        # Suppress non-critical Python warnings (like FutureWarnings from torch/pynvml)
        # and prevent PowerShell from treating stderr as a fatal exception.
        $env:PYTHONWARNINGS = "ignore"
        $oldEAP = $ErrorActionPreference
        $ErrorActionPreference = "Continue"

        # Run with Tee-Object to show output AND log it
        & py -3 "$ProjectRoot\scripts\pipeline\auto_resume_aegis.py" 2>&1 | Tee-Object -FilePath $LogFile
        
        $ErrorActionPreference = $oldEAP
        $exitCode = $LASTEXITCODE
        if ($exitCode -eq 0) {
            Write-Host ""
            Write-Host "============================================================" -ForegroundColor Green
            Write-Host "  [SUCCESS] Pipeline completed successfully at $(Get-Date)" -ForegroundColor Green
            Write-Host "============================================================" -ForegroundColor Green
            $host.ui.RawUI.WindowTitle = "AEGIS-v3.0 Pipeline [FINISHED]"
            
            # Show checkpoint status
            $cpFiles = Get-ChildItem -Path $CheckpointDir -Filter "pipeline_checkpoint_*.json" -ErrorAction SilentlyContinue
            if ($cpFiles.Count -gt 0) {
                Write-Host "[CHECKPOINT] $($cpFiles.Count) checkpoint files saved" -ForegroundColor DarkGreen
            }
            break
        }
        else {
            Write-Host "[WARNING] Pipeline exited with code $exitCode." -ForegroundColor Yellow
            Show-RecentErrors
        }
    }
    catch {
        Write-Host "[CRITICAL] Exception occurred: $_" -ForegroundColor Red
        $_ | Out-File -FilePath $ErrorLogFile -Append
        Show-RecentErrors
    }

    $RetryCount++
    $endTime = Get-Date
    $duration = $endTime - $startTime
    
    if ($duration.TotalSeconds -lt 60) {
        Write-Host "[ALERT] Pipeline crashed too quickly ($([math]::Round($duration.TotalSeconds))s). Waiting 30s before retry..." -ForegroundColor Red
        Start-Sleep -Seconds 30
    }

    Write-Host "[RETRY] Attempting restart ($RetryCount/$MaxRetries)..." -ForegroundColor Magenta
    $host.ui.RawUI.WindowTitle = "AEGIS-v3.0 Pipeline [RETRYING $RetryCount]"
}

if ($RetryCount -ge $MaxRetries) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Red
    Write-Host "  [FATAL] Max retries reached. System execution halted." -ForegroundColor Red
    Write-Host "============================================================" -ForegroundColor Red
    Show-RecentErrors
}

Write-Host ""
Write-Host "[END] Session closed at $(Get-Date)."
Write-Host "[LOG] Full log: $LogFile"
Write-Host "[LOG] Error log: $ErrorLogFile"
