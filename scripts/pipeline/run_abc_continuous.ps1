#!/usr/bin/env pwsh
# ABC Pipeline Continuous Operation Script
# 5-minute rolling checkpoints, auto-resume on power-on, startup file cleanup

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$LogDir = "$ProjectRoot\logs"
$CheckpointDir = "D:\webdataset\checkpoints\abc_pipeline"
$LockFile = "$ScriptDir\running.lock"

$PythonExe = "py -3"
$PipelineScript = "$ProjectRoot\src\evaluation\abc_pipeline.py"
$LogFile = "$LogDir\abc_pipeline_continuous.log"

function Write-Log {
    param([string]$Message)
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogEntry = "[$Timestamp] $Message"
    Write-Host $LogEntry
    Add-Content -Path $LogFile -Value $LogEntry -Encoding UTF8
}

function Initialize-Directories {
    if (-not (Test-Path $LogDir)) {
        New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
    }
    if (-not (Test-Path $CheckpointDir)) {
        New-Item -ItemType Directory -Path $CheckpointDir -Force | Out-Null
    }
}

function Test-OllamaHealth {
    try {
        $Result = ollama list 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Log "[OK] Ollama is healthy"
            return $true
        }
    } catch {
        Write-Log "[WARNING] Ollama health check failed: $_"
    }
    return $false
}

function Remove-StartupFiles {
    $LockPatterns = @(
        "$ScriptDir\running.lock",
        "$CheckpointDir\*.lock",
        "$ProjectRoot\logs\pipeline_auto_resume.log"
    )
    foreach ($Pattern in $LockPatterns) {
        try {
            Get-Item $Pattern -ErrorAction SilentlyContinue | ForEach-Object {
                Remove-Item $_.FullName -Force
                Write-Log "[CLEANUP] Removed: $($_.FullName)"
            }
        } catch {}
    }
}

function Start-ABCPipeline {
    param(
        [switch]$Resume
    )

    Write-Log "========================================"
    Write-Log "ABC Pipeline Starting"
    Write-Log "Models: A=microsoft-phi3.5mini-instinct"
    Write-Log "        B=AXCEPT-Borea-phi3.5mini-jp"
    Write-Log "        C=zapabobouj-AEGIS-phi3.5-jp_v4.0"
    Write-Log "========================================"

    $OllamaReady = Test-OllamaHealth
    $RetryCount = 0
    $MaxRetries = 3

    while (-not $OllamaReady -and $RetryCount -lt $MaxRetries) {
        Write-Log "[WAIT] Waiting for Ollama... (attempt $($RetryCount + 1)/$MaxRetries)"
        Start-Sleep -Seconds 30
        $OllamaReady = Test-OllamaHealth
        $RetryCount++
    }

    if (-not $OllamaReady) {
        Write-Log "[ERROR] Ollama not available after $MaxRetries attempts"
        return $false
    }

    try {
        $ProcessArgs = @(
            $PipelineScript,
            "--interval", "300"
        )
        if ($Resume) {
            $ProcessArgs += "--resume"
        }

        $Process = Start-Process -FilePath $PythonExe -ArgumentList $ProcessArgs -PassThru -NoNewWindow

        Write-Log "[PROCESS] Pipeline PID: $($Process.Id)"

        $LockContent = @{
            PID = $Process.Id
            StartTime = (Get-Date).ToString("o")
            CheckpointDir = $CheckpointDir
        }
        $LockContent | ConvertTo-Json -Depth 3 | Set-Content -Path $LockFile -Encoding UTF8

        while (-not $Process.HasExited) {
            Start-Sleep -Seconds 60
            $Status = if ($Process.HasExited) "EXCLUDED" else "RUNNING"
            Write-Log "[STATUS] Pipeline $Status (PID: $($Process.Id))"
        }

        $ExitCode = $Process.ExitCode
        Write-Log "[COMPLETE] Pipeline exited with code: $ExitCode"

        if ($ExitCode -eq 0) {
            Write-Log "[SUCCESS] ABC Pipeline completed successfully"
            Remove-StartupFiles
            return $true
        } else {
            Write-Log "[ERROR] Pipeline failed with exit code: $ExitCode"
            return $false
        }
    }
    catch {
        Write-Log "[ERROR] Pipeline execution error: $_"
        return $false
    }
}

function Invoke-AutoResume {
    Write-Log "[AUTO-RESUME] Checking for checkpoints..."

    $LatestCheckpoint = Get-ChildItem -Path $CheckpointDir -Filter "checkpoint_*.json" |
                        Sort-Object -Property LastWriteTime -Descending |
                        Select-Object -First 1

    if ($LatestCheckpoint) {
        Write-Log "[AUTO-RESUME] Found checkpoint: $($LatestCheckpoint.Name)"
        Start-ABCPipeline -Resume
    }
    else {
        Write-Log "[AUTO-RESUME] No checkpoints found, starting fresh"
        Start-ABCPipeline
    }
}

Initialize-Directories
Write-Log "[START] ABC Continuous Pipeline Script"

if ($args -contains "--resume") {
    Invoke-AutoResume
}
elseif ($args -contains "--check") {
    Test-OllamaHealth
    Get-ChildItem $CheckpointDir -Filter "checkpoint_*.json" | Format-Table Name, LastWriteTime
}
else {
    Invoke-AutoResume
}

Write-Log "[END] ABC Continuous Pipeline Script completed"
