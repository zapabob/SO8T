#!/usr/bin/env pwsh
# ABC Pipeline Continuous Operation Script
# 5-minute rolling checkpoints, auto-resume on power-on, startup file cleanup
# Supports skipping data collection/processing if data already exists

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$LogDir = "$ProjectRoot\logs"
$CheckpointDir = "D:\webdataset\checkpoints\abc_pipeline"
$DataDir = "D:\webdataset\data"
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
    if (-not (Test-Path $DataDir)) {
        New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
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

function Test-DataReadiness {
    param([string]$DataDir)

    Write-Log "[DATA] Checking data readiness..."

    $DatasetPatterns = @(
        "datasets\arxiv_papers\*",
        "datasets\biorxiv_papers\*",
        "datasets\world_events\*",
        "datasets\cleansed\*",
        "datasets\vssi_tagged\*"
    )

    $ReadyCount = 0
    $TotalPatterns = $DatasetPatterns.Length

    foreach ($Pattern in $DatasetPatterns) {
        $FullPath = Join-Path $DataDir $Pattern
        if (Test-Path $FullPath) {
            $Files = Get-ChildItem $FullPath -ErrorAction SilentlyContinue | Measure-Object
            if ($Files.Count -gt 0) {
                $ReadyCount++
                Write-Log "[OK] Found data matching: $Pattern"
            }
        }
    }

    $IsReady = $ReadyCount -ge ($TotalPatterns * 0.5)  # 50% ready
    Write-Log "[DATA] Data readiness: $ReadyCount/$TotalPatterns patterns found, IsReady=$IsReady"
    return $IsReady
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
        [switch]$Resume,
        [switch]$SkipDataCollection,
        [switch]$SkipDataProcessing,
        [switch]$SkipDataCleansing
    )

    Write-Log "========================================"
    Write-Log "ABC Pipeline Starting"
    Write-Log "Models: A=microsoft-phi3.5mini-instinct"
    Write-Log "        B=AXCEPT-Borea-phi3.5mini-jp"
    Write-Log "        C=zapabobouj-AEGIS-phi3.5-jp_v4.0"
    Write-Log "========================================"
    Write-Log "Skip flags: Collection=$SkipDataCollection, Processing=$SkipDataProcessing, Cleansing=$SkipDataCleansing"

    $DataReady = Test-DataReadiness -DataDir $DataDir

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
        if ($SkipDataCollection) {
            $ProcessArgs += "--skip-data-collection"
        }
        if ($SkipDataProcessing) {
            $ProcessArgs += "--skip-data-processing"
        }
        if ($SkipDataCleansing) {
            $ProcessArgs += "--skip-data-cleansing"
        }

        $Process = Start-Process -FilePath $PythonExe -ArgumentList $ProcessArgs -PassThru -NoNewWindow

        Write-Log "[PROCESS] Pipeline PID: $($Process.Id)"

        $LockContent = @{
            PID = $Process.Id
            StartTime = (Get-Date).ToString("o")
            CheckpointDir = $CheckpointDir
            DataReady = $DataReady
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

        $LockContent = @{}
        if (Test-Path $LockFile) {
            try {
                $LockContent = Get-Content $LockFile | ConvertFrom-Json
            } catch {}
        }

        $SkipCollection = if ($LockContent.DataReady -eq $true) { $true } else { $false }
        $SkipProcessing = if ($LockContent.DataReady -eq $true) { $true } else { $false }

        Start-ABCPipeline -Resume -SkipDataCollection:$SkipCollection -SkipDataProcessing:$SkipProcessing
    }
    else {
        Write-Log "[AUTO-RESUME] No checkpoints found"
        Start-ABCPipeline
    }
}

Initialize-Directories
Write-Log "[START] ABC Continuous Pipeline Script"

$Arguments = $args

if ($Arguments -contains "--resume") {
    Invoke-AutoResume
}
elseif ($Arguments -contains "--check") {
    Test-OllamaHealth
    Test-DataReadiness -DataDir $DataDir
    Get-ChildItem $CheckpointDir -Filter "checkpoint_*.json" | Format-Table Name, LastWriteTime
}
else {
    $SkipCollection = $Arguments -contains "--skip-data-collection"
    $SkipProcessing = $Arguments -contains "--skip-data-processing"
    $SkipCleansing = $Arguments -contains "--skip-data-cleansing"

    Start-ABCPipeline -SkipDataCollection:$SkipCollection -SkipDataProcessing:$SkipProcessing -SkipDataCleansing:$SkipCleansing
}

Write-Log "[END] ABC Continuous Pipeline Script completed"
