# ============================================================================
# PowerShell Auto-Resume Script for Evolved Shinka Pipeline
# ============================================================================
# 電源投入時にAEGIS-V3.0パイプラインを自動再開
#
# Features:
# - Ollama健全性チェック
# - 最新チェックポイントから自動再開
# - ログ出力とプログレス監視
# - Task Scheduler統合対応
# - エラー処理と再試行
#
# Usage:
#   .\power_on_auto_resume.ps1                    # 通常実行
#   .\power_on_auto_resume.ps1 -Force            # 強制再実行
#   .\power_on_auto_resume.ps1 -DryRun            # ドライラン
#   .\power_on_auto_resume.ps1 -InstallScheduler # Task Scheduler登録
#
# ============================================================================

param(
    [switch]$Force = $false,
    [switch]$DryRun = $false,
    [switch]$InstallScheduler = $false,
    [string]$ConfigPath = "",
    [int]$MaxRetries = 3,
    [int]$OllamaTimeout = 60
)

# ============================================================================
# Configuration
# ============================================================================

$Script:Version = "3.0.0"
$Script:LogFile = "$PSScriptRoot\..\..\logs\pipeline_auto_resume.log"
$Script:CheckpointDir = "$PSScriptRoot\..\..\checkpoints\evolved_pipeline"
$Script:StateFile = "data\evolved_pipeline_state.json"
$Script:PipelineScript = "src\infrastructure\pipeline\evolved_shinka_pipeline.py"
$Script:PythonExe = "py -3"

$Script:Config = @{
    OllamaModel = "borea-phi-3.5-instinct-jp"
    OllamaUrl = "http://localhost:11434"
    WebDatasetPath = "D:\webdataset"
    LogRetentionDays = 7
    RetryDelaySeconds = 30
}

# ============================================================================
# Logging Functions
# ============================================================================

function Write-Log {
    param(
        [string]$Message,
        [string]$Level = "INFO",
        [string]$Component = "AutoResume"
    )

    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogEntry = "[$Timestamp] [$Level] [$Component] $Message"

    Add-Content -Path $Script:LogFile -Value $LogEntry -Encoding UTF8

    if ($Level -eq "ERROR") {
        Write-Host $LogEntry -ForegroundColor Red
    } elseif ($Level -eq "WARNING") {
        Write-Host $LogEntry -ForegroundColor Yellow
    } elseif ($Level -eq "SUCCESS") {
        Write-Host $LogEntry -ForegroundColor Green
    } else {
        Write-Host $LogEntry
    }
}

function Initialize-Logging {
    $BaseDir = "$PSScriptRoot\..\.."
    $LogDir = Join-Path $BaseDir "logs"

    if (-not (Test-Path $LogDir)) {
        try {
            New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
        } catch {
            $LogDir = $PSScriptRoot
        }
    }

    $Timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
    $SessionLog = Join-Path $LogDir "auto_resume_$Timestamp.log"
    $Script:LogFile = $SessionLog

    Write-Log "========================================" "INFO" "Startup"
    Write-Log "Evolved Shinka Pipeline Auto-Resume v$Script:Version" "INFO" "Startup"
    Write-Log "Power-on auto-resume script initialized" "INFO" "Startup"
}

# ============================================================================
# Utility Functions
# ============================================================================

function Get-LatestCheckpoint {
    $BaseDir = "$PSScriptRoot\..\.."
    $CheckpointDir = Join-Path $BaseDir "checkpoints\evolved_pipeline"

    if (-not (Test-Path $CheckpointDir)) {
        return $null
    }

    $checkpoints = Get-ChildItem -Path $CheckpointDir -Filter "checkpoint_*.json" -File

    if ($checkpoints.Count -eq 0) {
        return $null
    }

    $latest = $checkpoints | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    return $latest.FullName
}

function Test-OllamaHealth {
    param(
        [string]$Url,
        [int]$Timeout = 60
    )

    Write-Log "Checking Ollama health at $Url" "INFO" "Ollama"

    try {
        $response = Invoke-WebRequest -Uri "$Url/api/version" -TimeoutSec $Timeout -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            Write-Log "Ollama is healthy" "SUCCESS" "Ollama"
            return $true
        }
    } catch {
        Write-Log "Ollama health check failed: $($_.Exception.Message)" "WARNING" "Ollama"
        return $false
    }

    return $false
}

function Get-PipelineState {
    $BaseDir = "$PSScriptRoot\..\.."
    $StateFilePath = Join-Path $BaseDir "data\evolved_pipeline_state.json"

    if (-not (Test-Path $StateFilePath)) {
        return @{
            Exists = $false
            CurrentPhase = "none"
            IsCompleted = $false
        }
    }

    try {
        $state = Get-Content -Path $StateFilePath -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
        return @{
            Exists = $true
            CurrentPhase = $state.current_phase
            IsCompleted = $state.is_completed
            TotalSamples = $state.total_samples_processed
            CheckpointCount = $state.checkpoint_count
            StartTime = $state.start_time
            LastCheckpoint = $state.last_checkpoint_time
        }
    } catch {
        Write-Log "Failed to read pipeline state: $($_.Exception.Message)" "WARNING" "State"
        return @{
            Exists = $false
            CurrentPhase = "unknown"
            IsCompleted = $false
        }
    }
}

function Start-Pipeline {
    param(
        [string]$PythonExe,
        [string]$ScriptPath,
        [switch]$Resume,
        [string]$ConfigPath = "",
        [int]$MaxRetries = 3
    )

    $BaseDir = "$PSScriptRoot\..\.."
    $FullScriptPath = Join-Path $BaseDir $ScriptPath

    $retryCount = 0
    $lastError = $null

    while ($retryCount -lt $MaxRetries) {
        try {
            $arguments = @(
                "`"$FullScriptPath`""
            )

            if ($Resume) {
                $arguments += "--resume"
            }

            if ($ConfigPath -and (Test-Path $ConfigPath)) {
                $arguments += "--config"
                $arguments += "`"$ConfigPath`""
            }

            Write-Log "Starting pipeline..." "INFO" "Pipeline"

            if ($DryRun) {
                Write-Log "[DRY RUN] Would execute: $PythonExe $arguments" "INFO" "Pipeline"
                return @{ Success = $true; DryRun = $true }
            }

            $process = Start-Process -FilePath $PythonExe -ArgumentList $arguments -PassThru -NoNewWindow -Wait

            if ($process.ExitCode -eq 0) {
                Write-Log "Pipeline completed successfully" "SUCCESS" "Pipeline"
                return @{ Success = $true; ExitCode = 0 }
            } else {
                Write-Log "Pipeline failed with exit code: $($process.ExitCode)" "ERROR" "Pipeline"
                $lastError = "Exit code $($process.ExitCode)"
            }
        } catch {
            Write-Log "Pipeline execution error: $($_.Exception.Message)" "ERROR" "Pipeline"
            $lastError = $_.Exception.Message
        }

        $retryCount++
        if ($retryCount -lt $MaxRetries) {
            Write-Log "Retrying in $Script:Config.RetryDelaySeconds seconds... (Attempt $retryCount/$MaxRetries)" "WARNING" "Pipeline"
            Start-Sleep -Seconds $Script:Config.RetryDelaySeconds
        }
    }

    Write-Log "Pipeline failed after $MaxRetries attempts" "ERROR" "Pipeline"
    return @{ Success = $false; Error = $lastError }
}

function Install-ScheduledTask {
    $taskName = "AEGIS-Evolved-Pipeline-AutoResume"
    $taskDescription = "AEGIS v3.0 Evolved Shinka Pipeline auto-resume on power-on"
    $scriptPath = Join-Path $PSScriptRoot "power_on_auto_resume.ps1"
    $workingDir = $PSScriptRoot

    Write-Log "Installing scheduled task: $taskName" "INFO" "Scheduler"
    Write-Log "Script path: $scriptPath" "INFO" "Scheduler"

    $action = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-WindowStyle Hidden -ExecutionPolicy Bypass -File `"$scriptPath`""

    $trigger = New-ScheduledTaskTrigger -AtStartup

    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -RunOnlyIfNetworkAvailable

    try {
        $existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
        if ($existingTask) {
            Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction Stop
            Write-Log "Removed existing scheduled task" "INFO" "Scheduler"
        }

        Register-ScheduledTask `
            -TaskName $taskName `
            -Action $action `
            -Trigger $trigger `
            -Description $taskDescription `
            -Settings $settings `
            -RunLevel Highest `
            -ErrorAction Stop | Out-Null

        Write-Log "Scheduled task installed successfully" "SUCCESS" "Scheduler"
        return $true
    } catch {
        Write-Log "Failed to install scheduled task: $($_.Exception.Message)" "ERROR" "Scheduler"
        return $false
    }
}

function Remove-OldLogs {
    $BaseDir = "$PSScriptRoot\..\.."
    $LogDir = Join-Path $BaseDir "logs"
    $cutoffDate = (Get-Date).AddDays(-$Script:Config.LogRetentionDays)

    if (-not (Test-Path $LogDir)) { return }

    try {
        Get-ChildItem -Path $LogDir -Filter "*.log" -File | Where-Object {
            $_.LastWriteTime -lt $cutoffDate
        } | ForEach-Object {
            Remove-Item $_.FullName -Force
            Write-Log "Removed old log: $($_.Name)" "INFO" "Cleanup"
        }
    } catch {
        Write-Log "Failed to clean old logs: $($_.Exception.Message)" "WARNING" "Cleanup"
    }
}

function Show-PipelineStatus {
    $BaseDir = "$PSScriptRoot\..\.."
    $StateFilePath = Join-Path $BaseDir "data\evolved_pipeline_state.json"
    $CheckpointDir = Join-Path $BaseDir "checkpoints\evolved_pipeline"

    $state = Get-PipelineState
    $latestCheckpoint = Get-LatestCheckpoint

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Pipeline Status" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""

    if ($state.Exists) {
        Write-Host "  State File: EXISTS" -ForegroundColor $(if ($state.IsCompleted) { "Green" } else { "Yellow" })
        Write-Host "  Current Phase: $($state.CurrentPhase)" -ForegroundColor White
        Write-Host "  Completed: $($state.IsCompleted)" -ForegroundColor White
        Write-Host "  Total Samples: $($state.TotalSamples)" -ForegroundColor White
        Write-Host "  Checkpoints: $($state.CheckpointCount)" -ForegroundColor White
        if ($state.StartTime) {
            Write-Host "  Start Time: $($state.StartTime)" -ForegroundColor White
        }
    } else {
        Write-Host "  State File: NOT FOUND" -ForegroundColor Red
    }

    if ($latestCheckpoint) {
        Write-Host "  Latest Checkpoint: $latestCheckpoint" -ForegroundColor Green
    } else {
        Write-Host "  Latest Checkpoint: NONE" -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
}

# ============================================================================
# Main Execution
# ============================================================================

function Start-Main {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Evolved Shinka Pipeline Auto-Resume" -ForegroundColor Cyan
    Write-Host "  Version $Script:Version" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""

    Initialize-Logging

    if ($InstallScheduler) {
        Install-ScheduledTask
        return
    }

    $startTime = Get-Date

    Show-PipelineStatus

    Write-Log "Checking prerequisites..." "INFO" "Startup"

    $ollamaHealthy = Test-OllamaHealth -Url $Script:Config.OllamaUrl -Timeout $OllamaTimeout

    if (-not $ollamaHealthy) {
        Write-Log "Ollama is not available. Waiting..." "WARNING" "Ollama"

        for ($i = 1; $i -le $OllamaTimeout / 10; $i++) {
            Start-Sleep -Seconds 10
            Write-Log "Waiting for Ollama... ($i)" "INFO" "Ollama"
            if (Test-OllamaHealth -Url $Script:Config.OllamaUrl -Timeout 10) {
                break
            }
        }
    }

    $state = Get-PipelineState -StateFilePath $Script:StateFile
    $latestCheckpoint = Get-LatestCheckpoint -CheckpointDir $Script:CheckpointDir

    $shouldResume = $false
    $resumeReason = ""

    if ($state.Exists -and -not $state.IsCompleted) {
        $shouldResume = $true
        $resumeReason = "Pipeline was not completed (phase: $($state.CurrentPhase))"
    } elseif ($latestCheckpoint) {
        $shouldResume = $true
        $resumeReason = "Checkpoint available at: $latestCheckpoint"
    }

    if ($Force) {
        $shouldResume = $false
        $resumeReason = "Force restart requested"
        Write-Log "Force restart requested" "INFO" "Startup"
    }

    if ($shouldResume) {
        Write-Log "Will resume: $resumeReason" "INFO" "Resume"
        $result = Start-Pipeline `
            -PythonExe $Script:PythonExe `
            -ScriptPath $Script:PipelineScript `
            -Resume:$true `
            -ConfigPath $ConfigPath `
            -MaxRetries $MaxRetries
    } else {
        Write-Log "Starting fresh pipeline execution" "INFO" "Startup"
        $result = Start-Pipeline `
            -PythonExe $Script:PythonExe `
            -ScriptPath $Script:PipelineScript `
            -ConfigPath $ConfigPath `
            -MaxRetries $MaxRetries
    }

    Remove-OldLogs

    $endTime = Get-Date
    $duration = New-TimeSpan -Start $startTime -End $endTime

    Write-Log "========================================" "INFO" "Summary"
    Write-Log "Execution Summary" "INFO" "Summary"
    Write-Log "Duration: $($duration.ToString('hh\\:mm\\:ss'))" "INFO" "Summary"
    Write-Log "Result: $(if ($result.Success) { 'SUCCESS' } else { 'FAILED' })" `
        $(if ($result.Success) { "SUCCESS" } else { "ERROR" }) "Summary"

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Execution Complete" -ForegroundColor Cyan
    Write-Host "  Duration: $($duration.ToString('hh\\:mm\\:ss'))" -ForegroundColor White
    Write-Host "  Result: $(if ($result.Success) { 'SUCCESS' } else { 'FAILED' })" `
        -ForegroundColor $(if ($result.Success) { "Green" } else { "Red" })
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""

    return $result
}

$Script:ExecutionResult = Start-Main

if (-not $ExecutionResult.Success -and -not $DryRun) {
    exit 1
}
