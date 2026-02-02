# SO8T Progress Report Script
# Displays comprehensive project progress using PowerShell

param(
    [switch]$Detailed,
    [switch]$Training,
    [switch]$Data,
    [switch]$Logs
)

# Set console encoding to UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  SO8T Progress Report" -ForegroundColor Cyan
Write-Host "  Borea-Phi3.5-instinct-jp Enhancement" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get project root
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

Write-Host "Report Time: $Timestamp" -ForegroundColor Gray
Write-Host "Project Root: $ProjectRoot" -ForegroundColor Gray
Write-Host ""

# ================================================================================
# DATASET STATUS
# ================================================================================
if (!$Training -and !$Logs) {
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  DATASET STATUS" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow

    $DatasetPath = Join-Path $ProjectRoot "data\integrated\so8t_integrated_training_dataset_utf8.jsonl"
    if (Test-Path $DatasetPath) {
        $DatasetSize = (Get-Item $DatasetPath).Length / 1MB
        $DatasetSizeStr = "{0:N2} MB" -f $DatasetSize
        Write-Host "[OK] Dataset: $DatasetSizeStr (59.2MB expected)" -ForegroundColor Green
    } else {
        Write-Host "[NG] Dataset: NOT FOUND" -ForegroundColor Red
    }

    $ConfigPath = Join-Path $ProjectRoot "configs\train_borea_phi35_so8t_thinking.yaml"
    if (Test-Path $ConfigPath) {
        Write-Host "[OK] Config: train_borea_phi35_so8t_thinking.yaml" -ForegroundColor Green
    } else {
        Write-Host "[NG] Config: NOT FOUND" -ForegroundColor Red
    }

    Write-Host ""
}

# ================================================================================
# TRAINING SESSIONS STATUS
# ================================================================================
if (!$Data -and !$Logs) {
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  TRAINING SESSIONS STATUS" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow

    $TrainingBase = "D:/webdataset/checkpoints/training"
    $Sessions = Get-ChildItem -Path $TrainingBase -Directory -Filter "so8t_*" -ErrorAction SilentlyContinue

    if ($Sessions) {
        $Sessions | Sort-Object LastWriteTime -Descending | ForEach-Object {
            $Session = $_.FullName
            $SessionName = $_.Name
            $StatusFile = Join-Path $Session "training_status.json"
            $FinalModel = Join-Path $Session "final_model"
            $Checkpoints = Get-ChildItem -Path $Session -Filter "checkpoint_*" -ErrorAction SilentlyContinue

            Write-Host "Session: $SessionName" -ForegroundColor White

            # Status check
            if (Test-Path $StatusFile) {
                try {
                    $Status = Get-Content $StatusFile -Raw | ConvertFrom-Json
                    switch ($Status.status) {
                        "completed" {
                            Write-Host "  Status: [OK] COMPLETED" -ForegroundColor Green
                        }
                        "running" {
                            Write-Host "  Status: [RUNNING] ACTIVE" -ForegroundColor Yellow
                        }
                        "failed" {
                            Write-Host "  Status: [ERROR] FAILED" -ForegroundColor Red
                        }
                        "interrupted" {
                            Write-Host "  Status: [PAUSED] INTERRUPTED" -ForegroundColor Yellow
                        }
                        default {
                            Write-Host "  Status: [UNKNOWN] $($Status.status)" -ForegroundColor Gray
                        }
                    }
                    if ($Status.start_time) {
                        Write-Host "  Started: $($Status.start_time)" -ForegroundColor Gray
                    }
                } catch {
                    Write-Host "  Status: [WARNING] INVALID STATUS FILE" -ForegroundColor Yellow
                }
            } else {
                Write-Host "  Status: [NO FILE] NO STATUS FILE" -ForegroundColor Gray
            }

            # Completion check
            if (Test-Path $FinalModel) {
                Write-Host "  Completion: [COMPLETE] FINAL MODEL EXISTS" -ForegroundColor Green
            } else {
                Write-Host "  Completion: [IN PROGRESS] TRAINING ACTIVE" -ForegroundColor Yellow
            }

            # Checkpoint count
            $CheckpointCount = $Checkpoints | Measure-Object | Select-Object -ExpandProperty Count
            if ($CheckpointCount -gt 0) {
                Write-Host "  Checkpoints: $CheckpointCount files" -ForegroundColor Gray
            } else {
                Write-Host "  Checkpoints: 0 files" -ForegroundColor Gray
            }

            Write-Host ""
        }
    } else {
        Write-Host "[INFO] No training sessions found" -ForegroundColor Yellow
        Write-Host ""
    }
}

# ================================================================================
# IMPLEMENTATION LOGS STATUS
# ================================================================================
if (!$Data -and !$Training) {
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  IMPLEMENTATION LOGS" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow

    Write-Host "[LOGS] Implementation logs: Available in _docs/ directory" -ForegroundColor Green
    Write-Host "  - Use -Detailed flag for full listing" -ForegroundColor Gray

    Write-Host ""
}

# ================================================================================
# SYSTEM STATUS
# ================================================================================
if (!$Data -and !$Training -and !$Logs) {
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  SYSTEM STATUS" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow

    # Auto-resume script
    $AutoResumeScript = Join-Path $PSScriptRoot "..\training\auto_power_on_resume.bat"
    if (Test-Path $AutoResumeScript) {
        Write-Host "[OK] Auto-resume script: EXISTS" -ForegroundColor Green
    } else {
        Write-Host "[NG] Auto-resume script: NOT FOUND" -ForegroundColor Red
    }

    # Task scheduler
    try {
        $TaskOutput = schtasks /query /tn "SO8T_PowerOn_Resume" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] Power-on task: REGISTERED" -ForegroundColor Green
        } else {
            Write-Host "[NG] Power-on task: NOT REGISTERED" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "[ERROR] Power-on task: ERROR CHECKING" -ForegroundColor Red
    }

    # Audio notification
    $AudioFile = Join-Path $ProjectRoot ".cursor\marisa_owattaze.wav"
    if (Test-Path $AudioFile) {
        Write-Host "[OK] Audio notification: EXISTS" -ForegroundColor Green
    } else {
        Write-Host "[NG] Audio notification: NOT FOUND" -ForegroundColor Yellow
    }

    Write-Host ""
}

# ================================================================================
# OVERALL PROGRESS SUMMARY
# ================================================================================
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  OVERALL PROGRESS SUMMARY" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$CompletedTasks = @(
    "Dataset acquisition and integration",
    "Four-class labeling and cleansing",
    "SO8T pipeline integration",
    "Rolling checkpoint system (3min/5stock)",
    "Power-on auto-resume system",
    "Audio notification system",
    "Implementation documentation"
)

$InProgressTasks = @(
    "Final training execution",
    "Performance measurement (/thinking, CoT, MCP)",
    "Model validation and testing"
)

Write-Host "[COMPLETE] COMPLETED ($($CompletedTasks.Count) items):" -ForegroundColor Green
$CompletedTasks | ForEach-Object { Write-Host "  - $_" -ForegroundColor Green }

Write-Host ""
Write-Host "[IN PROGRESS] ACTIVE ($($InProgressTasks.Count) items):" -ForegroundColor Yellow
$InProgressTasks | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }

Write-Host ""
$ProgressPercent = [math]::Round(($CompletedTasks.Count / ($CompletedTasks.Count + $InProgressTasks.Count)) * 100)
Write-Host "[PROGRESS] PROJECT COMPLETION: $ProgressPercent%" -ForegroundColor Cyan

# Next steps
Write-Host ""
Write-Host "[NEXT] RECOMMENDED ACTIONS:" -ForegroundColor White
Write-Host "  1. Run training: .\scripts\training\auto_power_on_resume.bat" -ForegroundColor White
Write-Host "  2. Monitor progress: .\scripts\utils\show_progress_report.ps1 -Detailed" -ForegroundColor White
Write-Host "  3. Check logs: Get-Content _docs\*.md | Select-String 'Status|Progress'" -ForegroundColor White

Write-Host ""
Write-Host "Report completed at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
