# Monitor Pipeline Progress & Health
$LogPath = "$PSScriptRoot\..\logs"
$PipelineLog = "$LogPath\aegis_v3_pipeline.log"
$SftLog = "$LogPath\sft_progress.log"

# Set window size
if ($Host.Name -eq 'ConsoleHost') {
    $Host.UI.RawUI.WindowSize = New-Object Management.Automation.Host.Size (120, 40)
}

function Show-Header {
    Clear-Host
    Write-Host "============================" -ForegroundColor Cyan
    Write-Host " AEGIS v3.0 PIPELINE MONITOR" -ForegroundColor Cyan
    Write-Host "============================" -ForegroundColor Cyan
    Write-Host "Time: $(Get-Date)" -ForegroundColor Gray
    
    # Process Status
    $proc = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*run_aegis_pipeline*" -or $_.CommandLine -like "*train_unsloth*" }
    if ($proc) {
        Write-Host "STATUS: " -NoNewline
        Write-Host "RUNNING" -ForegroundColor Green
        Write-Host "PID: $($proc.Id)" -ForegroundColor DarkGray
        Write-Host "CPU: $($proc.CPU)s" -ForegroundColor DarkGray
    }
    else {
        Write-Host "STATUS: " -NoNewline
        Write-Host "STOPPED / WAITING" -ForegroundColor Yellow
    }
    Write-Host "----------------------------" -ForegroundColor DarkGray
}

while ($true) {
    Show-Header
    
    Write-Host "`n[LATEST SFT PROGRESS]" -ForegroundColor Magenta
    if (Test-Path $SftLog) {
        Get-Content $SftLog -Tail 5 | ForEach-Object {
            if ($_ -match "loss") {
                Write-Host $_ -ForegroundColor Green
            }
            elseif ($_ -match "Epoch") {
                Write-Host $_ -ForegroundColor Cyan
            }
            else {
                Write-Host $_
            }
        }
    }
    else {
        Write-Host "Waiting for SFT log..." -ForegroundColor DarkGray
    }

    Write-Host "`n[PIPELINE LOGS (Errors/Warnings)]" -ForegroundColor Red
    if (Test-Path $PipelineLog) {
        Get-Content $PipelineLog -Tail 10 | ForEach-Object {
            if ($_ -match "ERROR" -or $_ -match "Exception") {
                Write-Host $_ -ForegroundColor Red
            }
            elseif ($_ -match "WARNING") {
                Write-Host $_ -ForegroundColor Yellow
            }
            else {
                Write-Host $_ -ForegroundColor Gray
            }
        }
    }

    Start-Sleep -Seconds 5
}
