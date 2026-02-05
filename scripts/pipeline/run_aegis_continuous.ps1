# AEGIS-v3.0 全自動継続運転システム (Power-on Auto-Resume Wrapper)
# 2026-02-05 Implementation

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent -Path (Split-Path -Parent -Path $PSScriptRoot)
$LogDir = Join-Path $ProjectRoot "logs"
$LogFile = Join-Path $LogDir "pipeline_continuous_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "      AEGIS-v3.0 FULL AUTOMATION & CONTINUOUS OPERATION" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "[INIT] System Start: $(Get-Date)"
Write-Host "[PATH] Project Root: $ProjectRoot"
Write-Host "[LOG] Error Log: $LogFile"

# 1. Python 環境チェック
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

# 2. 自動再開ループ
$MaxRetries = 10
$RetryCount = 0

while ($RetryCount -lt $MaxRetries) {
    Write-Host "[RUN] Executing Auto-Resume Pipeline..." -ForegroundColor Yellow
    $startTime = Get-Date
    
    # ターミナルタイトルの更新
    $host.ui.RawUI.WindowTitle = "AEGIS-v3.0 Pipeline [RUNNING]"

    # プロセス実行 (Tqdm 表示を維持)
    try {
        & py -3 "$ProjectRoot\scripts\pipeline\auto_resume_aegis.py" | Tee-Object -FilePath $LogFile
        
        $exitCode = $LASTEXITCODE
        if ($exitCode -eq 0) {
            Write-Host "[SUCCESS] Pipeline completed successfully at $(Get-Date)." -ForegroundColor Green
            $host.ui.RawUI.WindowTitle = "AEGIS-v3.0 Pipeline [FINISHED]"
            break
        }
        else {
            Write-Host "[WARNING] Pipeline exited with code $exitCode." -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "[CRITICAL] Exception occurred: $_" -ForegroundColor Red
        $_ | Out-File -FilePath $LogFile -Append
    }

    $RetryCount++
    $endTime = Get-Date
    $duration = $endTime - $startTime
    
    if ($duration.TotalSeconds -lt 60) {
        Write-Host "[ALERT] Pipeline crashed too quickly. Waiting 30s before retry..." -ForegroundColor Red
        Start-Sleep -Seconds 30
    }

    Write-Host "[RETRY] Attempting restart ($RetryCount/$MaxRetries)..." -ForegroundColor Magenta
    $host.ui.RawUI.WindowTitle = "AEGIS-v3.0 Pipeline [RETRYING $RetryCount]"
}

if ($RetryCount -ge $MaxRetries) {
    Write-Host "[FATAL] Max retries reached. System execution halted." -ForegroundColor Red
}

Write-Host "[END] Session closed at $(Get-Date)."
