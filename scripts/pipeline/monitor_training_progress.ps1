<#
.SYNOPSIS
    AEGIS-v3.0 Training Progress Monitor
.DESCRIPTION
    リアルタイムで GPU 学習の進捗を監視し、エラーログを表示
.NOTES
    5分ローリングチェックポイント (3世代) + 電源投入時自動再開対応
#>

param(
    [string]$LogPath = "logs\sft_progress.log",
    [int]$RefreshSeconds = 2
)

$Host.UI.RawUI.WindowTitle = "AEGIS-v3.0 Training Monitor"

Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "      AEGIS-v3.0 GPU TRAINING PROGRESS MONITOR" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host ""
Write-Host "[CONFIG] Log file: $LogPath" -ForegroundColor Gray
Write-Host "[CONFIG] Refresh interval: ${RefreshSeconds}s" -ForegroundColor Gray
Write-Host "[CONFIG] Checkpoint: 5min rolling (3 generations)" -ForegroundColor Gray
Write-Host ""
Write-Host "Press Ctrl+C to exit" -ForegroundColor Yellow
Write-Host ""

$lastPosition = 0
$errorCount = 0
$warningCount = 0
$stepCount = 0

while ($true) {
    if (Test-Path $LogPath) {
        $content = Get-Content $LogPath -Raw -ErrorAction SilentlyContinue
        if ($content -and $content.Length -gt $lastPosition) {
            $newContent = $content.Substring($lastPosition)
            $lastPosition = $content.Length
            
            # 行ごとに処理
            $lines = $newContent -split "`n"
            foreach ($line in $lines) {
                if ([string]::IsNullOrWhiteSpace($line)) { continue }
                
                # エラー検出
                if ($line -match "ERROR|CRITICAL|Exception|Traceback") {
                    Write-Host "[ERROR] $line" -ForegroundColor Red
                    $errorCount++
                }
                # 警告検出
                elseif ($line -match "WARNING|WARN") {
                    Write-Host "[WARN] $line" -ForegroundColor Yellow
                    $warningCount++
                }
                # 進捗検出 (loss, step, %)
                elseif ($line -match "loss|step|%|\d+/\d+") {
                    Write-Host "[PROGRESS] $line" -ForegroundColor Green
                    $stepCount++
                }
                # チェックポイント検出
                elseif ($line -match "checkpoint|saved|saving") {
                    Write-Host "[CHECKPOINT] $line" -ForegroundColor Magenta
                }
                # 通常ログ
                else {
                    Write-Host "[INFO] $line" -ForegroundColor White
                }
            }
        }
    }
    else {
        Write-Host "[WAITING] Log file not found: $LogPath" -ForegroundColor DarkGray
    }
    
    # ステータスバー
    $status = "Steps: $stepCount | Errors: $errorCount | Warnings: $warningCount"
    Write-Host "`r$status" -NoNewline -ForegroundColor DarkCyan
    
    Start-Sleep -Seconds $RefreshSeconds
}
