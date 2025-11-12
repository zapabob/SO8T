# Git Commit with Audio Notification
# コミット前に魔理沙のおわったぜを鳴らしてからコミットするスクリプト

param(
    [Parameter(Mandatory=$true)]
    [string]$Message,
    
    [switch]$All,
    [string[]]$Files
)

# 音声通知を再生
Write-Host "[AUDIO] Playing completion notification before commit..." -ForegroundColor Green
& "$PSScriptRoot\play_audio_notification.ps1"

# 少し待機（音声再生完了を待つ）
Start-Sleep -Seconds 1

# Gitステージング
if ($All) {
    Write-Host "[GIT] Staging all changes..." -ForegroundColor Cyan
    git add -A
} elseif ($Files) {
    Write-Host "[GIT] Staging specified files..." -ForegroundColor Cyan
    foreach ($file in $Files) {
        git add $file
    }
} else {
    Write-Host "[GIT] Staging all changes..." -ForegroundColor Cyan
    git add -A
}

# コミット
Write-Host "[GIT] Committing changes..." -ForegroundColor Cyan
git commit -m $Message

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Commit completed successfully!" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Commit failed!" -ForegroundColor Red
    exit 1
}













