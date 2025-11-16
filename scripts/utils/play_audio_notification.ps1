# Audio Notification Script
# BASIC PRINCIPLE: Play marisa_owattaze.wav first, fallback to beep if it fails
# STANDARD: This is the standard audio notification script for SO8T project

$audioFile = "C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav"
$audioPlayed = $false

# Method 1: Try to play marisa_owattaze.wav (PRIMARY METHOD)
if (Test-Path $audioFile) {
    try {
        Add-Type -AssemblyName System.Windows.Forms
        $player = New-Object System.Media.SoundPlayer($audioFile)
        $player.PlaySync()
        Write-Host "[OK] marisa_owattaze.wav played successfully" -ForegroundColor Green
        $audioPlayed = $true
    } catch {
        Write-Host "[WARNING] Failed to play marisa_owattaze.wav: $($_.Exception.Message)" -ForegroundColor Yellow
    }
} else {
    Write-Host "[WARNING] marisa_owattaze.wav not found: $audioFile" -ForegroundColor Yellow
}

# Method 2: Fallback to beep if marisa_owattaze.wav failed
if (-not $audioPlayed) {
    try {
        [System.Console]::Beep(1000, 500)
        Write-Host "[OK] Fallback beep played successfully" -ForegroundColor Green
    } catch {
        Write-Host "[ERROR] All audio methods failed" -ForegroundColor Red
    }
}
