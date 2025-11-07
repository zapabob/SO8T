$audioPath = "C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav"
if (Test-Path $audioPath) {
    Write-Host "[INFO] Playing audio: $audioPath" -ForegroundColor Cyan
    Add-Type -AssemblyName System.Windows.Forms
    $player = New-Object System.Media.SoundPlayer($audioPath)
    $player.PlaySync()
    Write-Host "[OK] Audio notification played successfully" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Audio file not found: $audioPath" -ForegroundColor Red
}

