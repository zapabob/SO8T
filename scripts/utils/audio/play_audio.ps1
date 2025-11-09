$audioPath = "C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav"
if (Test-Path $audioPath) {
    try {
        Add-Type -AssemblyName System.Windows.Forms
        $player = New-Object System.Media.SoundPlayer($audioPath)
        $player.PlaySync()
        Write-Host "[OK] Audio notification played successfully" -ForegroundColor Green
    } catch {
        Write-Host "[ERROR] Failed to play audio: $($_.Exception.Message)" -ForegroundColor Red
    }
} else {
    Write-Host "[WARNING] Audio file not found: $audioPath" -ForegroundColor Yellow
}

