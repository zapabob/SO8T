# Rename GGUF folders from agiasi to AEGIS
Write-Host "[RENAME] Starting GGUF folder rename from agiasi to AEGIS..."

Get-ChildItem -Path "D:\webdataset\gguf_models" -Directory | Where-Object { $_.Name -match "agiasi" } | ForEach-Object {
    $oldName = $_.Name
    $newName = $oldName -replace "agiasi", "AEGIS"
    $newPath = Join-Path $_.Parent.FullName $newName

    Write-Host "[RENAME] $oldName -> $newName"
    Rename-Item -Path $_.FullName -NewName $newName -Force
}

Write-Host "[OK] GGUF folder rename completed!"

# Play audio notification
Write-Host "[AUDIO] Playing completion notification..."
$audioFile = "C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav"
$audioPlayed = $false

# Method 1: Try to play marisa_owattaze.wav (PRIMARY METHOD)
if (Test-Path $audioFile) {
    try {
        Add-Type -AssemblyName System.Windows.Forms
        $player = New-Object System.Media.SoundPlayer $audioFile
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
