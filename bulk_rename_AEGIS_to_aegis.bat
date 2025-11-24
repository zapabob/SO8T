@echo off
chcp 65001 >nul
echo [RENAME] Starting bulk rename from AEGIS/agiasi to AEGIS/aegis...

powershell -ExecutionPolicy Bypass -Command "
Get-ChildItem -Path '.' -Recurse -File | Where-Object { $_.Name -match 'agiasi|AEGIS' } | ForEach-Object {
    $oldName = $_.Name
    $newName = $oldName -replace 'AEGIS', 'AEGIS' -replace 'agiasi', 'aegis'
    $newPath = Join-Path $_.Directory.FullName $newName
    
    Write-Host \"[RENAME] $oldName -> $newName\"
    Rename-Item -Path $_.FullName -NewName $newName -Force
}
"

echo [OK] Bulk rename completed!
echo [AUDIO] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"
