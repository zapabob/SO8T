@echo off
chcp 65001 >nul
echo [RENAME] Starting GGUF folder rename from agiasi to AEGIS...

powershell -ExecutionPolicy Bypass -Command "
Get-ChildItem -Path 'D:\webdataset\gguf_models' -Directory | Where-Object { $_.Name -match 'agiasi' } | ForEach-Object {
    $oldName = $_.Name
    $newName = $oldName -replace 'agiasi', 'AEGIS'
    $newPath = Join-Path $_.Parent.FullName $newName
    
    Write-Host \"[RENAME] $oldName -> $newName\"
    Rename-Item -Path $_.FullName -NewName $newName -Force
}
"

echo [OK] GGUF folder rename completed!
echo [AUDIO] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"
