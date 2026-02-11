@echo off
chcp 65001 >nul
echo [RENAME] Starting GGUF file rename from agiasi to AEGIS...

powershell -ExecutionPolicy Bypass -Command "
Get-ChildItem -Path 'D:\webdataset\gguf_models' -Recurse -File -Filter '*.gguf' | Where-Object { $_.Name -match 'agiasi' } | ForEach-Object {
    $oldName = $_.Name
    $newName = $oldName -replace 'agiasi', 'AEGIS'
    
    Write-Host \"[RENAME] $oldName -> $newName\"
    Rename-Item -Path $_.FullName -NewName $newName -Force
}
"

echo [OK] GGUF file rename completed!
