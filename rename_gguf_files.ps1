# Rename GGUF files from agiasi to AEGIS
Write-Host "[RENAME] Starting GGUF file rename from agiasi to AEGIS..."

Get-ChildItem -Path "D:\webdataset\gguf_models" -Recurse -File -Filter "*.gguf" | Where-Object { $_.Name -match "agiasi" } | ForEach-Object {
    $oldName = $_.Name
    $newName = $oldName -replace "agiasi", "AEGIS"

    Write-Host "[RENAME] $oldName -> $newName"
    Rename-Item -Path $_.FullName -NewName $newName -Force
}

Write-Host "[OK] GGUF file rename completed!"
