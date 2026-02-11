param(
  [switch]$RemoveStartupShortcuts
)

$ErrorActionPreference = "Continue"

$tasks = @(
  "SO8T_Power_On_Startup",
  "Advanced_Science_Pipeline_Power_On",
  "SO8T_AEGIS_Automatic_Pipeline",
  "SO8T_AEGIS_Automatic_Pipeline",
  "AEGIS_AB_Test_Automation",
  "AEGIS_AB_Test_Daily",
  "SO8T_GGUF_AB_Test_Auto_Resume",
  "GGUF_Conversion_Auto_Resume"
)

Write-Host "=== SO8T Autostart Disable ==="
Write-Host "Deleting scheduled tasks (requires admin)..."

foreach ($t in $tasks | Select-Object -Unique) {
  schtasks /query /tn $t *> $null
  if ($LASTEXITCODE -eq 0) {
    schtasks /delete /tn $t /f *> $null
    if ($LASTEXITCODE -eq 0) {
      Write-Host "[OK] deleted task: $t"
    } else {
      Write-Host "[FAIL] could not delete task (run PowerShell as Admin): $t"
    }
  } else {
    Write-Host "[SKIP] task not found: $t"
  }
}

if ($RemoveStartupShortcuts) {
  Write-Host ""
  Write-Host "Removing Startup folder shortcuts..."
  $startup = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Startup"
  foreach ($lnk in @("AEGIS_AB_Test_Launch.lnk", "AEGIS_AB_Test_Monitor.lnk")) {
    $p = Join-Path $startup $lnk
    if (Test-Path $p) {
      Remove-Item -Force $p
      Write-Host "[OK] removed: $p"
    } else {
      Write-Host "[SKIP] not found: $p"
    }
  }
}

Write-Host ""
Write-Host "Done."
