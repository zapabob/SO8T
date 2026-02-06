# AEGIS Pipeline Persistent Runner
# Checks if pipeline is running, keeping it alive.
# Add shortcut of this script to shell:startup for auto-resume on boot.

$ScriptPath = "$PSScriptRoot\..\src\run_aegis_pipeline.py"
$PythonPath = "py"  # Assumes 'py' launcher is available

Write-Host "Initializing AEGIS Persistent Runner..." -ForegroundColor Cyan

while ($true) {
    $proc = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*run_aegis_pipeline*" }
    
    if (-not $proc) {
        Write-Host "[$(Get-Date)] Pipeline not running. Starting..." -ForegroundColor Yellow
        
        # Set Environment Variables for Robustness
        $env:PYTHONPATH = "$PSScriptRoot\.."
        $env:SO8T_CHECKPOINT_ROLLING = "3"
        $env:SO8T_CHECKPOINT_INTERVAL = "300"
        
        # Start Process
        Start-Process $PythonPath -ArgumentList "-3", "$ScriptPath", "--phase", "all", "--resume" -NoNewWindow -Wait
        
        Write-Host "[$(Get-Date)] Pipeline process ended. Restarting in 10 seconds..." -ForegroundColor Red
        Start-Sleep -Seconds 10
    }
    else {
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 30
    }
}
