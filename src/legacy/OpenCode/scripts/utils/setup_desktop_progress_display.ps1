# SO8T Desktop Progress Display Setup Script
# Sets up Windows Task Scheduler to display progress report on desktop at power-on

param(
    [switch]$Install,
    [switch]$Uninstall,
    [switch]$Test,
    [switch]$Run
)

$ScriptPath = Split-Path -Parent $PSScriptRoot
$ProgressScript = Join-Path $PSScriptRoot "show_progress_report.ps1"
$TaskName = "SO8T_Desktop_Progress_Display"

Write-Host "SO8T Desktop Progress Display Setup" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan

if ($Run) {
    Write-Host "Running progress report directly..." -ForegroundColor Yellow
    if (Test-Path $ProgressScript) {
        & $ProgressScript
    } else {
        Write-Host "Progress script not found: $ProgressScript" -ForegroundColor Red
    }
    exit
}

if ($Test) {
    Write-Host "Testing progress report execution..." -ForegroundColor Yellow
    if (Test-Path $ProgressScript) {
        Write-Host "Progress script found: $ProgressScript" -ForegroundColor Green
        Write-Host "Testing execution..." -ForegroundColor Yellow

        # Test execution in new window
        Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$ProgressScript`"" -Wait
        Write-Host "Test completed." -ForegroundColor Green
    } else {
        Write-Host "Progress script not found: $ProgressScript" -ForegroundColor Red
    }
    exit
}

if ($Uninstall) {
    Write-Host "Uninstalling SO8T desktop progress display task..." -ForegroundColor Yellow

    try {
        # Remove existing task
        $existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        if ($existingTask) {
            Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
            Write-Host "Task '$TaskName' uninstalled successfully." -ForegroundColor Green
        } else {
            Write-Host "Task '$TaskName' not found." -ForegroundColor Yellow
        }
    } catch {
        Write-Host "Error uninstalling task: $($_.Exception.Message)" -ForegroundColor Red
    }
    exit
}

if ($Install) {
    Write-Host "Installing SO8T desktop progress display task..." -ForegroundColor Yellow

    # Check if progress script exists
    if (!(Test-Path $ProgressScript)) {
        Write-Host "Error: Progress script not found at $ProgressScript" -ForegroundColor Red
        exit 1
    }

    try {
        # Remove existing task if it exists
        $existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        if ($existingTask) {
            Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
            Write-Host "Removed existing task '$TaskName'." -ForegroundColor Yellow
        }

        # Create new task action - PowerShell with visible window
        $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -File `"$ProgressScript`""

        # Trigger: At logon (when user logs in after power-on)
        $trigger = New-ScheduledTaskTrigger -AtLogOn

        # Principal: Current user, interactive
        $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType InteractiveToken

        # Settings: Allow start on battery, don't stop on battery, start when available
        $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

        # Register the task
        Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description "SO8T Progress Report Display on Desktop at Power-On"

        Write-Host "Task '$TaskName' installed successfully." -ForegroundColor Green
        Write-Host "The progress report will now display on desktop at every logon (power-on)." -ForegroundColor Green
        Write-Host "Task will run with visible PowerShell window." -ForegroundColor Green

    } catch {
        Write-Host "Error installing task: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Make sure you run this script as Administrator." -ForegroundColor Yellow
    }

    exit
}

# Help display
Write-Host "Usage:" -ForegroundColor White
Write-Host "  .\setup_desktop_progress_display.ps1 -Install    # Install desktop display task" -ForegroundColor White
Write-Host "  .\setup_desktop_progress_display.ps1 -Uninstall  # Uninstall desktop display task" -ForegroundColor White
Write-Host "  .\setup_desktop_progress_display.ps1 -Test       # Test the progress display" -ForegroundColor White
Write-Host "  .\setup_desktop_progress_display.ps1 -Run        # Run progress report directly" -ForegroundColor White
Write-Host "" -ForegroundColor White
Write-Host "Note: Install/Uninstall operations require Administrator privileges." -ForegroundColor Yellow
Write-Host "The task will display SO8T progress report on desktop at every Windows logon." -ForegroundColor Cyan
