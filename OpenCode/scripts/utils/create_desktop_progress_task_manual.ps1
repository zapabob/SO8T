# Manual SO8T Desktop Progress Display Task Creation
# This script provides instructions and attempts to create the task

Write-Host "SO8T Desktop Progress Display Task Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$ProgressScript = "$PSScriptRoot\show_progress_report.ps1"
$TaskName = "SO8T_Desktop_Progress_Display"

Write-Host "This script will set up a Windows Task Scheduler task to display" -ForegroundColor White
Write-Host "the SO8T progress report on your desktop at every Windows logon." -ForegroundColor White
Write-Host ""

# Check if script exists
if (!(Test-Path $ProgressScript)) {
    Write-Host "ERROR: Progress script not found at: $ProgressScript" -ForegroundColor Red
    exit 1
}

Write-Host "Progress script location: $ProgressScript" -ForegroundColor Green
Write-Host ""

# Method 1: Try to create task using PowerShell cmdlets
Write-Host "Attempting to create task using PowerShell..." -ForegroundColor Yellow

try {
    # Remove existing task if it exists
    $existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existingTask) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Removed existing task." -ForegroundColor Yellow
    }

    # Create task action
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -File `"$ProgressScript`""

    # Create trigger (at logon)
    $trigger = New-ScheduledTaskTrigger -AtLogOn

    # Create principal (current user)
    $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType InteractiveToken

    # Create settings
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

    # Register task
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description "SO8T Progress Report Display on Desktop at Logon"

    Write-Host "SUCCESS: Task created successfully!" -ForegroundColor Green
    Write-Host "The progress report will now display on desktop at every logon." -ForegroundColor Green

} catch {
    Write-Host "PowerShell method failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""

    # Method 2: Provide manual instructions
    Write-Host "MANUAL SETUP INSTRUCTIONS:" -ForegroundColor Yellow
    Write-Host "==========================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "1. Open Task Scheduler (taskschd.msc) as Administrator" -ForegroundColor White
    Write-Host "2. Click 'Create Task...' in the right panel" -ForegroundColor White
    Write-Host "3. General Tab:" -ForegroundColor White
    Write-Host "   - Name: SO8T_Desktop_Progress_Display" -ForegroundColor Gray
    Write-Host "   - Check 'Run with highest privileges'" -ForegroundColor Gray
    Write-Host "   - Check 'Run only when user is logged on'" -ForegroundColor Gray
    Write-Host "4. Triggers Tab:" -ForegroundColor White
    Write-Host "   - New... -> At log on" -ForegroundColor Gray
    Write-Host "   - User: $env:USERNAME" -ForegroundColor Gray
    Write-Host "5. Actions Tab:" -ForegroundColor White
    Write-Host "   - New... -> Start a program" -ForegroundColor Gray
    Write-Host "   - Program: powershell.exe" -ForegroundColor Gray
    Write-Host "   - Arguments: -ExecutionPolicy Bypass -File `"$ProgressScript`"" -ForegroundColor Gray
    Write-Host "6. Conditions Tab:" -ForegroundColor White
    Write-Host "   - Uncheck 'Start the task only if the computer is on AC power'" -ForegroundColor Gray
    Write-Host "7. Settings Tab:" -ForegroundColor White
    Write-Host "   - Check 'Allow task to be run on demand'" -ForegroundColor Gray
    Write-Host "   - Check 'Run task as soon as possible after a scheduled start is missed'" -ForegroundColor Gray
    Write-Host "8. Click OK to save" -ForegroundColor White
    Write-Host ""
    Write-Host "After setup, the progress report will appear on desktop at every logon!" -ForegroundColor Green
}

Write-Host ""
Write-Host "To test the setup, you can run:" -ForegroundColor Cyan
Write-Host ".\scripts\utils\show_progress_report.ps1" -ForegroundColor White
