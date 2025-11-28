# SO8Tトレーニング 電源投入時自動再開設定スクリプト
# このスクリプトはWindowsタスクスケジューラに電源投入時の自動再開タスクを登録します

param(
    [switch]$Install,
    [switch]$Uninstall,
    [switch]$Test
)

$ScriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptPath)
$BatchScript = Join-Path $ScriptPath "auto_power_on_resume.bat"
$TaskName = "SO8T_PowerOn_Resume"

Write-Host "SO8T Power-On Resume Setup Script" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

if ($Test) {
    Write-Host "Testing auto-resume script..." -ForegroundColor Yellow
    if (Test-Path $BatchScript) {
        Write-Host "Batch script found: $BatchScript" -ForegroundColor Green
        & $BatchScript
    } else {
        Write-Host "Batch script not found: $BatchScript" -ForegroundColor Red
    }
    exit
}

if ($Uninstall) {
    Write-Host "Uninstalling SO8T power-on resume task..." -ForegroundColor Yellow

    try {
        # 既存のタスクを削除
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
    Write-Host "Installing SO8T power-on resume task..." -ForegroundColor Yellow

    # バッチスクリプトの存在確認
    if (!(Test-Path $BatchScript)) {
        Write-Host "Error: Batch script not found at $BatchScript" -ForegroundColor Red
        exit 1
    }

    try {
        # 既存のタスクを削除（もし存在する場合）
        $existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        if ($existingTask) {
            Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
            Write-Host "Removed existing task '$TaskName'." -ForegroundColor Yellow
        }

        # 新しいタスクを作成
        $action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$BatchScript`""
        $trigger = New-ScheduledTaskTrigger -AtLogOn  # ログオン時（電源投入時を含む）
        $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType InteractiveToken
        $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

        # タスク登録
        Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description "SO8T Training Auto-Resume on Power-On"

        Write-Host "Task '$TaskName' installed successfully." -ForegroundColor Green
        Write-Host "The task will run automatically when you log on to Windows (including after power-on)." -ForegroundColor Green

    } catch {
        Write-Host "Error installing task: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Make sure you run this script as Administrator." -ForegroundColor Yellow
    }

    exit
}

# ヘルプ表示
Write-Host "Usage:" -ForegroundColor White
Write-Host "  .\setup_power_on_resume.ps1 -Install    # Install the auto-resume task" -ForegroundColor White
Write-Host "  .\setup_power_on_resume.ps1 -Uninstall  # Uninstall the auto-resume task" -ForegroundColor White
Write-Host "  .\setup_power_on_resume.ps1 -Test       # Test the auto-resume script" -ForegroundColor White
Write-Host "" -ForegroundColor White
Write-Host "Note: Install/Uninstall operations require Administrator privileges." -ForegroundColor Yellow






# このスクリプトはWindowsタスクスケジューラに電源投入時の自動再開タスクを登録します

param(
    [switch]$Install,
    [switch]$Uninstall,
    [switch]$Test
)

$ScriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptPath)
$BatchScript = Join-Path $ScriptPath "auto_power_on_resume.bat"
$TaskName = "SO8T_PowerOn_Resume"

Write-Host "SO8T Power-On Resume Setup Script" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

if ($Test) {
    Write-Host "Testing auto-resume script..." -ForegroundColor Yellow
    if (Test-Path $BatchScript) {
        Write-Host "Batch script found: $BatchScript" -ForegroundColor Green
        & $BatchScript
    } else {
        Write-Host "Batch script not found: $BatchScript" -ForegroundColor Red
    }
    exit
}

if ($Uninstall) {
    Write-Host "Uninstalling SO8T power-on resume task..." -ForegroundColor Yellow

    try {
        # 既存のタスクを削除
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
    Write-Host "Installing SO8T power-on resume task..." -ForegroundColor Yellow

    # バッチスクリプトの存在確認
    if (!(Test-Path $BatchScript)) {
        Write-Host "Error: Batch script not found at $BatchScript" -ForegroundColor Red
        exit 1
    }

    try {
        # 既存のタスクを削除（もし存在する場合）
        $existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        if ($existingTask) {
            Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
            Write-Host "Removed existing task '$TaskName'." -ForegroundColor Yellow
        }

        # 新しいタスクを作成
        $action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$BatchScript`""
        $trigger = New-ScheduledTaskTrigger -AtLogOn  # ログオン時（電源投入時を含む）
        $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType InteractiveToken
        $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

        # タスク登録
        Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description "SO8T Training Auto-Resume on Power-On"

        Write-Host "Task '$TaskName' installed successfully." -ForegroundColor Green
        Write-Host "The task will run automatically when you log on to Windows (including after power-on)." -ForegroundColor Green

    } catch {
        Write-Host "Error installing task: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Make sure you run this script as Administrator." -ForegroundColor Yellow
    }

    exit
}

# ヘルプ表示
Write-Host "Usage:" -ForegroundColor White
Write-Host "  .\setup_power_on_resume.ps1 -Install    # Install the auto-resume task" -ForegroundColor White
Write-Host "  .\setup_power_on_resume.ps1 -Uninstall  # Uninstall the auto-resume task" -ForegroundColor White
Write-Host "  .\setup_power_on_resume.ps1 -Test       # Test the auto-resume script" -ForegroundColor White
Write-Host "" -ForegroundColor White
Write-Host "Note: Install/Uninstall operations require Administrator privileges." -ForegroundColor Yellow





