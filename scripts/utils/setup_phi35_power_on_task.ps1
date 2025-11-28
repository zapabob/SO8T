# Phi-3.5 Pipeline Power-On Task Setup
# Windows Task Schedulerで電源投入時に自動再開を設定

Write-Host "=== Phi-3.5 Pipeline Power-On Task Setup ===" -ForegroundColor Green

# 管理者権限チェック
$IsAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (!$IsAdmin) {
    Write-Host "Administrator privileges required. Please run as administrator." -ForegroundColor Red
    exit 1
}

# タスク名
$TaskName = "Phi35_SO8T_Pipeline_PowerOn_Resume"
$ScriptPath = "$PSScriptRoot\..\pipeline\phi35_power_on_resume.ps1"

# スクリプト存在確認
if (!(Test-Path $ScriptPath)) {
    Write-Host "Power-on resume script not found: $ScriptPath" -ForegroundColor Red
    exit 1
}

Write-Host "Setting up Task Scheduler task: $TaskName" -ForegroundColor Cyan
Write-Host "Script path: $ScriptPath" -ForegroundColor Cyan

# 既存タスクの削除（存在する場合）
$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($ExistingTask) {
    Write-Host "Removing existing task..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# PowerShell実行ポリシーの設定（一時的に）
$PowerShellCmd = @"
powershell.exe -ExecutionPolicy Bypass -File "$ScriptPath" -Force
"@

# タスク作成
try {
    $Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -File `"$ScriptPath`" -Force"
    $Trigger = New-ScheduledTaskTrigger -AtLogon
    $Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
    $Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType InteractiveToken

    $Task = New-ScheduledTask -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal

    Register-ScheduledTask -TaskName $TaskName -InputObject $Task -Description "Phi-3.5 SO8T Pipeline Power-On Auto Resume"

    Write-Host "Task created successfully!" -ForegroundColor Green
    Write-Host "Task will run automatically at user logon (power-on)." -ForegroundColor Green

} catch {
    Write-Host "Failed to create task: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# タスク確認
$CreatedTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($CreatedTask) {
    Write-Host "Task verification:" -ForegroundColor Cyan
    Write-Host "  Name: $($CreatedTask.TaskName)" -ForegroundColor White
    Write-Host "  State: $($CreatedTask.State)" -ForegroundColor White
    Write-Host "  Triggers: $($CreatedTask.Triggers.Count)" -ForegroundColor White
    Write-Host "  Actions: $($CreatedTask.Actions.Count)" -ForegroundColor White
}

Write-Host "" -ForegroundColor White
Write-Host "=== Setup Complete ===" -ForegroundColor Green
Write-Host "The Phi-3.5 pipeline will now automatically resume on power-on/logon." -ForegroundColor Green
Write-Host "You can also manually run the resume check with:" -ForegroundColor Cyan
Write-Host "  .\scripts\pipeline\phi35_power_on_resume.ps1" -ForegroundColor White
Write-Host "" -ForegroundColor White

# オーディオ通知
try {
    & "$PSScriptRoot\play_audio_notification.ps1"
} catch {
    Write-Host "Audio notification failed" -ForegroundColor Yellow
}
