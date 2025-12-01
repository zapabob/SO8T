# RTX3060 SO8T PPO Pipeline 自動起動タスク作成スクリプト
# 電源投入時に自動実行されるタスクをWindowsタスクスケジューラーに登録

# 管理者権限チェック
$currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
$adminRole = [Security.Principal.WindowsBuiltInRole]::Administrator

if (-not $principal.IsInRole($adminRole)) {
    Write-Host "[ERROR] This script requires administrator privileges!" -ForegroundColor Red
    Write-Host "Please run as administrator and try again." -ForegroundColor Yellow
    exit 1
}

Write-Host "[RTX3060] Creating Power-On Startup Task" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# スクリプトのパスを取得
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent $scriptDir)
$ps1Script = Join-Path $projectRoot "scripts\startup\start_rtx3060_pipeline.ps1"

# タスク名
$taskName = "SO8T_RTX3060_PPO_Pipeline"

# 既存タスクの確認と削除
$existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existingTask) {
    Write-Host "[INFO] Removing existing task..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

# 新しいタスクの作成
Write-Host "[STEP 1] Creating scheduled task..." -ForegroundColor Yellow
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -File `"$ps1Script`""
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType InteractiveToken

try {
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description "RTX3060向けSO8T PPO Pipeline自動起動"
    Write-Host "[SUCCESS] Task created successfully!" -ForegroundColor Green
    Write-Host "Task Name: $taskName" -ForegroundColor White
    Write-Host "Triggers: At logon (power on)" -ForegroundColor White
    Write-Host "Action: $ps1Script" -ForegroundColor White
} catch {
    Write-Host "[ERROR] Failed to create task: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# タスクの確認
Write-Host "[STEP 2] Verifying task creation..." -ForegroundColor Yellow
$createdTask = Get-ScheduledTask -TaskName $taskName
if ($createdTask) {
    Write-Host "[OK] Task verified successfully" -ForegroundColor Green
    Write-Host "State: $($createdTask.State)" -ForegroundColor White
    Write-Host "Next Run Time: $($createdTask.NextRunTime)" -ForegroundColor White
} else {
    Write-Host "[ERROR] Task verification failed!" -ForegroundColor Red
}

Write-Host "[RTX3060] Task creation completed!" -ForegroundColor Cyan
Write-Host "" -ForegroundColor White
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Restart your computer to test the task" -ForegroundColor White
Write-Host "2. Check Windows Event Viewer for any errors" -ForegroundColor White
Write-Host "3. Monitor logs\aegis_v2_ppo_training.log for pipeline execution" -ForegroundColor White

# 完了通知
Write-Host "[AUDIO] Playing completion notification..." -ForegroundColor Green
try {
    $audioFile = Join-Path $projectRoot ".cursor\marisa_owattaze.wav"
    if (Test-Path $audioFile) {
        Add-Type -AssemblyName System.Windows.Forms
        $player = New-Object System.Media.SoundPlayer($audioFile)
        $player.PlaySync()
        Write-Host "[OK] Audio notification played" -ForegroundColor Green
    } else {
        [System.Console]::Beep(1000, 500)
        Write-Host "[OK] Fallback beep played" -ForegroundColor Green
    }
} catch {
    Write-Host "[WARNING] Audio failed" -ForegroundColor Yellow
}
