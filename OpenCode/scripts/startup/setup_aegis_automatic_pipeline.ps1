# AEGIS Automatic Pipeline Task Scheduler Setup
# AEGIS自動パイプラインのタスクスケジューラ設定スクリプト

param(
    [switch]$Remove
)

# 管理者権限チェックと自動昇格
$currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
$adminRole = [Security.Principal.WindowsBuiltInRole]::Administrator

if (-not $principal.IsInRole($adminRole)) {
    Write-Host "🔄 Administrator privileges required. Elevating..." -ForegroundColor Yellow

    # 自分自身を管理者権限で再実行
    $scriptPath = $MyInvocation.MyCommand.Path
    Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$scriptPath`"" -Verb RunAs -Wait

    # 再実行後のタスク確認
    Write-Host ""
    Write-Host "🔍 Checking if task was created..." -ForegroundColor Cyan
    $verifyResult = schtasks /query /tn $taskName 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Task created successfully!" -ForegroundColor Green
        exit 0
    } else {
        Write-Host "❌ Task creation failed even with elevation" -ForegroundColor Red
        exit 1
    }
}

$taskName = "SO8T_AEGIS_Automatic_Pipeline"
# 絶対パスを使用
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$scriptPath = Join-Path -Path $projectRoot -ChildPath "automation\automatic_aegis_phi35_thinking_pipeline.py"
$pythonPath = (Get-Command python).Source

if (-not (Test-Path $scriptPath)) {
    Write-Host "❌ Script not found: $scriptPath" -ForegroundColor Red
    exit 1
}

if ($Remove) {
    Write-Host "🗑️ Removing existing AEGIS pipeline task..." -ForegroundColor Yellow

    try {
        schtasks /delete /tn $taskName /f 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Task removed successfully" -ForegroundColor Green
        } else {
            Write-Host "ℹ️ Task not found or already removed" -ForegroundColor Gray
        }
    } catch {
        Write-Host "❌ Failed to remove task: $($_.Exception.Message)" -ForegroundColor Red
    }

    exit 0
}

Write-Host "🚀 Setting up AEGIS Automatic Pipeline task..." -ForegroundColor Cyan
Write-Host "Task Name: $taskName" -ForegroundColor White
Write-Host "Script: $scriptPath" -ForegroundColor White
Write-Host "Python: $pythonPath" -ForegroundColor White
Write-Host ""

# タスク作成
$taskCommand = "`"$pythonPath`" `"$scriptPath`" --resume"

try {
    Write-Host "Creating task with command: $taskCommand" -ForegroundColor Gray

    # schtasksコマンドを正しく実行
    $result = schtasks /create /tn $taskName /tr $taskCommand /sc ONLOGON /rl HIGHEST /delay 0000:30 /f 2>&1

    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Task created successfully!" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to create task (Exit code: $LASTEXITCODE)" -ForegroundColor Red
        Write-Host "Error output: $result" -ForegroundColor Red
        exit 1
    }

    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Task created successfully!" -ForegroundColor Green
        Write-Host ""
        Write-Host "📋 Task Details:" -ForegroundColor Cyan
        Write-Host "  - Name: $taskName" -ForegroundColor White
        Write-Host "  - Trigger: At logon (power-on)" -ForegroundColor White
        Write-Host "  - Delay: 30 seconds" -ForegroundColor White
        Write-Host "  - Privileges: Highest" -ForegroundColor White
        Write-Host "  - Command: $taskCommand" -ForegroundColor White
        Write-Host ""
        Write-Host "🎯 The AEGIS pipeline will automatically start 30 seconds after each power-on/logon." -ForegroundColor Green
        Write-Host "💡 Use --resume flag to continue from last checkpoint if interrupted." -ForegroundColor Yellow
    } else {
        Write-Host "❌ Failed to create task (Exit code: $LASTEXITCODE)" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ Error creating task: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# タスク確認
Write-Host ""
Write-Host "🔍 Verifying task creation..." -ForegroundColor Yellow

try {
    $verifyResult = schtasks /query /tn $taskName 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Task verification successful" -ForegroundColor Green

        # 詳細表示
        Write-Host ""
        Write-Host "📄 Task Information:" -ForegroundColor Cyan
        schtasks /query /tn $taskName /v /fo list | Select-String -Pattern "TaskName|Status|Logon Mode|Last Run Time|Task To Run" | ForEach-Object {
            Write-Host "  $_" -ForegroundColor White
        }
    } else {
        Write-Host "⚠️ Task verification failed, but creation may have succeeded" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️ Task verification error: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎉 Setup completed! The AEGIS automatic pipeline is now configured for power-on execution." -ForegroundColor Green
Write-Host ""
Write-Host "💡 Tips:" -ForegroundColor Cyan
Write-Host "  - The pipeline includes automatic checkpointing every 3 minutes" -ForegroundColor White
Write-Host "  - Use 'schtasks /query /tn $taskName /v' to check task status" -ForegroundColor White
Write-Host "  - Use 'schtasks /delete /tn $taskName /f' to remove the task" -ForegroundColor White
Write-Host "  - Check logs in 'checkpoints/automatic_aegis/' for progress" -ForegroundColor White

# 音声通知
try {
    $audioScript = Join-Path $PSScriptRoot "utils" "play_audio_notification.ps1"
    if (Test-Path $audioScript) {
        & $audioScript
    }
} catch {
    Write-Host "🔊 Audio notification not available" -ForegroundColor Gray
}
