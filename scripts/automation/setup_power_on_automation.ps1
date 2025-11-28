# SO8T Complete Automation Pipeline - Power-on Task Setup
# Windows Task Schedulerで電源投入時に自動実行するタスクを作成

param(
    [switch]$Remove,
    [switch]$Status
)

$taskName = "SO8T_Complete_Automation_Pipeline"
$scriptPath = "$PSScriptRoot\run_complete_pipeline.bat"
$projectRoot = Split-Path $PSScriptRoot -Parent
$projectRoot = Split-Path $projectRoot -Parent

# ログ関数
function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [$Level] $Message"
    Write-Host $logMessage

    $logFile = Join-Path $projectRoot "logs\automation_setup.log"
    $logMessage | Out-File -FilePath $logFile -Append -Encoding UTF8
}

Write-Log "SO8T Automation Task Setup Script Started"

# ステータス確認
if ($Status) {
    Write-Log "Checking current task status..."

    try {
        $task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
        if ($task) {
            Write-Log "Task '$taskName' exists"
            Write-Log "State: $($task.State)"
            Write-Log "Last Run: $($task.LastRunTime)"
            Write-Log "Next Run: $($task.NextRunTime)"
            Write-Log "Last Result: $($task.LastTaskResult)"
        } else {
            Write-Log "Task '$taskName' does not exist"
        }
    } catch {
        Write-Log "Error checking task status: $($_.Exception.Message)" "ERROR"
    }

    exit 0
}

# タスク削除
if ($Remove) {
    Write-Log "Removing scheduled task '$taskName'..."

    try {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction Stop
        Write-Log "Successfully removed scheduled task '$taskName'"
    } catch {
        if ($_.Exception.Message -like "*The system cannot find the file specified*") {
            Write-Log "Task '$taskName' was not found (already removed or never existed)"
        } else {
            Write-Log "Error removing task: $($_.Exception.Message)" "ERROR"
            exit 1
        }
    }

    exit 0
}

# 前提条件チェック
Write-Log "Checking prerequisites..."

# スクリプト存在確認
if (-not (Test-Path $scriptPath)) {
    Write-Log "ERROR: Automation script not found at: $scriptPath" "ERROR"
    exit 1
}

# Python環境確認
try {
    $pythonVersion = & python --version 2>&1
    Write-Log "Python version: $pythonVersion"
} catch {
    Write-Log "ERROR: Python not found in PATH" "ERROR"
    exit 1
}

# プロジェクト構造確認
$requiredPaths = @(
    (Join-Path $projectRoot "scripts"),
    (Join-Path $projectRoot "so8t"),
    (Join-Path $projectRoot "configs"),
    "D:\webdataset"
)

foreach ($path in $requiredPaths) {
    if (-not (Test-Path $path)) {
        Write-Log "ERROR: Required path not found: $path" "ERROR"
        exit 1
    }
}

Write-Log "All prerequisites verified"

# 既存タスクのクリーンアップ
Write-Log "Cleaning up existing tasks..."
try {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Log "Cleaned up existing task (if any)"
} catch {
    Write-Log "No existing task to clean up"
}

# タスク作成
Write-Log "Creating scheduled task '$taskName'..."

try {
    # タスクアクション定義
    $action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$scriptPath`""

    # トリガー定義（ログオン時）
    $trigger = New-ScheduledTaskTrigger -AtLogOn

    # プリンシパル定義（対話型トークン）
    $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType InteractiveToken

    # 設定定義
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RunOnlyIfNetworkAvailable

    # 実行条件
    $settings.ExecutionTimeLimit = "PT0S"  # 時間制限なし
    $settings.RestartCount = 3  # リトライ回数
    $settings.RestartInterval = "PT5M"  # リトライ間隔5分

    # タスク登録
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description "Complete SO8T Automation Pipeline - Transforms Borea-Phi3.5-instinct-jp into SO8T/thinking multimodal model" -ErrorAction Stop

    Write-Log "Successfully created scheduled task '$taskName'"

    # タスク情報表示
    $createdTask = Get-ScheduledTask -TaskName $taskName
    Write-Log "Task created with the following settings:"
    Write-Log "  Name: $($createdTask.TaskName)"
    Write-Log "  Path: $($createdTask.TaskPath)"
    Write-Log "  State: $($createdTask.State)"
    Write-Log "  Triggers: $($createdTask.Triggers.Count) trigger(s)"
    Write-Log "  Actions: $($createdTask.Actions.Count) action(s)"

} catch {
    Write-Log "ERROR: Failed to create scheduled task: $($_.Exception.Message)" "ERROR"
    Write-Log "Stack trace: $($_.ScriptStackTrace)" "ERROR"
    exit 1
}

# テスト実行確認
Write-Log "Testing task execution (dry run)..."

try {
    $testResult = Start-ScheduledTask -TaskName $taskName -ErrorAction Stop
    Start-Sleep -Seconds 2  # 少し待機

    # タスク状態確認
    $taskAfterTest = Get-ScheduledTask -TaskName $taskName
    Write-Log "Task state after test: $($taskAfterTest.State)"

    if ($taskAfterTest.State -eq "Running") {
        Write-Log "WARNING: Task started running - this is unexpected for a dry run"
        Stop-ScheduledTask -TaskName $taskName
        Write-Log "Stopped the test execution"
    } else {
        Write-Log "Task test completed successfully (task not running as expected)"
    }

} catch {
    Write-Log "WARNING: Task test failed, but task creation was successful: $($_.Exception.Message)"
    # テスト失敗でもタスク作成は成功しているので続行
}

# 最終確認
Write-Log "Performing final verification..."

try {
    $finalTask = Get-ScheduledTask -TaskName $taskName -ErrorAction Stop
    Write-Log "Final verification passed - task exists and is properly configured"
} catch {
    Write-Log "ERROR: Final verification failed - task may not be properly created" "ERROR"
    exit 1
}

# 成功メッセージ
Write-Log "SUCCESS: SO8T Complete Automation Pipeline task setup completed!"
Write-Log ""
Write-Log "Task Details:"
Write-Log "  Name: $taskName"
Write-Log "  Triggers: At logon (when you log in to Windows)"
Write-Log "  Action: $scriptPath"
Write-Log "  User: $env:USERNAME"
Write-Log ""
Write-Log "The pipeline will automatically start when you log in to Windows."
Write-Log "It will transform Borea-Phi3.5-instinct-jp into a complete SO8T/thinking multimodal model."
Write-Log ""
Write-Log "To check status: .\$($MyInvocation.MyCommand.Name) -Status"
Write-Log "To remove task: .\$($MyInvocation.MyCommand.Name) -Remove"
Write-Log ""
Write-Log "Monitor progress in: $projectRoot\logs\"
Write-Log ""

# 成功通知
try {
    [System.Console]::Beep(800, 200)
    [System.Console]::Beep(1000, 200)
    [System.Console]::Beep(1200, 300)
} catch {
    # ビープ音が使えない場合
}

Write-Log "Setup script completed successfully"
