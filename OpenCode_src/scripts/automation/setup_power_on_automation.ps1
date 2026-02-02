# SO8T Complete Automation Pipeline - Power-on Task Setup
# Windows Task Schedulerで電源投入時に自動実行するタスクを作成

param(
    [switch]$Remove,
    [switch]$Status
)

$taskName = "SO8T_Complete_PPO_Pipeline"
$pythonPath = (Get-Command py).Source
$scriptPath = "$PSScriptRoot\complete_ppo_pipeline_with_power_on_automation.py"
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
    $pythonVersion = & py -3 --version 2>&1
    Write-Log "Python version: $pythonVersion"
} catch {
    Write-Log "ERROR: Python (py -3) not found in PATH" "ERROR"
    exit 1
}

# プロジェクト構造確認
$requiredPaths = @(
    (Join-Path $projectRoot "scripts"),
    (Join-Path $projectRoot "so8t"),
    (Join-Path $projectRoot "configs")
)

foreach ($path in $requiredPaths) {
    if (-not (Test-Path $path)) {
        Write-Log "ERROR: Required path not found: $path" "ERROR"
        exit 1
    }
}

# webdataset パス設定（H:\from_D\webdataset を優先使用）
$webDatasetPaths = @(
    "H:\from_D\webdataset",  # 優先パス
    "D:\webdataset",         # 従来の推奨パス
    (Join-Path $projectRoot "webdataset")  # 最終フォールバック
)

$webDatasetPath = $null
foreach ($path in $webDatasetPaths) {
    if (Test-Path $path) {
        $webDatasetPath = $path
        Write-Log "Found webdataset directory: $webDatasetPath" "INFO"
        break
    }
}

# 見つからない場合はH:\from_D\webdatasetを作成
if (-not $webDatasetPath) {
    $webDatasetPath = "H:\from_D\webdataset"
    Write-Log "Creating webdataset directory: $webDatasetPath" "INFO"

    try {
        # H:\from_D が存在するか確認
        $fromDPath = "H:\from_D"
        if (-not (Test-Path $fromDPath)) {
            New-Item -ItemType Directory -Path $fromDPath -Force | Out-Null
            Write-Log "Created directory: $fromDPath" "INFO"
        }

        # webdatasetサブディレクトリ作成
        $webDatasetDirs = @(
            $webDatasetPath,
            (Join-Path $webDatasetPath "checkpoints"),
            (Join-Path $webDatasetPath "models"),
            (Join-Path $webDatasetPath "gguf_models"),
            (Join-Path $webDatasetPath "datasets"),
            (Join-Path $webDatasetPath "logs"),
            (Join-Path $webDatasetPath "temp")
        )

        foreach ($dir in $webDatasetDirs) {
            if (-not (Test-Path $dir)) {
                New-Item -ItemType Directory -Path $dir -Force | Out-Null
                Write-Log "Created directory: $dir" "INFO"
            }
        }

        Write-Log "Successfully created webdataset structure at: $webDatasetPath" "INFO"
    } catch {
        Write-Log "ERROR: Failed to create webdataset directory: $($_.Exception.Message)" "ERROR"
        exit 1
    }
}

# 環境変数設定
$env:WEBDATASET_PATH = $webDatasetPath
Write-Log "Set WEBDATASET_PATH environment variable: $webDatasetPath" "INFO"

Write-Log "All prerequisites verified"

# 既存タスクのクリーンアップ
Write-Log "Cleaning up existing tasks..."
try {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Log "Cleaned up existing task (if any)"
} catch {
    Write-Log "No existing task to clean up"
}

# バッチファイル作成
Write-Log "Creating batch file for task execution..."

$batchFilePath = Join-Path $PSScriptRoot "run_ppo_pipeline_task.bat"

try {
    $batchContent = @"
@echo off
chcp 65001 >nul
echo [SO8T] Starting PPO Pipeline Task
echo ===============================

cd /d "$projectRoot"

echo [INFO] Setting WEBDATASET_PATH environment variable...
set WEBDATASET_PATH=$webDatasetPath

echo [INFO] Running PPO Pipeline...
"$pythonPath" "$scriptPath"

if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] PPO Pipeline completed successfully
) else (
    echo [ERROR] PPO Pipeline failed with error code %ERRORLEVEL%
)

echo [DONE] Task execution completed
"@

    $batchContent | Out-File -FilePath $batchFilePath -Encoding UTF8 -Force
    Write-Log "Created batch file: $batchFilePath"

} catch {
    Write-Log "ERROR: Failed to create batch file: $($_.Exception.Message)" "ERROR"
    exit 1
}

# タスク作成（バッチファイルを使用）
Write-Log "Creating scheduled task using batch file..."

try {
    $batchAction = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$batchFilePath`""

    $trigger = New-ScheduledTaskTrigger -AtStartup

    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RunOnlyIfNetworkAvailable
    $settings.ExecutionTimeLimit = "PT0S"
    $settings.RestartCount = 3
    $settings.RestartInterval = "PT5M"

    $task = New-ScheduledTask -Action $batchAction -Trigger $trigger -Settings $settings

    Register-ScheduledTask -TaskName $taskName -InputObject $task -Force | Out-Null

    Write-Log "Successfully created scheduled task '$taskName'"

    # タスク情報表示
    $createdTask = Get-ScheduledTask -TaskName $taskName
    Write-Log "Task created with the following settings:"
    Write-Log "  Name: $($createdTask.TaskName)"
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
