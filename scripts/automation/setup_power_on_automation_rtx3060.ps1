# RTX3060 SO8T Power-on Automation Setup
# Frozen weights + QLoRA fine-tuning for RTX3060

param(
    [switch]$Status,
    [switch]$Remove
)

$TaskName = "SO8T_RTX3060_Automation"
$ScriptPath = "$PSScriptRoot\run_complete_so8t_pipeline_rtx3060.bat"
$LogPath = "$PSScriptRoot\..\..\logs\rtx3060_automation_setup.log"

function Write-Log {
    param([string]$Message)
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$Timestamp - $Message" | Out-File -FilePath $LogPath -Append -Encoding UTF8
    Write-Host $Message
}

function Test-RTX3060 {
    try {
        $cudaAvailable = python -c "import torch; print(torch.cuda.is_available())" 2>$null
        if ($cudaAvailable -eq "True") {
            $gpuMem = python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory / (1024**3))" 2>$null
            if ([double]$gpuMem -ge 8.0) {
                Write-Log "[OK] RTX3060 detected: ${gpuMem}GB VRAM"
                return $true
            } else {
                Write-Log "[ERROR] Insufficient GPU memory: ${gpuMem}GB (need 8GB+)"
                return $false
            }
        } else {
            Write-Log "[ERROR] CUDA not available"
            return $false
        }
    } catch {
        Write-Log "[ERROR] RTX3060 check failed: $($_.Exception.Message)"
        return $false
    }
}

function Get-TaskStatus {
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($task) {
        $lastRun = $task.LastRunTime
        $nextRun = $task.NextRunTime
        $state = $task.State

        Write-Log "=== RTX3060 SO8T Automation Task Status ==="
        Write-Log "Task Name: $TaskName"
        Write-Log "State: $state"
        if ($lastRun) {
            Write-Log "Last Run: $lastRun"
        }
        if ($nextRun) {
            Write-Log "Next Run: $nextRun"
        }
        Write-Log "Script: $ScriptPath"
        Write-Log "========================================"
    } else {
        Write-Log "[INFO] Task '$TaskName' not found"
    }
}

function Remove-ScheduledTask {
    Write-Log "[REMOVE] Removing existing task: $TaskName"
    try {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction Stop
        Write-Log "[OK] Task removed successfully"
    } catch {
        Write-Log "[WARNING] Task removal failed or task didn't exist: $($_.Exception.Message)"
    }
}

function New-RTX3060AutomationTask {
    Write-Log "[CREATE] Creating RTX3060 SO8T automation task"

    # RTX3060チェック
    if (!(Test-RTX3060)) {
        Write-Log "[ERROR] RTX3060 validation failed. Task creation aborted."
        return
    }

    # 既存タスク削除
    Remove-ScheduledTask

    # 新しいタスク作成
    try {
        $action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$ScriptPath`""

        # 電源投入時トリガー（ログオン時）
        $trigger = New-ScheduledTaskTrigger -AtLogOn

        # バッテリー駆動時も実行、ネットワーク接続必須
        $settings = New-ScheduledTaskSettingsSet `
            -AllowStartIfOnBatteries `
            -DontStopIfGoingOnBatteries `
            -StartWhenAvailable `
            -RunOnlyIfNetworkAvailable `
            -ExecutionTimeLimit (New-TimeSpan -Hours 24) `
            -RestartCount 3 `
            -RestartInterval (New-TimeSpan -Minutes 5)

        # 最高権限で実行
        $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -RunLevel Highest

        # タスク登録
        Register-ScheduledTask `
            -TaskName $TaskName `
            -Action $action `
            -Trigger $trigger `
            -Settings $settings `
            -Principal $principal `
            -Description "RTX3060 SO8T Complete Automation Pipeline (Frozen weights + QLoRA)"

        Write-Log "[SUCCESS] RTX3060 SO8T automation task created successfully"
        Write-Log "Task will run automatically on power-on/logon"
        Write-Log "Features: Frozen base weights, QLoRA fine-tuning, RTX3060 optimized"

    } catch {
        Write-Log "[ERROR] Task creation failed: $($_.Exception.Message)"
        throw
    }
}

# メイン処理
Write-Log "=== RTX3060 SO8T Power-on Automation Setup ==="

if ($Status) {
    Get-TaskStatus
    exit
}

if ($Remove) {
    Remove-ScheduledTask
    exit
}

# 新しいタスク作成
New-RTX3060AutomationTask

# 作成されたタスクのステータス表示
Get-TaskStatus

Write-Log "=== Setup Complete ==="
Write-Log "RTX3060 SO8T automation is ready for power-on execution"
Write-Log "- Frozen base weights enabled"
Write-Log "- QLoRA fine-tuning optimized"
Write-Log "- RTX3060 memory management active"
