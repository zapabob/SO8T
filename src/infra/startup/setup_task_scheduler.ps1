# SO(8)T Task Scheduler Setup Script (PowerShell)
# タスクスケジューラに電源投入時自動起動を設定

param(
    [switch]$Uninstall,
    [switch]$Test
)

# 管理者権限チェック
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# メイン処理
function Main {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "SO(8)T Task Scheduler Setup (PowerShell)" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    # 管理者権限チェック
    if (-not (Test-Administrator)) {
        Write-Host "[ERROR] Administrator privileges required!" -ForegroundColor Red
        Write-Host "Please run this script as Administrator." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }

    # 作業ディレクトリ設定
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $projectRoot = Split-Path -Parent (Split-Path -Parent $scriptDir)
    $startupScript = Join-Path $projectRoot "scripts\startup\so8t_power_on_startup.bat"

    Write-Host "[INFO] Project root: $projectRoot" -ForegroundColor Gray
    Write-Host "[INFO] Startup script: $startupScript" -ForegroundColor Gray

    # テストモード
    if ($Test) {
        Write-Host "[TEST] Testing current configuration..." -ForegroundColor Yellow
        Test-Configuration -StartupScript $startupScript
        return
    }

    # アンインストールモード
    if ($Uninstall) {
        Write-Host "[UNINSTALL] Removing SO8T task from Task Scheduler..." -ForegroundColor Yellow
        Remove-SO8TTask
        return
    }

    # インストールモード
    Write-Host "[INSTALL] Setting up SO8T power-on auto startup..." -ForegroundColor Green

    # スクリプト存在確認
    if (-not (Test-Path $startupScript)) {
        Write-Host "[ERROR] Startup script not found: $startupScript" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }

    # 既存タスクの削除
    Write-Host "[INFO] Removing existing SO8T task if present..." -ForegroundColor Gray
    try {
        schtasks /delete /tn "SO8T_Power_On_Startup" /f 2>$null
    } catch {
        # タスクが存在しない場合は無視
    }

    # 新規タスク作成
    Write-Host "[INFO] Creating new SO8T power-on startup task..." -ForegroundColor Gray

    $taskCommand = "schtasks /create /tn `"SO8T_Power_On_Startup`" /tr `"`"$startupScript`"`" /sc ONLOGON /rl HIGHEST /delay 0000:30 /f"

    Write-Host "[DEBUG] Executing: $taskCommand" -ForegroundColor DarkGray

    try {
        $result = Invoke-Expression $taskCommand
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[SUCCESS] Task created successfully!" -ForegroundColor Green
            Write-Host "Task Name: SO8T_Power_On_Startup" -ForegroundColor White
            Write-Host "Trigger: At logon (power-on)" -ForegroundColor White
            Write-Host "Delay: 30 seconds" -ForegroundColor White
            Write-Host "Run Level: Highest privileges" -ForegroundColor White
        } else {
            throw "Task creation failed with exit code: $LASTEXITCODE"
        }
    } catch {
        Write-Host "[ERROR] Failed to create task!" -ForegroundColor Red
        Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host ""
        Write-Host "Troubleshooting:" -ForegroundColor Yellow
        Write-Host "1. Make sure you're running as Administrator" -ForegroundColor Yellow
        Write-Host "2. Check if Task Scheduler service is running" -ForegroundColor Yellow
        Write-Host "3. Try running the .bat version instead" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }

    # タスク確認
    Write-Host ""
    Write-Host "[INFO] Verifying task creation..." -ForegroundColor Gray
    try {
        $queryResult = schtasks /query /tn "SO8T_Power_On_Startup" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[SUCCESS] Task verification passed!" -ForegroundColor Green
        } else {
            Write-Host "[WARNING] Task verification failed!" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "[WARNING] Could not verify task status" -ForegroundColor Yellow
    }

    # 完了メッセージ
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Task Scheduler Setup Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "The SO8T pipeline will now start automatically when you log on to Windows." -ForegroundColor White
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Restart your computer to test the automatic startup" -ForegroundColor Yellow
    Write-Host "2. Check logs\startup\ for startup logs" -ForegroundColor Yellow
    Write-Host "3. Use 'scripts\startup\test_startup.bat' to verify setup" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to continue"
}

# タスク削除関数
function Remove-SO8TTask {
    try {
        $result = schtasks /delete /tn "SO8T_Power_On_Startup" /f 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[SUCCESS] Task deleted successfully!" -ForegroundColor Green
            Write-Host "Task 'SO8T_Power_On_Startup' has been removed." -ForegroundColor White
        } else {
            Write-Host "[WARNING] Task may not exist or deletion failed." -ForegroundColor Yellow
            Write-Host "This is normal if the task was already removed." -ForegroundColor Gray
        }
    } catch {
        Write-Host "[ERROR] Failed to delete task: $($_.Exception.Message)" -ForegroundColor Red
    }

    # ログファイル削除オプション
    Write-Host ""
    $deleteLogs = Read-Host "Delete startup log files? (y/N)"
    if ($deleteLogs -eq 'y' -or $deleteLogs -eq 'Y') {
        $logDir = Join-Path (Split-Path -Parent (Split-Path -Parent $scriptDir)) "logs\startup"
        if (Test-Path $logDir) {
            try {
                Remove-Item $logDir -Recurse -Force
                Write-Host "[INFO] Startup log files deleted." -ForegroundColor Green
            } catch {
                Write-Host "[WARNING] Could not delete log files: $($_.Exception.Message)" -ForegroundColor Yellow
            }
        } else {
            Write-Host "[INFO] No startup log files found." -ForegroundColor Gray
        }
    } else {
        Write-Host "[INFO] Startup log files preserved." -ForegroundColor Gray
    }

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Task Scheduler Removal Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
}

# 設定テスト関数
function Test-Configuration {
    param([string]$StartupScript)

    Write-Host "[TEST 1] Checking if SO8T task exists..." -ForegroundColor Yellow
    try {
        $queryResult = schtasks /query /tn "SO8T_Power_On_Startup" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[PASS] SO8T task found in Task Scheduler" -ForegroundColor Green
            $status = $queryResult | Select-String "Ready|Running|Disabled"
            if ($status) {
                Write-Host "[INFO] Task status: $($status.Line.Trim())" -ForegroundColor White
            }
        } else {
            Write-Host "[FAIL] SO8T task not found in Task Scheduler" -ForegroundColor Red
        }
    } catch {
        Write-Host "[ERROR] Could not check task status: $($_.Exception.Message)" -ForegroundColor Red
    }

    Write-Host ""
    Write-Host "[TEST 2] Checking startup script..." -ForegroundColor Yellow
    if (Test-Path $startupScript) {
        Write-Host "[PASS] Startup script exists" -ForegroundColor Green
        Write-Host "[INFO] Path: $startupScript" -ForegroundColor White
    } else {
        Write-Host "[FAIL] Startup script not found: $startupScript" -ForegroundColor Red
    }

    Write-Host ""
    Write-Host "[TEST 3] Checking Python environment..." -ForegroundColor Yellow
    try {
        $pythonVersion = python --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[PASS] Python is available" -ForegroundColor Green
            Write-Host "[INFO] $pythonVersion" -ForegroundColor White
        } else {
            Write-Host "[FAIL] Python not found or not in PATH" -ForegroundColor Red
        }
    } catch {
        Write-Host "[FAIL] Python check failed: $($_.Exception.Message)" -ForegroundColor Red
    }

    Write-Host ""
    Write-Host "[TEST 4] Checking GPU availability..." -ForegroundColor Yellow
    try {
        $gpuInfo = nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[PASS] GPU detected" -ForegroundColor Green
            Write-Host "[INFO] GPU: $gpuInfo" -ForegroundColor White
        } else {
            Write-Host "[WARNING] GPU not detected or nvidia-smi not available" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "[WARNING] GPU check failed: $($_.Exception.Message)" -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "[TEST 5] Checking log directories..." -ForegroundColor Yellow
    $projectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $StartupScript))
    $logDir = Join-Path $projectRoot "logs"
    $startupLogDir = Join-Path $logDir "startup"

    if (Test-Path $logDir) {
        Write-Host "[PASS] Logs directory exists" -ForegroundColor Green
        if (Test-Path $startupLogDir) {
            Write-Host "[PASS] Startup logs directory exists" -ForegroundColor Green
            $logFiles = Get-ChildItem $startupLogDir -File 2>$null
            if ($logFiles) {
                Write-Host "[INFO] Existing startup logs:" -ForegroundColor White
                $logFiles | ForEach-Object { Write-Host "  - $($_.Name)" -ForegroundColor White }
            } else {
                Write-Host "[INFO] No startup logs found yet" -ForegroundColor Gray
            }
        } else {
            Write-Host "[FAIL] Startup logs directory missing" -ForegroundColor Red
        }
    } else {
        Write-Host "[FAIL] Logs directory missing" -ForegroundColor Red
    }

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Configuration Test Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
}

# スクリプト実行
Main

