# SO8T統制による完全自動バックグラウンドスクレイピングをWindowsサービスとしてインストール

$ErrorActionPreference = "Continue"

$PROJECT_ROOT = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$SCRIPT_PATH = Join-Path $PROJECT_ROOT "scripts\data\so8t_auto_background_scraping.py"
$OUTPUT_DIR = "D:\webdataset\processed"
$LOG_DIR = Join-Path $PROJECT_ROOT "logs"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "SO8T Auto Background Scraping Service Installer" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Project Root: $PROJECT_ROOT" -ForegroundColor Yellow
Write-Host "Script Path: $SCRIPT_PATH" -ForegroundColor Yellow
Write-Host ""

# Pythonパスの確認
$PYTHON = $null
if (Test-Path (Join-Path $PROJECT_ROOT "venv\Scripts\python.exe")) {
    $PYTHON = Join-Path $PROJECT_ROOT "venv\Scripts\python.exe"
} else {
    $PYTHON = (Get-Command py -ErrorAction SilentlyContinue).Source
    if (-not $PYTHON) {
        Write-Host "[ERROR] Python not found" -ForegroundColor Red
        exit 1
    }
}

Write-Host "[INFO] Python: $PYTHON" -ForegroundColor Green

# バッチファイルを作成（サービス実行用）
$BATCH_FILE = Join-Path $PROJECT_ROOT "scripts\data\so8t_auto_scraping_service.bat"
$BATCH_CONTENT = @"
@echo off
chcp 65001 >nul
cd /d "$PROJECT_ROOT"
"$PYTHON" -3 scripts\data\so8t_auto_background_scraping.py --output "$OUTPUT_DIR" --daemon --auto-restart --max-restarts 10 --restart-delay 60.0
"@

$BATCH_CONTENT | Out-File -FilePath $BATCH_FILE -Encoding UTF8
Write-Host "[OK] Service batch file created: $BATCH_FILE" -ForegroundColor Green

# タスクスケジューラーに登録
Write-Host "[INFO] Registering with Task Scheduler..." -ForegroundColor Yellow

$TASK_NAME = "SO8T_Auto_Background_Scraping"
$TASK_DESCRIPTION = "SO8T統制による完全自動バックグラウンドDeepResearch Webスクレイピング"

# 既存のタスクを削除（存在する場合）
$existingTask = Get-ScheduledTask -TaskName $TASK_NAME -ErrorAction SilentlyContinue
if ($existingTask) {
    Unregister-ScheduledTask -TaskName $TASK_NAME -Confirm:$false
    Write-Host "[INFO] Existing task removed" -ForegroundColor Yellow
}

# タスクアクションを作成
$action = New-ScheduledTaskAction -Execute $BATCH_FILE -WorkingDirectory $PROJECT_ROOT

# タスクトリガーを作成（システム起動時）
$trigger = New-ScheduledTaskTrigger -AtStartup

# タスク設定
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RunOnlyIfNetworkAvailable

# タスクを登録
Register-ScheduledTask -TaskName $TASK_NAME -Description $TASK_DESCRIPTION -Action $action -Trigger $trigger -Settings $settings -RunLevel Highest

Write-Host "[OK] Task registered: $TASK_NAME" -ForegroundColor Green
Write-Host "[INFO] Task will start automatically on system startup" -ForegroundColor Cyan
Write-Host "[INFO] To start manually: Start-ScheduledTask -TaskName `"$TASK_NAME`"" -ForegroundColor Cyan
Write-Host "[INFO] To stop: Stop-ScheduledTask -TaskName `"$TASK_NAME`"" -ForegroundColor Cyan
Write-Host "[INFO] To remove: Unregister-ScheduledTask -TaskName `"$TASK_NAME`" -Confirm:`$false" -ForegroundColor Cyan

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "[SUCCESS] Service installation completed" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan

# 音声通知
& (Join-Path $PROJECT_ROOT "scripts\utils\play_audio_notification.ps1")





