# ========================================
# SO8T Auto Resume Setup Script (PowerShell)
# Windowsタスクスケジューラーに自動実行タスクを登録
# ========================================

# UTF-8エンコーディング設定
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# 管理者権限チェック
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "[ERROR] This script requires administrator privileges." -ForegroundColor Red
    Write-Host "[ERROR] Please run as administrator." -ForegroundColor Red
    Write-Host ""
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# プロジェクトルートパス（スクリプトの場所から自動検出）
if ($PSScriptRoot) {
    $PROJECT_ROOT = Split-Path -Parent $PSScriptRoot
} else {
    # PSScriptRootが利用できない場合、現在のディレクトリから検索
    $currentDir = Get-Location
    $scriptPath = Join-Path $currentDir "scripts\setup_auto_resume.ps1"
    if (Test-Path $scriptPath) {
        $PROJECT_ROOT = $currentDir
    } else {
        # デフォルトのプロジェクトパス
        $PROJECT_ROOT = "C:\Users\downl\Desktop\SO8T"
        Write-Host "[WARNING] Using default project path: $PROJECT_ROOT" -ForegroundColor Yellow
        Write-Host "[INFO] If this is incorrect, please run from the project directory or specify the path." -ForegroundColor Yellow
    }
}

$STARTUP_SCRIPT = Join-Path $PROJECT_ROOT "scripts\auto_resume_startup.bat"

if (-not (Test-Path $STARTUP_SCRIPT)) {
    Write-Host "[ERROR] Startup script not found: $STARTUP_SCRIPT" -ForegroundColor Red
    Write-Host ""
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# タスク名
$TASK_NAME = "SO8T-AutoResume"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SO8T Auto Resume Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Project Root: $PROJECT_ROOT" -ForegroundColor Yellow
Write-Host "Startup Script: $STARTUP_SCRIPT" -ForegroundColor Yellow
Write-Host "Task Name: $TASK_NAME" -ForegroundColor Yellow
Write-Host ""

# 既存のタスクを削除（存在する場合）
$existingTask = schtasks /query /tn $TASK_NAME 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "[INFO] Removing existing task..." -ForegroundColor Yellow
    schtasks /delete /tn $TASK_NAME /f | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Existing task removed" -ForegroundColor Green
    } else {
        Write-Host "[WARNING] Failed to remove existing task" -ForegroundColor Yellow
    }
}

# 新しいタスクを作成
Write-Host "[INFO] Creating new task..." -ForegroundColor Yellow
Write-Host ""

# タスクスケジューラーコマンド
$TASK_COMMAND = "schtasks /create /tn `"$TASK_NAME`" /tr `"$STARTUP_SCRIPT`" /sc onlogon /rl highest /f"

# タスクを作成
Invoke-Expression $TASK_COMMAND

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to create task" -ForegroundColor Red
    Write-Host "[ERROR] Command: $TASK_COMMAND" -ForegroundColor Red
    Write-Host ""
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

Write-Host ""
Write-Host "[OK] Task created successfully" -ForegroundColor Green
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Task Configuration" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Task Name: $TASK_NAME" -ForegroundColor Yellow
Write-Host "Trigger: On user logon" -ForegroundColor Yellow
Write-Host "Script: $STARTUP_SCRIPT" -ForegroundColor Yellow
Write-Host ""
Write-Host "[INFO] The task will run automatically when you log in." -ForegroundColor Cyan
Write-Host "[INFO] To view the task: schtasks /query /tn `"$TASK_NAME`"" -ForegroundColor Cyan
Write-Host "[INFO] To delete the task: schtasks /delete /tn `"$TASK_NAME`" /f" -ForegroundColor Cyan
Write-Host ""

# タスクの詳細を表示
Write-Host "[INFO] Task details:" -ForegroundColor Yellow
schtasks /query /tn $TASK_NAME /fo list /v

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Completed" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 音声通知
$audioFile = Join-Path $PROJECT_ROOT ".cursor\marisa_owattaze.wav"
if (Test-Path $audioFile) {
    try {
        Add-Type -AssemblyName System.Windows.Forms
        $player = New-Object System.Media.SoundPlayer $audioFile
        $player.Play()
        Write-Host "[OK] Audio notification played successfully" -ForegroundColor Green
    } catch {
        Write-Host "[WARNING] Failed to play audio: $($_.Exception.Message)" -ForegroundColor Yellow
    }
} else {
    Write-Host "[WARNING] Audio file not found: $audioFile" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

