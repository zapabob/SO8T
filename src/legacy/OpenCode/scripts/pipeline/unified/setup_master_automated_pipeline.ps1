# SO8T完全自動化マスターパイプライン セットアップスクリプト (PowerShell版)
# 管理者権限で自動的に再起動する機能付き

# UTF-8エンコーディング設定
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# 管理者権限チェック
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "SO8T Master Automated Pipeline Setup" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "[INFO] This script requires administrator privileges." -ForegroundColor Yellow
    Write-Host "[INFO] Restarting with administrator privileges..." -ForegroundColor Yellow
    Write-Host ""
    
    # 管理者権限で再起動
    $scriptPath = $MyInvocation.MyCommand.Path
    $arguments = "-ExecutionPolicy Bypass -File `"$scriptPath`""
    
    Start-Process powershell.exe -Verb RunAs -ArgumentList $arguments -Wait
    
    exit $LASTEXITCODE
}

# プロジェクトルートパス
$PROJECT_ROOT = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SO8T Master Automated Pipeline Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "[OK] Running with administrator privileges" -ForegroundColor Green
Write-Host "Project Root: $PROJECT_ROOT" -ForegroundColor Yellow
Write-Host ""

# Python実行ファイルの検出
$pythonCmd = $null
$pythonCommands = @("py", "python", "python3")

foreach ($cmd in $pythonCommands) {
    $found = Get-Command $cmd -ErrorAction SilentlyContinue
    if ($found) {
        $pythonCmd = $cmd
        break
    }
}

if (-not $pythonCmd) {
    Write-Host "[ERROR] Python not found in PATH" -ForegroundColor Red
    Write-Host "[ERROR] Please install Python or add it to PATH" -ForegroundColor Red
    Write-Host ""
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

Write-Host "[INFO] Using Python: $pythonCmd" -ForegroundColor Green
Write-Host ""

# セットアップスクリプトの実行
Write-Host "[INFO] Running setup script..." -ForegroundColor Yellow
Write-Host ""

$setupScript = Join-Path $PROJECT_ROOT "scripts\pipelines\setup_master_automated_pipeline.py"

if (-not (Test-Path $setupScript)) {
    Write-Host "[ERROR] Setup script not found: $setupScript" -ForegroundColor Red
    Write-Host ""
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# Pythonスクリプトを実行
$process = Start-Process -FilePath $pythonCmd -ArgumentList "-3", "`"$setupScript`"" -Wait -NoNewWindow -PassThru

if ($process.ExitCode -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Setup failed (Exit code: $($process.ExitCode))" -ForegroundColor Red
    Write-Host ""
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "[SUCCESS] Setup completed successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "The pipeline will automatically run on system startup." -ForegroundColor Green
Write-Host "To test manually, run:" -ForegroundColor Yellow
Write-Host "  $pythonCmd -3 `"$PROJECT_ROOT\scripts\pipelines\master_automated_pipeline.py`" --run" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")






