# webdatasetシンボリックリンク設定スクリプト
# 
# D:\webdatasetをリポジトリ内のwebdatasetディレクトリにシンボリックリンクで接続
# 
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\utils\setup_webdataset_symlink.ps1

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "webdataset Symlink Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# プロジェクトルート
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $projectRoot

$symlinkPath = Join-Path $projectRoot "webdataset"
$targetPath = "D:\webdataset"

Write-Host "Symlink path: $symlinkPath" -ForegroundColor Yellow
Write-Host "Target path: $targetPath" -ForegroundColor Yellow
Write-Host ""

# 管理者権限チェック
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "[WARNING] Administrator privileges required to create symlinks" -ForegroundColor Yellow
    Write-Host "[INFO] Attempting to create symlink anyway..." -ForegroundColor Yellow
}

# 既存のwebdatasetディレクトリをチェック
if (Test-Path $symlinkPath) {
    $item = Get-Item $symlinkPath -ErrorAction SilentlyContinue
    
    if ($item.LinkType -eq "SymbolicLink" -or $item.LinkType -eq "Junction") {
        $currentTarget = $item.Target
        if ($currentTarget -eq $targetPath) {
            Write-Host "[OK] Symlink already exists and points to correct location" -ForegroundColor Green
            Write-Host "     Current target: $currentTarget" -ForegroundColor Gray
            exit 0
        } else {
            Write-Host "[WARNING] Symlink exists but points to different location" -ForegroundColor Yellow
            Write-Host "     Current target: $currentTarget" -ForegroundColor Gray
            Write-Host "     Expected target: $targetPath" -ForegroundColor Gray
            $response = Read-Host "Remove existing symlink and create new one? (y/n)"
            if ($response -eq "y" -or $response -eq "Y") {
                Remove-Item $symlinkPath -Force
            } else {
                Write-Host "[CANCELLED] Exiting without changes" -ForegroundColor Yellow
                exit 1
            }
        }
    } else {
        Write-Host "[WARNING] webdataset exists but is not a symlink" -ForegroundColor Yellow
        Write-Host "     Type: $($item.GetType().Name)" -ForegroundColor Gray
        $response = Read-Host "Remove existing directory and create symlink? (y/n)"
        if ($response -eq "y" -or $response -eq "Y") {
            Remove-Item $symlinkPath -Recurse -Force
        } else {
            Write-Host "[CANCELLED] Exiting without changes" -ForegroundColor Yellow
            exit 1
        }
    }
}

# D:\webdatasetが存在するか確認
if (-not (Test-Path $targetPath)) {
    Write-Host "[INFO] Creating target directory: $targetPath" -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $targetPath -Force | Out-Null
    Write-Host "[OK] Target directory created" -ForegroundColor Green
}

# シンボリックリンク作成
Write-Host "[INFO] Creating symlink..." -ForegroundColor Yellow

try {
    # 方法1: New-Item (PowerShell 5.1+)
    New-Item -ItemType SymbolicLink -Path $symlinkPath -Target $targetPath -Force | Out-Null
    Write-Host "[OK] Symlink created successfully using New-Item" -ForegroundColor Green
} catch {
    Write-Host "[WARNING] New-Item failed: $($_.Exception.Message)" -ForegroundColor Yellow
    Write-Host "[INFO] Trying mklink command..." -ForegroundColor Yellow
    
    try {
        # 方法2: mklink (CMD)
        $result = cmd /c mklink /D "$symlinkPath" "$targetPath" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] Symlink created successfully using mklink" -ForegroundColor Green
        } else {
            Write-Host "[ERROR] mklink failed: $result" -ForegroundColor Red
            throw "Failed to create symlink"
        }
    } catch {
        Write-Host "[ERROR] All symlink creation methods failed" -ForegroundColor Red
        Write-Host "[INFO] You may need to run this script as Administrator" -ForegroundColor Yellow
        Write-Host "[INFO] Or manually create the symlink:" -ForegroundColor Yellow
        Write-Host "       cmd /c mklink /D `"$symlinkPath`" `"$targetPath`"" -ForegroundColor White
        exit 1
    }
}

# 確認
$createdItem = Get-Item $symlinkPath
Write-Host ""
Write-Host "[VERIFY] Symlink created:" -ForegroundColor Cyan
Write-Host "  Path: $($createdItem.FullName)" -ForegroundColor White
Write-Host "  Type: $($createdItem.LinkType)" -ForegroundColor White
Write-Host "  Target: $($createdItem.Target)" -ForegroundColor White

# テスト: ファイル作成テスト
$testFile = Join-Path $symlinkPath ".symlink_test"
try {
    "test" | Out-File -FilePath $testFile -Encoding UTF8
    $testFileInTarget = Join-Path $targetPath ".symlink_test"
    if (Test-Path $testFileInTarget) {
        Write-Host "[OK] Symlink is working correctly" -ForegroundColor Green
        Remove-Item $testFile -Force
        Remove-Item $testFileInTarget -Force -ErrorAction SilentlyContinue
    } else {
        Write-Host "[WARNING] Symlink may not be working correctly" -ForegroundColor Yellow
    }
} catch {
    Write-Host "[WARNING] Could not verify symlink: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[SUCCESS] webdataset symlink setup completed" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

