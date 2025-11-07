# webdatasetシンボリックリンク設定スクリプト（自動版）
# 
# D:\webdatasetをリポジトリ内のwebdatasetディレクトリにシンボリックリンクで接続
# 既存のwebdatasetディレクトリが空の場合は自動的に削除してシンボリックリンクを作成
# 
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\utils\setup_webdataset_symlink_auto.ps1

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "webdataset Symlink Setup (Auto)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# プロジェクトルート
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $projectRoot

$symlinkPath = Join-Path $projectRoot "webdataset"
$targetPath = "D:\webdataset"

Write-Host "Symlink path: $symlinkPath" -ForegroundColor Yellow
Write-Host "Target path: $targetPath" -ForegroundColor Yellow
Write-Host ""

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
            Write-Host "[INFO] Removing existing symlink (points to different location)" -ForegroundColor Yellow
            Remove-Item $symlinkPath -Force
        }
    } else {
        # 通常のディレクトリの場合、内容を確認
        $items = Get-ChildItem $symlinkPath -ErrorAction SilentlyContinue
        $itemCount = ($items | Measure-Object).Count
        
        if ($itemCount -eq 0) {
            Write-Host "[INFO] Removing empty webdataset directory" -ForegroundColor Yellow
            Remove-Item $symlinkPath -Force
        } else {
            Write-Host "[WARNING] webdataset directory contains $itemCount items" -ForegroundColor Yellow
            Write-Host "[INFO] Moving contents to D:\webdataset..." -ForegroundColor Yellow
            
            # 内容をD:\webdatasetに移動
            if (-not (Test-Path $targetPath)) {
                New-Item -ItemType Directory -Path $targetPath -Force | Out-Null
            }
            
            Move-Item -Path "$symlinkPath\*" -Destination $targetPath -Force -ErrorAction SilentlyContinue
            Remove-Item $symlinkPath -Force
            Write-Host "[OK] Contents moved to D:\webdataset" -ForegroundColor Green
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
    $null = New-Item -ItemType SymbolicLink -Path $symlinkPath -Target $targetPath -Force
    Write-Host "[OK] Symlink created successfully using New-Item" -ForegroundColor Green
} catch {
    Write-Host "[WARNING] New-Item failed: $($_.Exception.Message)" -ForegroundColor Yellow
    Write-Host "[INFO] Trying mklink command..." -ForegroundColor Yellow
    
    try {
        # 方法2: mklink (CMD) - ジャンクションを使用（管理者権限不要の場合がある）
        $result = cmd /c mklink /J "$symlinkPath" "$targetPath" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] Junction created successfully using mklink" -ForegroundColor Green
        } else {
            # 方法3: シンボリックリンク（管理者権限必要）
            $result = cmd /c mklink /D "$symlinkPath" "$targetPath" 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Host "[OK] Symlink created successfully using mklink" -ForegroundColor Green
            } else {
                Write-Host "[ERROR] mklink failed: $result" -ForegroundColor Red
                throw "Failed to create symlink"
            }
        }
    } catch {
        Write-Host "[ERROR] All symlink creation methods failed" -ForegroundColor Red
        Write-Host "[INFO] You may need to run this script as Administrator" -ForegroundColor Yellow
        Write-Host "[INFO] Or manually create the symlink:" -ForegroundColor Yellow
        Write-Host "       cmd /c mklink /J `"$symlinkPath`" `"$targetPath`"" -ForegroundColor White
        Write-Host "       (Junction does not require admin privileges)" -ForegroundColor Gray
        exit 1
    }
}

# 確認
$createdItem = Get-Item $symlinkPath
Write-Host ""
Write-Host "[VERIFY] Link created:" -ForegroundColor Cyan
Write-Host "  Path: $($createdItem.FullName)" -ForegroundColor White
Write-Host "  Type: $($createdItem.LinkType)" -ForegroundColor White
if ($createdItem.LinkType) {
    Write-Host "  Target: $($createdItem.Target)" -ForegroundColor White
}

# テスト: ファイル作成テスト
$testFile = Join-Path $symlinkPath ".symlink_test"
try {
    "test" | Out-File -FilePath $testFile -Encoding UTF8
    $testFileInTarget = Join-Path $targetPath ".symlink_test"
    if (Test-Path $testFileInTarget) {
        Write-Host "[OK] Link is working correctly" -ForegroundColor Green
        Remove-Item $testFile -Force
        Remove-Item $testFileInTarget -Force -ErrorAction SilentlyContinue
    } else {
        Write-Host "[WARNING] Link may not be working correctly" -ForegroundColor Yellow
    }
} catch {
    Write-Host "[WARNING] Could not verify link: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[SUCCESS] webdataset symlink setup completed" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

