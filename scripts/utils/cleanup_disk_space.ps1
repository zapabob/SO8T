# Disk Space Cleanup Script for SO8T Project
# Dドライブの容量を確保するためのクリーンアップスクリプト

param(
    [switch]$DryRun = $false,
    [int]$LogDays = 7,
    [switch]$CleanCache = $true,
    [switch]$CleanLogs = $true,
    [switch]$CleanCheckpoints = $false,
    [switch]$CleanMLRuns = $true,
    [switch]$CleanWandb = $true,
    [switch]$CleanRuns = $true
)

$ErrorActionPreference = "Continue"

# プロジェクトルート
$PROJECT_ROOT = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $PROJECT_ROOT

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "SO8T Disk Space Cleanup Script" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Project Root: $PROJECT_ROOT" -ForegroundColor Yellow
Write-Host "Dry Run: $DryRun" -ForegroundColor Yellow
Write-Host ""

# ディスク容量確認
Write-Host "[STEP 1] Checking disk space..." -ForegroundColor Green
try {
    $cDrive = Get-PSDrive C
    $dDrive = Get-PSDrive D -ErrorAction SilentlyContinue
    
    if ($cDrive) {
        $cFreeGB = [math]::Round($cDrive.Free / 1GB, 2)
        $cUsedGB = [math]::Round($cDrive.Used / 1GB, 2)
        Write-Host "  C Drive - Free: $cFreeGB GB, Used: $cUsedGB GB" -ForegroundColor Cyan
    }
    
    if ($dDrive) {
        $dFreeGB = [math]::Round($dDrive.Free / 1GB, 2)
        $dUsedGB = [math]::Round($dDrive.Used / 1GB, 2)
        Write-Host "  D Drive - Free: $dFreeGB GB, Used: $dUsedGB GB" -ForegroundColor Cyan
    } else {
        Write-Host "  D Drive not found" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  [ERROR] Failed to check disk space: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""

# 削除対象のサイズを計算
$totalSize = 0
$itemsToDelete = @()

# 1. __pycache__ ディレクトリ
if ($CleanCache) {
    Write-Host "[STEP 2] Scanning __pycache__ directories..." -ForegroundColor Green
    $pycacheDirs = Get-ChildItem -Path $PROJECT_ROOT -Directory -Recurse -Filter "__pycache__" -ErrorAction SilentlyContinue
    $pycacheSize = 0
    foreach ($dir in $pycacheDirs) {
        $size = (Get-ChildItem -Path $dir.FullName -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        $pycacheSize += $size
        $itemsToDelete += @{
            Path = $dir.FullName
            Size = $size
            Type = "__pycache__"
        }
    }
    $pycacheSizeGB = [math]::Round($pycacheSize / 1GB, 2)
    Write-Host "  Found $($pycacheDirs.Count) __pycache__ directories ($pycacheSizeGB GB)" -ForegroundColor Cyan
    $totalSize += $pycacheSize
}

# 2. ログファイル
if ($CleanLogs) {
    Write-Host "[STEP 3] Scanning log files..." -ForegroundColor Green
    $logFiles = Get-ChildItem -Path "$PROJECT_ROOT\logs" -File -Filter "*.log" -ErrorAction SilentlyContinue | 
        Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-$LogDays) }
    $logSize = ($logFiles | Measure-Object -Property Length -Sum).Sum
    $logSizeGB = [math]::Round($logSize / 1GB, 2)
    Write-Host "  Found $($logFiles.Count) log files older than $LogDays days ($logSizeGB GB)" -ForegroundColor Cyan
    foreach ($file in $logFiles) {
        $itemsToDelete += @{
            Path = $file.FullName
            Size = $file.Length
            Type = "Log File"
        }
    }
    $totalSize += $logSize
    
    # ルートディレクトリのログファイル
    $rootLogFiles = Get-ChildItem -Path $PROJECT_ROOT -File -Filter "*.log" -ErrorAction SilentlyContinue | 
        Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-$LogDays) }
    $rootLogSize = ($rootLogFiles | Measure-Object -Property Length -Sum).Sum
    $rootLogSizeGB = [math]::Round($rootLogSize / 1GB, 2)
    Write-Host "  Found $($rootLogFiles.Count) root log files ($rootLogSizeGB GB)" -ForegroundColor Cyan
    foreach ($file in $rootLogFiles) {
        $itemsToDelete += @{
            Path = $file.FullName
            Size = $file.Length
            Type = "Log File"
        }
    }
    $totalSize += $rootLogSize
}

# 3. MLflow実行履歴（mlruns/）
if ($CleanMLRuns) {
    Write-Host "[STEP 4] Scanning MLflow runs..." -ForegroundColor Green
    $mlrunsPath = "$PROJECT_ROOT\mlruns"
    if (Test-Path $mlrunsPath) {
        $mlrunsSize = (Get-ChildItem -Path $mlrunsPath -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        $mlrunsSizeGB = [math]::Round($mlrunsSize / 1GB, 2)
        Write-Host "  Found MLflow runs directory ($mlrunsSizeGB GB)" -ForegroundColor Cyan
        $itemsToDelete += @{
            Path = $mlrunsPath
            Size = $mlrunsSize
            Type = "MLflow Runs"
        }
        $totalSize += $mlrunsSize
    }
}

# 4. W&B実行履歴（wandb/）
if ($CleanWandb) {
    Write-Host "[STEP 5] Scanning W&B runs..." -ForegroundColor Green
    $wandbPath = "$PROJECT_ROOT\wandb"
    if (Test-Path $wandbPath) {
        $wandbSize = (Get-ChildItem -Path $wandbPath -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        $wandbSizeGB = [math]::Round($wandbSize / 1GB, 2)
        Write-Host "  Found W&B runs directory ($wandbSizeGB GB)" -ForegroundColor Cyan
        $itemsToDelete += @{
            Path = $wandbPath
            Size = $wandbSize
            Type = "W&B Runs"
        }
        $totalSize += $wandbSize
    }
}

# 5. TensorBoard実行履歴（runs/）
if ($CleanRuns) {
    Write-Host "[STEP 6] Scanning TensorBoard runs..." -ForegroundColor Green
    $runsPath = "$PROJECT_ROOT\runs"
    if (Test-Path $runsPath) {
        $runsSize = (Get-ChildItem -Path $runsPath -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        $runsSizeGB = [math]::Round($runsSize / 1GB, 2)
        Write-Host "  Found TensorBoard runs directory ($runsSizeGB GB)" -ForegroundColor Cyan
        $itemsToDelete += @{
            Path = $runsPath
            Size = $runsSize
            Type = "TensorBoard Runs"
        }
        $totalSize += $runsSize
    }
}

# 6. 古いデータベースバックアップ
Write-Host "[STEP 7] Scanning old database backups..." -ForegroundColor Green
$dbBackups = Get-ChildItem -Path "$PROJECT_ROOT\database" -File -Filter "*.backup_*" -ErrorAction SilentlyContinue | 
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) }
$dbBackupSize = ($dbBackups | Measure-Object -Property Length -Sum).Sum
$dbBackupSizeGB = [math]::Round($dbBackupSize / 1GB, 2)
Write-Host "  Found $($dbBackups.Count) old database backups ($dbBackupSizeGB GB)" -ForegroundColor Cyan
foreach ($file in $dbBackups) {
    $itemsToDelete += @{
        Path = $file.FullName
        Size = $file.Length
        Type = "Database Backup"
    }
}
$totalSize += $dbBackupSize

# 7. 一時ファイル（*.tmp, *.temp）
Write-Host "[STEP 8] Scanning temporary files..." -ForegroundColor Green
$tempFiles = Get-ChildItem -Path $PROJECT_ROOT -File -Include "*.tmp", "*.temp" -Recurse -ErrorAction SilentlyContinue
$tempSize = ($tempFiles | Measure-Object -Property Length -Sum).Sum
$tempSizeGB = [math]::Round($tempSize / 1GB, 2)
Write-Host "  Found $($tempFiles.Count) temporary files ($tempSizeGB GB)" -ForegroundColor Cyan
foreach ($file in $tempFiles) {
    $itemsToDelete += @{
        Path = $file.FullName
        Size = $file.Length
        Type = "Temporary File"
    }
}
$totalSize += $tempSize

# 8. D:\webdataset の古いチェックポイント（オプション）
if ($CleanCheckpoints) {
    Write-Host "[STEP 9] Scanning old checkpoints in D:\webdataset..." -ForegroundColor Green
    $checkpointPath = "D:\webdataset\checkpoints"
    if (Test-Path $checkpointPath) {
        $oldCheckpoints = Get-ChildItem -Path $checkpointPath -Recurse -File -Filter "*.pkl" -ErrorAction SilentlyContinue | 
            Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) }
        $checkpointSize = ($oldCheckpoints | Measure-Object -Property Length -Sum).Sum
        $checkpointSizeGB = [math]::Round($checkpointSize / 1GB, 2)
        Write-Host "  Found $($oldCheckpoints.Count) old checkpoints ($checkpointSizeGB GB)" -ForegroundColor Cyan
        foreach ($file in $oldCheckpoints) {
            $itemsToDelete += @{
                Path = $file.FullName
                Size = $file.Length
                Type = "Checkpoint"
            }
        }
        $totalSize += $checkpointSize
    }
}

# 9. Windows一時ファイル（%TEMP%, %LOCALAPPDATA%\Temp）
Write-Host "[STEP 10] Scanning Windows temporary files..." -ForegroundColor Green
$tempPaths = @(
    $env:TEMP,
    "$env:LOCALAPPDATA\Temp",
    "$env:USERPROFILE\AppData\Local\Temp"
)

foreach ($tempPath in $tempPaths) {
    if (Test-Path $tempPath) {
        try {
            $tempFiles = Get-ChildItem -Path $tempPath -Recurse -File -ErrorAction SilentlyContinue | 
                Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) }
            $tempPathSize = ($tempFiles | Measure-Object -Property Length -Sum).Sum
            $tempPathSizeGB = [math]::Round($tempPathSize / 1GB, 2)
            Write-Host "  Found $($tempFiles.Count) files in $tempPath ($tempPathSizeGB GB)" -ForegroundColor Cyan
            foreach ($file in $tempFiles) {
                $itemsToDelete += @{
                    Path = $file.FullName
                    Size = $file.Length
                    Type = "Windows Temp File"
                }
            }
            $totalSize += $tempPathSize
        } catch {
            Write-Host "  [WARNING] Failed to scan $tempPath : $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
}

# 10. Pythonキャッシュ（pip cache, __pycache__）
Write-Host "[STEP 11] Scanning Python caches..." -ForegroundColor Green
$pythonCachePaths = @(
    "$env:LOCALAPPDATA\pip\Cache",
    "$env:USERPROFILE\.cache\pip"
)

foreach ($cachePath in $pythonCachePaths) {
    if (Test-Path $cachePath) {
        try {
            $cacheSize = (Get-ChildItem -Path $cachePath -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
            $cacheSizeGB = [math]::Round($cacheSize / 1GB, 2)
            Write-Host "  Found Python cache in $cachePath ($cacheSizeGB GB)" -ForegroundColor Cyan
            $itemsToDelete += @{
                Path = $cachePath
                Size = $cacheSize
                Type = "Python Cache"
            }
            $totalSize += $cacheSize
        } catch {
            Write-Host "  [WARNING] Failed to scan $cachePath : $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
}

# 11. Hugging Faceキャッシュ
Write-Host "[STEP 12] Scanning Hugging Face cache..." -ForegroundColor Green
$hfCachePath = "$env:USERPROFILE\.cache\huggingface"
if (Test-Path $hfCachePath) {
    try {
        $hfCacheSize = (Get-ChildItem -Path $hfCachePath -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        $hfCacheSizeGB = [math]::Round($hfCacheSize / 1GB, 2)
        Write-Host "  Found Hugging Face cache ($hfCacheSizeGB GB)" -ForegroundColor Cyan
        $itemsToDelete += @{
            Path = $hfCachePath
            Size = $hfCacheSize
            Type = "Hugging Face Cache"
        }
        $totalSize += $hfCacheSize
    } catch {
        Write-Host "  [WARNING] Failed to scan Hugging Face cache : $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

# 12. ブラウザキャッシュ（Chrome, Edge）
Write-Host "[STEP 13] Scanning browser caches..." -ForegroundColor Green
$browserCachePaths = @(
    "$env:LOCALAPPDATA\Google\Chrome\User Data\Default\Cache",
    "$env:LOCALAPPDATA\Microsoft\Edge\User Data\Default\Cache"
)

foreach ($cachePath in $browserCachePaths) {
    if (Test-Path $cachePath) {
        try {
            $cacheSize = (Get-ChildItem -Path $cachePath -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
            $cacheSizeGB = [math]::Round($cacheSize / 1GB, 2)
            Write-Host "  Found browser cache in $cachePath ($cacheSizeGB GB)" -ForegroundColor Cyan
            $itemsToDelete += @{
                Path = $cachePath
                Size = $cacheSize
                Type = "Browser Cache"
            }
            $totalSize += $cacheSize
        } catch {
            Write-Host "  [WARNING] Failed to scan $cachePath : $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
}

# サマリー表示
$totalSizeGB = [math]::Round($totalSize / 1GB, 2)
Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Cleanup Summary" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Total items to delete: $($itemsToDelete.Count)" -ForegroundColor Yellow
Write-Host "Total size to free: $totalSizeGB GB" -ForegroundColor Yellow
Write-Host ""

# 削除実行
if ($DryRun) {
    Write-Host "[DRY RUN] No files will be deleted. Use -DryRun:`$false to actually delete files." -ForegroundColor Yellow
} else {
    Write-Host "[DELETE] Starting deletion..." -ForegroundColor Red
    $deletedCount = 0
    $deletedSize = 0
    
    foreach ($item in $itemsToDelete) {
        try {
            if (Test-Path $item.Path) {
                if ((Get-Item $item.Path) -is [System.IO.DirectoryInfo]) {
                    Remove-Item -Path $item.Path -Recurse -Force -ErrorAction Stop
                } else {
                    Remove-Item -Path $item.Path -Force -ErrorAction Stop
                }
                $deletedCount++
                $deletedSize += $item.Size
                Write-Host "  [OK] Deleted: $($item.Type) - $($item.Path)" -ForegroundColor Green
            }
        } catch {
            Write-Host "  [ERROR] Failed to delete: $($item.Path) - $($_.Exception.Message)" -ForegroundColor Red
        }
    }
    
    $deletedSizeGB = [math]::Round($deletedSize / 1GB, 2)
    Write-Host ""
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host "Deletion Complete" -ForegroundColor Cyan
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host "Deleted items: $deletedCount / $($itemsToDelete.Count)" -ForegroundColor Green
    Write-Host "Freed space: $deletedSizeGB GB" -ForegroundColor Green
    Write-Host ""
    
    # 最終的なディスク容量確認
    Write-Host "[FINAL] Checking disk space after cleanup..." -ForegroundColor Green
    try {
        $cDrive = Get-PSDrive C
        $dDrive = Get-PSDrive D -ErrorAction SilentlyContinue
        
        if ($cDrive) {
            $cFreeGB = [math]::Round($cDrive.Free / 1GB, 2)
            Write-Host "  C Drive - Free: $cFreeGB GB" -ForegroundColor Cyan
        }
        
        if ($dDrive) {
            $dFreeGB = [math]::Round($dDrive.Free / 1GB, 2)
            Write-Host "  D Drive - Free: $dFreeGB GB" -ForegroundColor Cyan
        }
    } catch {
        Write-Host "  [ERROR] Failed to check disk space: $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Cleanup script completed" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

