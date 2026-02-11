# Borea-Phi-3.5-mini Git LFS修復スクリプト
# ディスク容量不足問題を解決

param(
    [string]$TargetDrive = "D",
    [switch]$CleanTemp = $false
)

$ErrorActionPreference = "Stop"

Write-Host "[INFO] Borea-Phi-3.5-mini Git LFS修復スクリプト開始" -ForegroundColor Green

# 1. ディスク容量確認
Write-Host "`n[STEP 1] ディスク容量確認..." -ForegroundColor Yellow
$drives = Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Name -match "^[CDE]$" }
foreach ($drive in $drives) {
    $freeGB = [math]::Round($drive.Free / 1GB, 2)
    $usedGB = [math]::Round($drive.Used / 1GB, 2)
    Write-Host "  $($drive.Name): 空き $freeGB GB / 使用 $usedGB GB" -ForegroundColor $(if ($freeGB -gt 10) { "Green" } else { "Red" })
}

# 2. 一時ファイルクリーンアップ（オプション）
if ($CleanTemp) {
    Write-Host "`n[STEP 2] 一時ファイルクリーンアップ..." -ForegroundColor Yellow
    $tempPaths = @(
        $env:TEMP,
        "$env:USERPROFILE\AppData\Local\Temp",
        "$env:USERPROFILE\AppData\Local\Microsoft\Windows\INetCache"
    )
    
    $totalFreed = 0
    foreach ($tempPath in $tempPaths) {
        if (Test-Path $tempPath) {
            try {
                $files = Get-ChildItem -Path $tempPath -Recurse -File -ErrorAction SilentlyContinue
                $size = ($files | Measure-Object -Property Length -Sum).Sum
                $sizeGB = [math]::Round($size / 1GB, 2)
                Write-Host "  $tempPath : $sizeGB GB" -ForegroundColor Cyan
                
                # 7日以上古いファイルを削除
                $oldFiles = $files | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) }
                $freed = ($oldFiles | Measure-Object -Property Length -Sum).Sum
                $freedGB = [math]::Round($freed / 1GB, 2)
                
                if ($freedGB -gt 0) {
                    $oldFiles | Remove-Item -Force -ErrorAction SilentlyContinue
                    $totalFreed += $freedGB
                    Write-Host "    削除: $freedGB GB" -ForegroundColor Green
                }
            } catch {
                Write-Host "    エラー: $($_.Exception.Message)" -ForegroundColor Red
            }
        }
    }
    Write-Host "  合計削除: $totalFreed GB" -ForegroundColor Green
}

# 3. モデルディレクトリの確認
Write-Host "`n[STEP 3] モデルディレクトリ確認..." -ForegroundColor Yellow
$boreaPath = "C:\Users\downl\Desktop\SO8T\Borea-Phi-3.5-mini-Instruct-Common"
if (Test-Path $boreaPath) {
    $safetensors = Get-ChildItem -Path $boreaPath -Filter "*.safetensors" -ErrorAction SilentlyContinue
    foreach ($file in $safetensors) {
        $sizeGB = [math]::Round($file.Length / 1GB, 2)
        Write-Host "  $($file.Name): $sizeGB GB" -ForegroundColor Cyan
    }
} else {
    Write-Host "  モデルディレクトリが見つかりません: $boreaPath" -ForegroundColor Red
    exit 1
}

# 4. 別ドライブへの移動オプション
$cDrive = Get-PSDrive C
$cFreeGB = [math]::Round($cDrive.Free / 1GB, 2)

if ($cFreeGB -lt 10) {
    Write-Host "`n[WARNING] C drive free space is insufficient ($cFreeGB GB)" -ForegroundColor Red
    Write-Host "[OPTION] Moving to another drive is recommended" -ForegroundColor Yellow
    
    $targetPath = "${TargetDrive}:\SO8T\Borea-Phi-3.5-mini-Instruct-Common"
    Write-Host "  移動先: $targetPath" -ForegroundColor Cyan
    
    $response = Read-Host "  移動しますか? (y/n)"
    if ($response -eq "y") {
        Write-Host "`n[STEP 4] モデルを別ドライブに移動..." -ForegroundColor Yellow
        
        # 移動先ディレクトリ作成
        $targetDir = Split-Path $targetPath
        if (-not (Test-Path $targetDir)) {
            New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
        }
        
        # 移動実行
        try {
            Move-Item -Path $boreaPath -Destination $targetPath -Force
            Write-Host "  移動完了: $targetPath" -ForegroundColor Green
            
            # シンボリックリンク作成（オプション）
            $response2 = Read-Host "  元の場所にシンボリックリンクを作成しますか? (y/n)"
            if ($response2 -eq "y") {
                $linkPath = Split-Path $boreaPath -Parent
                New-Item -ItemType SymbolicLink -Path $boreaPath -Target $targetPath -Force | Out-Null
                Write-Host "  シンボリックリンク作成完了" -ForegroundColor Green
            }
        } catch {
            Write-Host "  移動エラー: $($_.Exception.Message)" -ForegroundColor Red
        }
    }
}

# 5. Git LFS再取得
Write-Host "`n[STEP 5] Git LFS再取得..." -ForegroundColor Yellow
$currentPath = if (Test-Path $boreaPath) { $boreaPath } else { $targetPath }

if (Test-Path $currentPath) {
    Push-Location $currentPath
    try {
        Write-Host "  Git LFS fetch実行中..." -ForegroundColor Cyan
        git lfs fetch --all 2>&1 | Out-Null
        
        Write-Host "  Git LFS checkout実行中..." -ForegroundColor Cyan
        git lfs checkout 2>&1 | Out-Null
        
        Write-Host "  Git LFS pull実行中..." -ForegroundColor Cyan
        git lfs pull 2>&1 | Out-Null
        
        Write-Host "  [OK] Git LFS再取得完了" -ForegroundColor Green
    } catch {
        Write-Host "  [ERROR] Git LFS再取得失敗: $($_.Exception.Message)" -ForegroundColor Red
    } finally {
        Pop-Location
    }
}

# 6. 最終確認
Write-Host "`n[STEP 6] 最終確認..." -ForegroundColor Yellow
$finalPath = if (Test-Path $boreaPath) { $boreaPath } else { $targetPath }
if (Test-Path $finalPath) {
    $finalFiles = Get-ChildItem -Path $finalPath -Filter "*.safetensors" -ErrorAction SilentlyContinue
    $allPresent = $true
    foreach ($file in $finalFiles) {
        $sizeGB = [math]::Round($file.Length / 1GB, 2)
        if ($file.Length -gt 0) {
            Write-Host "  [OK] $($file.Name): $sizeGB GB" -ForegroundColor Green
        } else {
            Write-Host "  [NG] $($file.Name): ファイルサイズが0" -ForegroundColor Red
            $allPresent = $false
        }
    }
    
    if ($allPresent) {
        Write-Host "`n[SUCCESS] すべてのファイルが正常に存在します" -ForegroundColor Green
    } else {
        Write-Host "`n[WARNING] 一部のファイルに問題があります" -ForegroundColor Yellow
    }
}

Write-Host "`n[INFO] Fix script completed" -ForegroundColor Green

