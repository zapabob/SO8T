# Eドライブのページファイルサイズを320GBに設定するスクリプト（220GB + 100GB追加）
# 管理者権限で実行が必要

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "E Drive Page File Configuration (320GB)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 管理者権限チェック
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "[ERROR] This script requires administrator privileges." -ForegroundColor Red
    Write-Host "[INFO] Please right-click and select 'Run as administrator'" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# 320GB = 327680 MB (220GB + 100GB)
$targetSizeMB = 327680
$targetSizeGB = 320

Write-Host "[INFO] Target Page File Size: $targetSizeGB GB ($targetSizeMB MB)" -ForegroundColor Green
Write-Host ""

# Eドライブの存在確認（複数の方法を試行）
$driveE = $null
$driveFound = $false

# Method 1: Test-Path
if (Test-Path "E:\") {
    $driveFound = $true
    Write-Host "[OK] E: drive found using Test-Path" -ForegroundColor Green
}

# Method 2: Get-PSDrive
if (-not $driveFound) {
    $driveE = Get-PSDrive -Name E -ErrorAction SilentlyContinue
    if ($driveE) {
        $driveFound = $true
        Write-Host "[OK] E: drive found using Get-PSDrive: $($driveE.Root)" -ForegroundColor Green
    }
}

# Method 3: Get-WmiObject
if (-not $driveFound) {
    try {
        $driveWMI = Get-WmiObject -Class Win32_LogicalDisk -Filter "DeviceID='E:'" -ErrorAction SilentlyContinue
        if ($driveWMI) {
            $driveFound = $true
            Write-Host "[OK] E: drive found using WMI: $($driveWMI.DeviceID)" -ForegroundColor Green
        }
    } catch {
        # Ignore WMI errors
    }
}

# Method 4: [System.IO.DriveInfo]
if (-not $driveFound) {
    try {
        $driveInfo = [System.IO.DriveInfo]::GetDrives() | Where-Object { $_.Name -eq "E:\" }
        if ($driveInfo -and $driveInfo.IsReady) {
            $driveFound = $true
            Write-Host "[OK] E: drive found using DriveInfo: $($driveInfo.Name)" -ForegroundColor Green
        }
    } catch {
        # Ignore DriveInfo errors
    }
}

if (-not $driveFound) {
    Write-Host "[ERROR] E: drive not found using any method." -ForegroundColor Red
    Write-Host "[INFO] Available drives:" -ForegroundColor Yellow
    Get-PSDrive -PSProvider FileSystem | ForEach-Object {
        Write-Host "  - $($_.Name): $($_.Root)" -ForegroundColor Cyan
    }
    Write-Host ""
    Write-Host "[INFO] Please ensure E: drive exists and is accessible." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

Write-Host "[OK] E: drive verified and accessible" -ForegroundColor Green
Write-Host ""

# 現在のページファイル設定を確認
Write-Host "[INFO] Current Page File Settings:" -ForegroundColor Green
$currentPageFiles = Get-CimInstance -ClassName Win32_PageFileSetting
foreach ($pageFile in $currentPageFiles) {
    Write-Host "  Drive: $($pageFile.Name)" -ForegroundColor Cyan
    Write-Host "    Initial Size: $($pageFile.InitialSize) MB ($([math]::Round($pageFile.InitialSize / 1024, 2)) GB)" -ForegroundColor White
    Write-Host "    Maximum Size: $($pageFile.MaximumSize) MB ($([math]::Round($pageFile.MaximumSize / 1024, 2)) GB)" -ForegroundColor White
}
Write-Host ""

# Eドライブのページファイル設定を確認
$eDrivePageFile = $currentPageFiles | Where-Object { $_.Name -like "*E:*" }

if ($eDrivePageFile) {
    Write-Host "[INFO] E: drive page file found" -ForegroundColor Green
    Write-Host "[INFO] Current E: drive settings:" -ForegroundColor Yellow
    Write-Host "  Initial: $($eDrivePageFile.InitialSize) MB ($([math]::Round($eDrivePageFile.InitialSize / 1024, 2)) GB)" -ForegroundColor White
    Write-Host "  Maximum: $($eDrivePageFile.MaximumSize) MB ($([math]::Round($eDrivePageFile.MaximumSize / 1024, 2)) GB)" -ForegroundColor White
    Write-Host ""
    
    # サイズを320GBに設定するか確認（100GB追加）
    Write-Host "[QUESTION] Do you want to set E: drive page file size to $targetSizeGB GB (adding 100GB)?" -ForegroundColor Yellow
    Write-Host "  Current: Initial=$($eDrivePageFile.InitialSize) MB, Maximum=$($eDrivePageFile.MaximumSize) MB" -ForegroundColor White
    Write-Host "  New: Initial=$targetSizeMB MB, Maximum=$targetSizeMB MB" -ForegroundColor White
    Write-Host ""
    $response = Read-Host "Enter 'yes' to proceed, or 'no' to cancel"
    
    if ($response -ne "yes") {
        Write-Host "[INFO] Operation cancelled." -ForegroundColor Yellow
        exit 0
    }
    
    # ページファイルサイズを更新
    Write-Host "[INFO] Updating E: drive page file size to $targetSizeGB GB..." -ForegroundColor Green
    try {
        $eDrivePageFile.InitialSize = $targetSizeMB
        $eDrivePageFile.MaximumSize = $targetSizeMB
        Set-CimInstance -InputObject $eDrivePageFile
        Write-Host "[OK] E: drive page file size updated successfully!" -ForegroundColor Green
        Write-Host "[INFO] New settings: Initial=$targetSizeMB MB ($targetSizeGB GB), Maximum=$targetSizeMB MB ($targetSizeGB GB)" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "[WARNING] Changes will take effect after system restart." -ForegroundColor Yellow
        Write-Host "[INFO] Please restart your computer for the changes to take effect." -ForegroundColor Cyan
    } catch {
        Write-Host "[ERROR] Failed to update page file size: $_" -ForegroundColor Red
        Write-Host "[INFO] Trying alternative method..." -ForegroundColor Yellow
        
        # 代替方法: WMIを使用
        try {
            $pageFileWMI = Get-WmiObject -Class Win32_PageFileSetting | Where-Object { $_.Name -like "*E:*" }
            if ($pageFileWMI) {
                $pageFileWMI.InitialSize = $targetSizeMB
                $pageFileWMI.MaximumSize = $targetSizeMB
                $pageFileWMI.Put()
                Write-Host "[OK] E: drive page file size updated successfully (using WMI)!" -ForegroundColor Green
                Write-Host "[INFO] New settings: Initial=$targetSizeMB MB ($targetSizeGB GB), Maximum=$targetSizeMB MB ($targetSizeGB GB)" -ForegroundColor Yellow
                Write-Host ""
                Write-Host "[WARNING] Changes will take effect after system restart." -ForegroundColor Yellow
                Write-Host "[INFO] Please restart your computer for the changes to take effect." -ForegroundColor Cyan
            } else {
                throw "E: drive page file not found in WMI"
            }
        } catch {
            Write-Host "[ERROR] Failed to update page file size: $_" -ForegroundColor Red
            Write-Host "[INFO] You may need to use System Properties GUI instead:" -ForegroundColor Yellow
            Write-Host "  1. Run: sysdm.cpl" -ForegroundColor Cyan
            Write-Host "  2. Advanced tab -> Performance Settings -> Advanced tab" -ForegroundColor Cyan
            Write-Host "  3. Virtual memory -> Change" -ForegroundColor Cyan
            Write-Host "  4. Uncheck 'Automatically manage paging file size'" -ForegroundColor Cyan
            Write-Host "  5. Select E: drive -> Custom size" -ForegroundColor Cyan
            Write-Host "  6. Set Initial: $targetSizeMB MB, Maximum: $targetSizeMB MB" -ForegroundColor Cyan
            Write-Host "  7. Click Set and OK" -ForegroundColor Cyan
            Write-Host "  8. Restart computer" -ForegroundColor Cyan
            exit 1
        }
    }
} else {
    Write-Host "[INFO] No page file found on E: drive" -ForegroundColor Yellow
    Write-Host "[QUESTION] Do you want to create a page file on E: drive ($targetSizeGB GB)?" -ForegroundColor Yellow
    Write-Host ""
    $response = Read-Host "Enter 'yes' to proceed, or 'no' to cancel"
    
    if ($response -ne "yes") {
        Write-Host "[INFO] Operation cancelled." -ForegroundColor Yellow
        exit 0
    }
    
    # Eドライブにページファイルを作成
    Write-Host "[INFO] Creating page file on E: drive ($targetSizeGB GB)..." -ForegroundColor Green
    try {
        # WMIを使用してページファイルを作成
        $pageFile = New-CimInstance -ClassName Win32_PageFileSetting -Property @{
            Name = "E:\pagefile.sys"
            InitialSize = $targetSizeMB
            MaximumSize = $targetSizeMB
        }
        Write-Host "[OK] E: drive page file created successfully!" -ForegroundColor Green
        Write-Host "[INFO] Settings: Initial=$targetSizeMB MB ($targetSizeGB GB), Maximum=$targetSizeMB MB ($targetSizeGB GB)" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "[WARNING] Changes will take effect after system restart." -ForegroundColor Yellow
        Write-Host "[INFO] Please restart your computer for the changes to take effect." -ForegroundColor Cyan
    } catch {
        Write-Host "[ERROR] Failed to create page file: $_" -ForegroundColor Red
        Write-Host "[INFO] You may need to use System Properties GUI instead:" -ForegroundColor Yellow
        Write-Host "  1. Run: sysdm.cpl" -ForegroundColor Cyan
        Write-Host "  2. Advanced tab -> Performance Settings -> Advanced tab" -ForegroundColor Cyan
        Write-Host "  3. Virtual memory -> Change" -ForegroundColor Cyan
        Write-Host "  4. Uncheck 'Automatically manage paging file size'" -ForegroundColor Cyan
        Write-Host "  5. Select E: drive -> Custom size" -ForegroundColor Cyan
        Write-Host "  6. Set Initial: $targetSizeMB MB, Maximum: $targetSizeMB MB" -ForegroundColor Cyan
        Write-Host "  7. Click Set and OK" -ForegroundColor Cyan
        Write-Host "  8. Restart computer" -ForegroundColor Cyan
        exit 1
    }
}

Write-Host ""
Write-Host "[SUCCESS] E: drive page file configuration completed!" -ForegroundColor Green
Write-Host "[INFO] Page file size: $targetSizeGB GB ($targetSizeMB MB)" -ForegroundColor Yellow
Write-Host "[INFO] Please restart your computer for changes to take effect." -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

