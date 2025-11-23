# Dドライブのページファイルサイズを100GBに設定するスクリプト
# 管理者権限で実行が必要

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "D Drive Page File Configuration (100GB)" -ForegroundColor Cyan
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

# 100GB = 102400 MB
$targetSizeMB = 102400
$targetSizeGB = 100

Write-Host "[INFO] Target Page File Size: $targetSizeGB GB ($targetSizeMB MB)" -ForegroundColor Green
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

# Dドライブのページファイル設定を確認
$dDrivePageFile = $currentPageFiles | Where-Object { $_.Name -like "*D:*" }

if ($dDrivePageFile) {
    Write-Host "[INFO] D: drive page file found" -ForegroundColor Green
    Write-Host "[INFO] Current D: drive settings:" -ForegroundColor Yellow
    Write-Host "  Initial: $($dDrivePageFile.InitialSize) MB ($([math]::Round($dDrivePageFile.InitialSize / 1024, 2)) GB)" -ForegroundColor White
    Write-Host "  Maximum: $($dDrivePageFile.MaximumSize) MB ($([math]::Round($dDrivePageFile.MaximumSize / 1024, 2)) GB)" -ForegroundColor White
    Write-Host ""
    
    # サイズを100GBに設定するか確認
    Write-Host "[QUESTION] Do you want to set D: drive page file size to $targetSizeGB GB?" -ForegroundColor Yellow
    Write-Host "  Current: Initial=$($dDrivePageFile.InitialSize) MB, Maximum=$($dDrivePageFile.MaximumSize) MB" -ForegroundColor White
    Write-Host "  New: Initial=$targetSizeMB MB, Maximum=$targetSizeMB MB" -ForegroundColor White
    Write-Host ""
    $response = Read-Host "Enter 'yes' to proceed, or 'no' to cancel"
    
    if ($response -ne "yes") {
        Write-Host "[INFO] Operation cancelled." -ForegroundColor Yellow
        exit 0
    }
    
    # ページファイルサイズを更新
    Write-Host "[INFO] Updating D: drive page file size to $targetSizeGB GB..." -ForegroundColor Green
    try {
        $dDrivePageFile.InitialSize = $targetSizeMB
        $dDrivePageFile.MaximumSize = $targetSizeMB
        Set-CimInstance -InputObject $dDrivePageFile
        Write-Host "[OK] D: drive page file size updated successfully!" -ForegroundColor Green
        Write-Host "[INFO] New settings: Initial=$targetSizeMB MB ($targetSizeGB GB), Maximum=$targetSizeMB MB ($targetSizeGB GB)" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "[WARNING] Changes will take effect after system restart." -ForegroundColor Yellow
        Write-Host "[INFO] Please restart your computer for the changes to take effect." -ForegroundColor Cyan
    } catch {
        Write-Host "[ERROR] Failed to update page file size: $_" -ForegroundColor Red
        Write-Host "[INFO] Trying alternative method..." -ForegroundColor Yellow
        
        # 代替方法: WMIを使用
        try {
            $pageFileWMI = Get-WmiObject -Class Win32_PageFileSetting | Where-Object { $_.Name -like "*D:*" }
            if ($pageFileWMI) {
                $pageFileWMI.InitialSize = $targetSizeMB
                $pageFileWMI.MaximumSize = $targetSizeMB
                $pageFileWMI.Put()
                Write-Host "[OK] D: drive page file size updated successfully (using WMI)!" -ForegroundColor Green
                Write-Host "[INFO] New settings: Initial=$targetSizeMB MB ($targetSizeGB GB), Maximum=$targetSizeMB MB ($targetSizeGB GB)" -ForegroundColor Yellow
                Write-Host ""
                Write-Host "[WARNING] Changes will take effect after system restart." -ForegroundColor Yellow
                Write-Host "[INFO] Please restart your computer for the changes to take effect." -ForegroundColor Cyan
            } else {
                throw "D: drive page file not found in WMI"
            }
        } catch {
            Write-Host "[ERROR] Failed to update page file size: $_" -ForegroundColor Red
            Write-Host "[INFO] You may need to use System Properties GUI instead:" -ForegroundColor Yellow
            Write-Host "  1. Run: sysdm.cpl" -ForegroundColor Cyan
            Write-Host "  2. Advanced tab -> Performance Settings -> Advanced tab" -ForegroundColor Cyan
            Write-Host "  3. Virtual memory -> Change" -ForegroundColor Cyan
            Write-Host "  4. Uncheck 'Automatically manage paging file size'" -ForegroundColor Cyan
            Write-Host "  5. Select D: drive -> Custom size" -ForegroundColor Cyan
            Write-Host "  6. Set Initial: $targetSizeMB MB, Maximum: $targetSizeMB MB" -ForegroundColor Cyan
            Write-Host "  7. Click Set and OK" -ForegroundColor Cyan
            Write-Host "  8. Restart computer" -ForegroundColor Cyan
            exit 1
        }
    }
} else {
    Write-Host "[INFO] No page file found on D: drive" -ForegroundColor Yellow
    Write-Host "[QUESTION] Do you want to create a page file on D: drive ($targetSizeGB GB)?" -ForegroundColor Yellow
    Write-Host ""
    $response = Read-Host "Enter 'yes' to proceed, or 'no' to cancel"
    
    if ($response -ne "yes") {
        Write-Host "[INFO] Operation cancelled." -ForegroundColor Yellow
        exit 0
    }
    
    # Dドライブにページファイルを作成
    Write-Host "[INFO] Creating page file on D: drive ($targetSizeGB GB)..." -ForegroundColor Green
    try {
        # WMIを使用してページファイルを作成
        $pageFile = New-CimInstance -ClassName Win32_PageFileSetting -Property @{
            Name = "D:\pagefile.sys"
            InitialSize = $targetSizeMB
            MaximumSize = $targetSizeMB
        }
        Write-Host "[OK] D: drive page file created successfully!" -ForegroundColor Green
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
        Write-Host "  5. Select D: drive -> Custom size" -ForegroundColor Cyan
        Write-Host "  6. Set Initial: $targetSizeMB MB, Maximum: $targetSizeMB MB" -ForegroundColor Cyan
        Write-Host "  7. Click Set and OK" -ForegroundColor Cyan
        Write-Host "  8. Restart computer" -ForegroundColor Cyan
        exit 1
    }
}

Write-Host ""
Write-Host "[SUCCESS] D: drive page file configuration completed!" -ForegroundColor Green
Write-Host "[INFO] Page file size: $targetSizeGB GB ($targetSizeMB MB)" -ForegroundColor Yellow
Write-Host "[INFO] Please restart your computer for changes to take effect." -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")






