# Cドライブのページファイルサイズを増やすスクリプト
# 管理者権限で実行が必要

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Page File Size Configuration" -ForegroundColor Cyan
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

# 現在のRAMサイズを取得
$computerSystem = Get-CimInstance Win32_ComputerSystem
$totalRAM = $computerSystem.TotalPhysicalMemory / 1MB
Write-Host "[INFO] Total RAM: $([math]::Round($totalRAM / 1024, 2)) GB" -ForegroundColor Green

# 推奨サイズを計算
$recommendedMin = [math]::Round($totalRAM * 1.5, 0)
$recommendedMax = [math]::Round($totalRAM * 3, 0)

Write-Host "[INFO] Recommended Page File Size:" -ForegroundColor Green
Write-Host "  Minimum: $recommendedMin MB ($([math]::Round($recommendedMin / 1024, 2)) GB)" -ForegroundColor Yellow
Write-Host "  Maximum: $recommendedMax MB ($([math]::Round($recommendedMax / 1024, 2)) GB)" -ForegroundColor Yellow
Write-Host ""

# 現在のページファイル設定を確認
Write-Host "[INFO] Current Page File Settings:" -ForegroundColor Green
$currentPageFiles = Get-CimInstance -ClassName Win32_PageFileSetting
foreach ($pageFile in $currentPageFiles) {
    Write-Host "  Drive: $($pageFile.Name)" -ForegroundColor Cyan
    Write-Host "    Initial Size: $($pageFile.InitialSize) MB" -ForegroundColor White
    Write-Host "    Maximum Size: $($pageFile.MaximumSize) MB" -ForegroundColor White
}
Write-Host ""

# Cドライブのページファイル設定を確認
$cDrivePageFile = $currentPageFiles | Where-Object { $_.Name -like "*C:*" }

if ($cDrivePageFile) {
    Write-Host "[INFO] C: drive page file found" -ForegroundColor Green
    Write-Host "[INFO] Current C: drive settings:" -ForegroundColor Yellow
    Write-Host "  Initial: $($cDrivePageFile.InitialSize) MB" -ForegroundColor White
    Write-Host "  Maximum: $($cDrivePageFile.MaximumSize) MB" -ForegroundColor White
    Write-Host ""
    
    # サイズを増やすか確認
    Write-Host "[QUESTION] Do you want to increase C: drive page file size?" -ForegroundColor Yellow
    Write-Host "  Current: Initial=$($cDrivePageFile.InitialSize) MB, Maximum=$($cDrivePageFile.MaximumSize) MB" -ForegroundColor White
    Write-Host "  Recommended: Initial=$recommendedMin MB, Maximum=$recommendedMax MB" -ForegroundColor White
    Write-Host ""
    $response = Read-Host "Enter 'yes' to proceed, or 'no' to cancel"
    
    if ($response -ne "yes") {
        Write-Host "[INFO] Operation cancelled." -ForegroundColor Yellow
        exit 0
    }
    
    # ページファイルサイズを更新
    Write-Host "[INFO] Updating C: drive page file size..." -ForegroundColor Green
    try {
        $cDrivePageFile.InitialSize = $recommendedMin
        $cDrivePageFile.MaximumSize = $recommendedMax
        Set-CimInstance -InputObject $cDrivePageFile
        Write-Host "[OK] C: drive page file size updated successfully!" -ForegroundColor Green
        Write-Host "[INFO] New settings: Initial=$recommendedMin MB, Maximum=$recommendedMax MB" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "[WARNING] Changes will take effect after system restart." -ForegroundColor Yellow
        Write-Host "[INFO] Please restart your computer for the changes to take effect." -ForegroundColor Cyan
    } catch {
        Write-Host "[ERROR] Failed to update page file size: $_" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[INFO] No page file found on C: drive" -ForegroundColor Yellow
    Write-Host "[QUESTION] Do you want to create a page file on C: drive?" -ForegroundColor Yellow
    Write-Host "  Recommended: Initial=$recommendedMin MB, Maximum=$recommendedMax MB" -ForegroundColor White
    Write-Host ""
    $response = Read-Host "Enter 'yes' to proceed, or 'no' to cancel"
    
    if ($response -ne "yes") {
        Write-Host "[INFO] Operation cancelled." -ForegroundColor Yellow
        exit 0
    }
    
    # Cドライブにページファイルを作成
    Write-Host "[INFO] Creating page file on C: drive..." -ForegroundColor Green
    try {
        # WMIを使用してページファイルを作成
        $pageFile = New-CimInstance -ClassName Win32_PageFileSetting -Property @{
            Name = "C:\pagefile.sys"
            InitialSize = $recommendedMin
            MaximumSize = $recommendedMax
        }
        Write-Host "[OK] C: drive page file created successfully!" -ForegroundColor Green
        Write-Host "[INFO] Settings: Initial=$recommendedMin MB, Maximum=$recommendedMax MB" -ForegroundColor Yellow
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
        Write-Host "  5. Select C: drive -> Custom size" -ForegroundColor Cyan
        Write-Host "  6. Set Initial: $recommendedMin MB, Maximum: $recommendedMax MB" -ForegroundColor Cyan
        exit 1
    }
}

Write-Host ""
Write-Host "[SUCCESS] Page file configuration completed!" -ForegroundColor Green
Write-Host "[INFO] Please restart your computer for changes to take effect." -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")






