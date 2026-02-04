# SO8T Pagefile Setup Script
# Dドライブにページングファイルを設定

# 管理者権限チェック
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "[ERROR] This script requires administrator privileges." -ForegroundColor Red
    Write-Host "[INFO] Please run PowerShell as Administrator and try again." -ForegroundColor Yellow
    Write-Host "[INFO] Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    exit 1
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SO8T Pagefile Setup - D Drive" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 現在のページングファイル設定を確認
Write-Host "[INFO] Current pagefile settings:" -ForegroundColor Yellow
Get-WmiObject -Class Win32_PageFileUsage | Select-Object Name, AllocatedBaseSize, CurrentUsage | Format-List

Write-Host ""
Write-Host "[INFO] Setting pagefile on D drive..." -ForegroundColor Yellow

# Cドライブのページングファイルを無効化
try {
    $cDrive = Get-WmiObject -Class Win32_PageFileSetting | Where-Object { $_.Name -like "C:\*" }
    if ($cDrive) {
        Write-Host "[INFO] Removing pagefile from C drive..." -ForegroundColor Yellow
        $cDrive.Delete()
        Write-Host "[OK] C drive pagefile removed" -ForegroundColor Green
    }
} catch {
    Write-Host "[WARNING] Failed to remove C drive pagefile: $($_.Exception.Message)" -ForegroundColor Yellow
}

# Dドライブのページングファイルを設定
try {
    $dDrivePagefile = Get-WmiObject -Class Win32_PageFileSetting | Where-Object { $_.Name -like "D:\*" }
    
    if ($dDrivePagefile) {
        Write-Host "[INFO] Updating existing D drive pagefile..." -ForegroundColor Yellow
        $dDrivePagefile.InitialSize = 16384  # 16GB
        $dDrivePagefile.MaximumSize = 32768  # 32GB
        $dDrivePagefile.Put()
        Write-Host "[OK] D drive pagefile updated" -ForegroundColor Green
    } else {
        Write-Host "[INFO] Creating new pagefile on D drive..." -ForegroundColor Yellow
        $pagefile = ([WmiClass]"Win32_PageFileSetting").CreateInstance()
        $pagefile.Name = "D:\pagefile.sys"
        $pagefile.InitialSize = 16384  # 16GB
        $pagefile.MaximumSize = 32768  # 32GB
        $pagefile.Put()
        Write-Host "[OK] D drive pagefile created" -ForegroundColor Green
    }
} catch {
    Write-Host "[ERROR] Failed to set D drive pagefile: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "[INFO] Please set pagefile manually:" -ForegroundColor Yellow
    Write-Host "  1. System Properties > Advanced > Performance Settings" -ForegroundColor White
    Write-Host "  2. Advanced tab > Virtual Memory > Change" -ForegroundColor White
    Write-Host "  3. Select D drive > Custom size" -ForegroundColor White
    Write-Host "  4. Initial size: 16384 MB, Maximum size: 32768 MB" -ForegroundColor White
    Write-Host "  5. Click Set > OK" -ForegroundColor White
    exit 1
}

Write-Host ""
Write-Host "[OK] Pagefile settings updated successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "[WARNING] System restart is required for changes to take effect." -ForegroundColor Yellow
Write-Host "[INFO] After restart, run SO8T training again." -ForegroundColor Cyan
Write-Host ""
Write-Host "New pagefile settings:" -ForegroundColor Yellow
Get-WmiObject -Class Win32_PageFileSetting | Select-Object Name, InitialSize, MaximumSize | Format-Table -AutoSize

Write-Host ""
$restart = Read-Host "Do you want to restart now? (Y/N)"
if ($restart -eq "Y" -or $restart -eq "y") {
    Write-Host "[INFO] Restarting system in 10 seconds..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10
    Restart-Computer -Force
} else {
    Write-Host "[INFO] Please restart manually to apply changes." -ForegroundColor Yellow
}








