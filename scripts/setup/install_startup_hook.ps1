# ==========================================
# AEGIS Startup Hook Installer
# ==========================================

$TargetFile = "C:\Users\downl\Desktop\SO8T\launch_on_boot.bat"
$ShortcutName = "AEGIS_Auto_Train.lnk"
$StartupDir = [Environment]::GetFolderPath("Startup")
$ShortcutPath = Join-Path $StartupDir $ShortcutName

Write-Host "🔥 Installing AEGIS Hook to Windows Startup..." -ForegroundColor Cyan
Write-Host "   Target: $TargetFile"
Write-Host "   Dest  : $ShortcutPath"

# COMオブジェクトを使ってショートカット作成
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)

$Shortcut.TargetPath = $TargetFile
$Shortcut.WorkingDirectory = "C:\Users\downl\Desktop\SO8T"
$Shortcut.WindowStyle = 7 # 7=Minimized (最小化で起動して邪魔しない)
$Shortcut.Description = "AEGIS-v2.0 Autonomous Training Loader"
$Shortcut.Save()

Write-Host ""
Write-Host "✅ Installation Complete!" -ForegroundColor Green
Write-Host "   Next time you reboot, AEGIS will rise automatically."
Write-Host "   (PC再起動時に自動で学習が始まります)"
Write-Host ""
