@echo off
REM SO8T Desktop Progress Display - Immediate Execution
REM Opens PowerShell window on desktop to show current progress

echo Opening SO8T Progress Report on Desktop...
start "SO8T Progress Report" powershell.exe -ExecutionPolicy Bypass -Command "& { $host.UI.RawUI.WindowTitle = 'SO8T Progress Report'; cd 'C:\Users\downl\Desktop\SO8T'; .\scripts\utils\show_progress_report.ps1; Read-Host 'Press Enter to close' }"
echo Progress report window opened on desktop.
pause
