@echo off
REM Create SO8T Desktop Progress Display Task
REM This batch file creates a Windows Task Scheduler task to display progress on desktop

echo Creating SO8T Desktop Progress Display task...

REM Delete existing task if it exists
schtasks /delete /tn "SO8T_Desktop_Progress_Display" /f 2>nul

REM Create new task
schtasks /create /tn "SO8T_Desktop_Progress_Display" ^
    /tr "powershell.exe -ExecutionPolicy Bypass -File \"C:\Users\downl\Desktop\SO8T\scripts\utils\show_progress_report.ps1\"" ^
    /sc onlogon ^
    /ru "%USERNAME%" ^
    /rl highest ^
    /f

if %errorlevel% equ 0 (
    echo [SUCCESS] SO8T Desktop Progress Display task created successfully!
    echo The progress report will now display on desktop at every Windows logon.
) else (
    echo [ERROR] Failed to create task. Error code: %errorlevel%
    echo Make sure you run this batch file as Administrator.
)

pause
