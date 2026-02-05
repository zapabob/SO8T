@echo off
REM AEGIS-v3.0 Power-on Auto-Start Launcher
REM Place this in Windows Startup folder: shell:startup

cd /d C:\Users\downl\Desktop\SO8T
powershell -ExecutionPolicy Bypass -File scripts\pipeline\run_aegis_continuous.ps1
