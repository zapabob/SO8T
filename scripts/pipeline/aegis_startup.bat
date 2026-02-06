@echo off
chcp 65001 >nul
REM AEGIS-v3.0 Power-on Auto-Start Launcher
REM Automatically placed in Windows Startup folder

cd /d C:\Users\downl\Desktop\SO8T

REM Set environment variables
set SO8T_BASE_MODEL=AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp
set SO8T_ARXIV_COUNT=50000
set SO8T_BIORXIV_COUNT=50000
set SO8T_SKIP_OLLAMA=1
set SO8T_CHECKPOINT_INTERVAL=300
set SO8T_CHECKPOINT_ROLLING=3

echo [AEGIS] Auto-start at %date% %time%
echo [AEGIS] Launching continuous pipeline with 5-min rolling checkpoints (3 gen)...

powershell -ExecutionPolicy Bypass -File scripts\pipeline\run_aegis_continuous.ps1
