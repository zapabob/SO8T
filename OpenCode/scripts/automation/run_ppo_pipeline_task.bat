@echo off
chcp 65001 >nul
echo [SO8T] Starting PPO Pipeline Task
echo ===============================

cd /d "C:\Users\downl\Desktop\SO8T"

echo [INFO] Setting WEBDATASET_PATH environment variable...
set WEBDATASET_PATH=H:\from_D\webdataset

echo [INFO] Running PPO Pipeline...
"C:\Windows\py.exe" "C:\Users\downl\Desktop\SO8T\scripts\automation\complete_ppo_pipeline_with_power_on_automation.py"

if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] PPO Pipeline completed successfully
) else (
    echo [ERROR] PPO Pipeline failed with error code %ERRORLEVEL%
)

echo [DONE] Task execution completed
