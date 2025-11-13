@echo off
chcp 65001 >nul
echo [INFO] Starting WSL2 and executing automatic installation...
echo [STEP 1] Restarting WSL2...
wsl --shutdown
timeout /t 3 /nobreak >nul

echo [STEP 2] Starting WSL2 installation...
wsl bash -c "cd /mnt/c/Users/downl/Desktop/SO8T && bash scripts/utils/setup/install_all_dependencies_wsl2.sh"

echo [INFO] Installation completed or failed. Check output above.
pause


