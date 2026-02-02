@echo off
:: ==========================================
:: AEGIS完全自動無人化システム フルセットアップ
:: Windowsタスクスケジューラ + システム監視 + 自動復旧
:: ==========================================

echo 🚀 AEGIS Full Automation Setup Starting...
echo ==========================================
echo Required: Administrator privileges for system configuration
echo ==========================================

:: 管理者権限チェック
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ❌ ERROR: Administrator privileges required!
    echo.
    echo 🔧 How to run with administrator privileges:
    echo    1. Right-click on Command Prompt or PowerShell
    echo    2. Select "Run as administrator"
    echo    3. Navigate to: cd /d "C:\Users\%USERNAME%\Desktop\SO8T"
    echo    4. Run: setup_full_automation.bat
    echo.
    echo Or use: powershell -Command "Start-Process cmd.exe -Verb RunAs -ArgumentList '/c cd /d C:\Users\%USERNAME%\Desktop\SO8T && setup_full_automation.bat'"
    echo.
    pause
    exit /b 1
)

:: プロジェクトディレクトリ設定
cd /d "C:\Users\%USERNAME%\Desktop\SO8T"

:: ==========================================
:: ステップ1: システムテスト
:: ==========================================
echo [STEP 1/5] Running system tests...
py -3 test_gguf_checkpoint.py
if %errorLevel% neq 0 (
    echo ❌ System test failed. Please check the errors above.
    echo.
    echo 🔧 Troubleshooting:
    echo    • Make sure Python 3 is installed and in PATH
    echo    • Try running: py -3 test_gguf_checkpoint.py manually
    echo    • Check Python installation: py --version
    echo.
    pause
    exit /b 1
)
echo ✅ System tests passed!

:: ==========================================
:: ステップ2: Windowsタスクスケジューラ登録
:: ==========================================
echo [STEP 2/5] Setting up Windows Task Scheduler...
powershell -ExecutionPolicy Bypass -File "setup_autonomous_system.ps1" -Install
if %errorLevel% neq 0 (
    echo ❌ Task scheduler setup failed.
    pause
    exit /b 1
)
echo ✅ Task scheduler configured!

:: ==========================================
:: ステップ3: スタートアップフック設定
:: ==========================================
echo [STEP 3/5] Setting up startup hooks...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"
echo ✅ Startup hooks configured!

:: ==========================================
:: ステップ4: システム監視デーモン設定
:: ==========================================
echo [STEP 4/5] Configuring system monitor daemon...

:: 監視デーモンをスタートアップに追加
set MONITOR_SCRIPT=py -3 scripts/utils/system_monitor.py --daemon
set STARTUP_DIR=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup
set MONITOR_LINK=%STARTUP_DIR%\AEGIS_System_Monitor.lnk

echo Creating monitor startup shortcut...
powershell -Command "& {$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%MONITOR_LINK%'); $s.TargetPath = 'cmd.exe'; $s.Arguments = '/c cd /d ""C:\Users\%USERNAME%\Desktop\SO8T"" && %MONITOR_SCRIPT%'; $s.WorkingDirectory = 'C:\Users\%USERNAME%\Desktop\SO8T'; $s.WindowStyle = 7; $s.Description = 'AEGIS System Monitor Daemon'; $s.Save()}"

echo ✅ System monitor daemon configured!

:: ==========================================
:: ステップ5: 最終テストと検証
:: ==========================================
echo [STEP 5/5] Final verification...

:: 設定されたタスクを確認
powershell -ExecutionPolicy Bypass -File "setup_autonomous_system.ps1" -Test
if %errorLevel% neq 0 (
    echo ❌ Final verification failed.
    pause
    exit /b 1
)

:: ==========================================
:: セットアップ完了
:: ==========================================
echo.
echo 🎉 AEGIS FULL AUTOMATION SETUP COMPLETED! 🎉
echo ============================================
echo.
echo What happens now:
echo   • System will automatically start training on boot
echo   • All processes have 3-minute checkpoint saving
echo   • System monitor runs continuously for auto-recovery
echo   • GGUF conversion includes progress tracking
echo   • Complete autonomous operation achieved
echo.
echo Manual controls:
echo   • Run training: .\auto_aegis_pipeline.bat
echo   • Check status: python scripts/utils/system_monitor.py --status
echo   • Stop monitor: Ctrl+C in monitor terminal
echo   • Uninstall: powershell -ExecutionPolicy Bypass -File "setup_autonomous_system.ps1" -Uninstall
echo.
echo Next boot will start autonomous training automatically!
echo.
powershell -c "[console]::beep(800,200); [console]::beep(1000,200); [console]::beep(1200,200); [console]::beep(1500,400)"

echo Press any key to exit...
pause >nul
