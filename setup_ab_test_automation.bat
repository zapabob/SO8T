@echo off
:: ==========================================
:: AEGIS A/Bテスト完全自動化システムセットアップ
:: 電源投入時自動起動 + 3分毎ローリングチェックポイント + 完了時自動終了
:: ==========================================

:: 管理者権限チェック
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [ADMIN] Administrator privileges confirmed
) else (
    echo [ERROR] Administrator privileges required!
    echo Please run this script as administrator.
    echo Right-click the script ^> "Run as administrator"
    pause
    exit /b 1
)

echo.
echo ========================================
echo AEGIS A/B Test Automation Setup
echo ========================================
echo.

:: システムテスト
echo [STEP 1/5] Running system tests...
py -3 simple_rlpo_test.py >nul 2>&1
if %errorLevel% neq 0 (
    echo ❌ System test failed. Please check Python environment.
    pause
    exit /b 1
)
echo ✅ System tests passed!

:: ==========================================
:: ステップ2: Windowsタスクスケジューラ設定
:: ==========================================
echo [STEP 2/5] Setting up Windows Task Scheduler...

:: タスク作成（電源投入時 + 毎日午前2時）
schtasks /create /tn "AEGIS_AB_Test_Automation" /tr "cmd.exe /c cd /d \"C:\Users\%USERNAME%\Desktop\SO8T\" && auto_ab_test_pipeline.bat" /sc onlogon /rl highest /f >nul 2>&1

:: 毎日午前2時の定期実行も追加
schtasks /create /tn "AEGIS_AB_Test_Daily" /tr "cmd.exe /c cd /d \"C:\Users\%USERNAME%\Desktop\SO8T\" && auto_ab_test_pipeline.bat" /sc daily /st 02:00 /rl highest /f >nul 2>&1

echo ✅ Task Scheduler configured!

:: ==========================================
:: ステップ3: スタートアップショートカット作成
:: ==========================================
echo [STEP 3/5] Creating startup shortcut...

set STARTUP_DIR=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup
set SHORTCUT_FILE=%STARTUP_DIR%\AEGIS_AB_Test_Launch.lnk

:: PowerShellでショートカット作成
powershell -Command "& {$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%SHORTCUT_FILE%'); $s.TargetPath = 'cmd.exe'; $s.Arguments = '/c cd /d \"C:\Users\%USERNAME%\Desktop\SO8T\" && auto_ab_test_pipeline.bat'; $s.WorkingDirectory = 'C:\Users\%USERNAME%\Desktop\SO8T'; $s.WindowStyle = 7; $s.Description = 'AEGIS A/B Test Automation Launcher'; $s.Save()}"

echo ✅ Startup shortcut created!

:: ==========================================
:: ステップ4: システム監視デーモン設定
:: ==========================================
echo [STEP 4/5] Configuring system monitor daemon...

:: 監視デーモンをスタートアップに追加（絶対パス使用で安全に）
set MONITOR_SCRIPT=py -3 "C:\Users\%USERNAME%\Desktop\SO8T\scripts\utils\system_monitor.py" --daemon
set MONITOR_LINK=%STARTUP_DIR%\AEGIS_AB_Test_Monitor.lnk

powershell -Command "& {$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%MONITOR_LINK%'); $s.TargetPath = 'cmd.exe'; $s.Arguments = '/c cd /d \"C:\Users\%USERNAME%\Desktop\SO8T\" && %MONITOR_SCRIPT%'; $s.WorkingDirectory = 'C:\Users\%USERNAME%\Desktop\SO8T'; $s.WindowStyle = 7; $s.Description = 'AEGIS A/B Test System Monitor'; $s.Save()}"

echo ✅ System monitor daemon configured!

:: ==========================================
:: ステップ5: 最終検証とドキュメント
:: ==========================================
echo [STEP 5/5] Final verification...

:: 必要なファイル存在確認
if not exist "auto_ab_test_pipeline.bat" (
    echo ❌ Main pipeline script not found!
    goto :error
)

if not exist "scripts\data\create_aegis_high_quality_dataset.py" (
    echo ❌ AEGIS dataset creator not found!
    goto :error
)

if not exist "scripts\evaluation\setup_lm_eval_elyza.py" (
    echo ❌ lm-eval setup script not found!
    goto :error
)

if not exist "scripts\evaluation\run_llama_cpp_ab_test.py" (
    echo ❌ A/B test runner not found!
    goto :error
)

if not exist "scripts\evaluation\analyze_ab_test_stats.py" (
    echo ❌ Statistical analyzer not found!
    goto :error
)

if not exist "scripts\evaluation\prepare_hf_upload.py" (
    echo ❌ HF upload preparer not found!
    goto :error
)

:: 設定確認
echo Checking Task Scheduler...
schtasks /query /tn "AEGIS_AB_Test_Automation" >nul 2>&1
if %errorLevel% neq 0 (
    echo ❌ Task Scheduler setup failed!
    goto :error
)

echo Checking startup shortcuts...
if not exist "%SHORTCUT_FILE%" (
    echo ❌ Startup shortcut creation failed!
    goto :error
)

echo ✅ All components verified!

:: セットアップ完了ログ
echo. >> ab_test_setup.log
echo [%DATE% %TIME%] ===== AEGIS A/B TEST AUTOMATION SETUP COMPLETED ===== >> ab_test_setup.log
echo [%DATE% %TIME%] Task Scheduler: AEGIS_AB_Test_Automation >> ab_test_setup.log
echo [%DATE% %TIME%] Daily Schedule: AEGIS_AB_Test_Daily (02:00) >> ab_test_setup.log
echo [%DATE% %TIME%] Startup: %SHORTCUT_FILE% >> ab_test_setup.log
echo [%DATE% %TIME%] Monitor: %MONITOR_LINK% >> ab_test_setup.log
echo [%DATE% %TIME%] Pipeline: auto_ab_test_pipeline.bat >> ab_test_setup.log
echo [%DATE% %TIME%] Checkpoint Interval: 3 minutes (rolling stock: 5) >> ab_test_setup.log

:: 完了メッセージ
echo.
echo ========================================
echo 🎉 SETUP COMPLETED SUCCESSFULLY!
echo ========================================
echo.
echo 📋 What happens next:
echo    • On next system boot/login: A/B test automation starts automatically
echo    • Daily at 02:00: Additional automation run
echo    • Every 3 minutes: Rolling checkpoints saved
echo    • On completion: Automatic cleanup and HF upload preparation
echo.
echo 📊 Expected completion time: 12-24 hours (depending on hardware)
echo.
echo 📁 Results will be saved to:
echo    • results/ab_test_results/ (evaluation data)
echo    • hf_upload_package/ (complete HF upload package)
echo.
echo 🚀 Ready for fully autonomous A/B testing!
echo.

:: 完了音
powershell -c "[console]::beep(1000,300); [console]::beep(1200,300); [console]::beep(1500,500)"

goto :end

:error
echo.
echo ========================================
echo ❌ SETUP FAILED!
echo ========================================
echo.
echo 🔧 Troubleshooting:
echo    • Check Python environment: py -3 --version
echo    • Verify all required files exist
echo    • Run as administrator
echo    • Check Windows Task Scheduler permissions
echo.
echo 📞 For support, check logs in ab_test_setup.log
echo.

:: エラー音
powershell -c "[console]::beep(400,500); [console]::beep(300,500)"

pause
exit /b 1

:end
echo [%DATE% %TIME%] AEGIS A/B test automation setup session ended >> ab_test_setup.log
