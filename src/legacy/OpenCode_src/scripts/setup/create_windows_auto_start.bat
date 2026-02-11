@echo off
REM AEGIS-v2.0 Windows自動起動スクリプト作成
REM Creates Windows auto-start script for AEGIS-v2.0 training

echo Creating AEGIS-v2.0 auto-start configuration...
echo.

REM 自動再開スクリプトが存在するか確認
if not exist "%~dp0..\training\auto_resume_aegis_v2.py" (
    echo Creating auto-resume script...
    python "%~dp0..\training\aegis_v2_training_pipeline.py" --create_resume_script
    echo.
)

REM スタートアップフォルダにショートカット作成
set STARTUP_FOLDER="%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set SCRIPT_PATH=%~dp0..\training\auto_resume_aegis_v2.py
set SHORTCUT_NAME=AEGIS-v2.0_Auto_Resume.lnk

echo Creating startup shortcut...
powershell -Command "
$WshShell = New-Object -comObject WScript.Shell;
$Shortcut = $WshShell.CreateShortcut('%STARTUP_FOLDER%\%SHORTCUT_NAME%');
$Shortcut.TargetPath = 'python.exe';
$Shortcut.Arguments = '%SCRIPT_PATH%';
$Shortcut.WorkingDirectory = '%~dp0..';
$Shortcut.IconLocation = 'python.exe,0';
$Shortcut.Description = 'AEGIS-v2.0 Auto Resume Training';
$Shortcut.Save();
"

echo.
echo Auto-start configuration created successfully!
echo.
echo The system will now automatically resume AEGIS-v2.0 training on power-on.
echo.
echo To disable auto-start, delete the shortcut from:
echo %STARTUP_FOLDER%\%SHORTCUT_NAME%
echo.
echo Press any key to continue...
pause >nul
