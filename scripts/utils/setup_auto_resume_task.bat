@echo off
chcp 65001 >nul
echo === 四値分類トレーニング自動復旧システム セットアップ ===

set SCRIPT_DIR=%~dp0
set AUTO_RESUME_SCRIPT=%SCRIPT_DIR%auto_resume_four_class_training.ps1

echo 自動復旧スクリプト: %AUTO_RESUME_SCRIPT%

REM PowerShellスクリプトの実行権限を確認
powershell -Command "Get-ExecutionPolicy" > temp_policy.txt 2>nul
set /p CURRENT_POLICY=<temp_policy.txt
del temp_policy.txt

if "%CURRENT_POLICY%"=="Restricted" (
    echo WARNING: PowerShell実行ポリシーがRestrictedです。
    echo 以下のコマンドで実行ポリシーを変更してください:
    echo powershell -ExecutionPolicy RemoteSigned -Command "Set-ExecutionPolicy RemoteSigned"
    echo.
    echo それとも続行しますか？ (Y/N)
    set /p CONTINUE=
    if /i not "!CONTINUE!"=="Y" exit /b 1
)

echo.
echo タスクスケジューラに自動復旧タスクを登録します...

REM タスクスケジューラに登録
schtasks /create /tn "SO8T Four Class Training Auto Resume" /tr "powershell -ExecutionPolicy Bypass -File \"%AUTO_RESUME_SCRIPT%\"" /sc onlogon /rl highest /f

if %errorlevel% equ 0 (
    echo SUCCESS: 自動復旧タスクが登録されました。
    echo システムログオン時に自動でトレーニングが再開されます。
) else (
    echo ERROR: タスク登録に失敗しました。
    echo 手動でタスクスケジューラを設定してください。
)

echo.
echo スタートアップフォルダにも配置します...

REM スタートアップフォルダにショートカットを作成
set STARTUP_FOLDER=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup
set SHORTCUT_NAME=SO8T_Four_Class_Auto_Resume.lnk

echo Set oWS = WScript.CreateObject("WScript.Shell") > CreateShortcut.vbs
echo sLinkFile = "%STARTUP_FOLDER%\%SHORTCUT_NAME%" >> CreateShortcut.vbs
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> CreateShortcut.vbs
echo oLink.TargetPath = "powershell.exe" >> CreateShortcut.vbs
echo oLink.Arguments = "-ExecutionPolicy Bypass -File ""%AUTO_RESUME_SCRIPT%""" >> CreateShortcut.vbs
echo oLink.WorkingDirectory = "%SCRIPT_DIR%" >> CreateShortcut.vbs
echo oLink.Description = "SO8T Four Class Training Auto Resume" >> CreateShortcut.vbs
echo oLink.IconLocation = "powershell.exe,0" >> CreateShortcut.vbs
echo oLink.Save >> CreateShortcut.vbs

cscript //nologo CreateShortcut.vbs
del CreateShortcut.vbs

if exist "%STARTUP_FOLDER%\%SHORTCUT_NAME%" (
    echo SUCCESS: スタートアップフォルダにショートカットが作成されました。
) else (
    echo WARNING: スタートアップショートカットの作成に失敗しました。
)

echo.
echo === セットアップ完了 ===
echo.
echo 自動復旧機能の設定:
echo - タスクスケジューラ: システムログオン時に実行
echo - スタートアップフォルダ: Windows起動時に実行
echo - チェック間隔: 60秒
echo - チェックポイント保存: 1458ステップごと
echo - ローリングストック: 最新5個
echo.
echo テスト実行:
echo powershell -ExecutionPolicy Bypass -File "%AUTO_RESUME_SCRIPT%" -NoAudio
echo.
pause
