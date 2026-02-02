@echo off
chcp 65001 >nul
echo [CURSOR + MOONSHOT] 完全無人運用統合システム
echo ============================================
echo 開始時刻: %date% %time%
echo.

REM Cursor実行ファイルのパス確認
echo [CURSOR] 実行ファイル確認...
set "CURSOR_PATH="
if exist "%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe" set "CURSOR_PATH=%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe"
if exist "%ProgramFiles%\Microsoft VS Code\Code.exe" set "CURSOR_PATH=%ProgramFiles%\Microsoft VS Code\Code.exe"
if exist "%ProgramFiles(x86)%\Microsoft VS Code\Code.exe" set "CURSOR_PATH=%ProgramFiles(x86)%\Microsoft VS Code\Code.exe"
if exist "%LOCALAPPDATA%\Programs\cursor\Cursor.exe" set "CURSOR_PATH=%LOCALAPPDATA%\Programs\cursor\Cursor.exe"

if "%CURSOR_PATH%"=="" (
    echo ❌ Cursor実行ファイルが見つかりません
    echo Cursorがインストールされているか確認してください
    goto :error
)
echo ✅ Cursor実行ファイル: %CURSOR_PATH%
echo.

REM Cursor設定ファイル作成・更新
echo [SETTINGS] Cursor設定更新...
if not exist "%APPDATA%\Cursor" mkdir "%APPDATA%\Cursor" 2>nul
echo { > "%APPDATA%\Cursor\settings.json"
echo   "window.restoreWindows": "all", >> "%APPDATA%\Cursor\settings.json"
echo   "window.newWindowDimensions": "maximized", >> "%APPDATA%\Cursor\settings.json"
echo   "files.autoSave": "afterDelay", >> "%APPDATA%\Cursor\settings.json"
echo   "files.autoSaveDelay": 1000, >> "%APPDATA%\Cursor\settings.json"
echo   "terminal.integrated.shell.windows": "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe", >> "%APPDATA%\Cursor\settings.json"
echo   "terminal.integrated.shellArgs.windows": ["-NoExit", "-Command", "cd '%~dp0'"], >> "%APPDATA%\Cursor\settings.json"
echo   "extensions.autoUpdate": true, >> "%APPDATA%\Cursor\settings.json"
echo   "workbench.startupEditor": "none", >> "%APPDATA%\Cursor\settings.json"
echo   "window.openFilesInNewWindow": "off", >> "%APPDATA%\Cursor\settings.json"
echo   "workbench.editor.enablePreview": false >> "%APPDATA%\Cursor\settings.json"
echo } >> "%APPDATA%\Cursor\settings.json"
echo ✅ Cursor設定ファイル更新完了
echo.

REM Cursorワークスペース設定
echo [WORKSPACE] Cursorワークスペース設定...
if not exist ".cursor" mkdir ".cursor" 2>nul
echo { > ".cursor\workspace.code-workspace"
echo   "folders": [ >> ".cursor\workspace.code-workspace"
echo     { >> ".cursor\workspace.code-workspace"
echo       "path": "." >> ".cursor\workspace.code-workspace"
echo     } >> ".cursor\workspace.code-workspace"
echo   ], >> ".cursor\workspace.code-workspace"
echo   "settings": { >> ".cursor\workspace.code-workspace"
echo     "python.defaultInterpreterPath": "python", >> ".cursor\workspace.code-workspace"
echo     "terminal.integrated.cwd": "${workspaceFolder}", >> ".cursor\workspace.code-workspace"
echo     "files.autoSave": "afterDelay", >> ".cursor\workspace.code-workspace"
echo     "files.autoSaveDelay": 1000, >> ".cursor\workspace.code-workspace"
echo     "git.autofetch": true, >> ".cursor\workspace.code-workspace"
echo     "git.enableSmartCommit": true >> ".cursor\workspace.code-workspace"
echo   }, >> ".cursor\workspace.code-workspace"
echo   "launch": { >> ".cursor\workspace.code-workspace"
echo     "version": "0.2.0", >> ".cursor\workspace.code-workspace"
echo     "configurations": [ >> ".cursor\workspace.code-workspace"
echo       { >> ".cursor\workspace.code-workspace"
echo         "name": "MOONSHOT Unattended Training", >> ".cursor\workspace.code-workspace"
echo         "type": "python", >> ".cursor\workspace.code-workspace"
echo         "request": "launch", >> ".cursor\workspace.code-workspace"
echo         "program": "${workspaceFolder}/scripts/training/phi35_soul_weight_trainer.py", >> ".cursor\workspace.code-workspace"
echo         "console": "integratedTerminal", >> ".cursor\workspace.code-workspace"
echo         "cwd": "${workspaceFolder}" >> ".cursor\workspace.code-workspace"
echo       } >> ".cursor\workspace.code-workspace"
echo     ] >> ".cursor\workspace.code-workspace"
echo   } >> ".cursor\workspace.code-workspace"
echo } >> ".cursor\workspace.code-workspace"
echo ✅ Cursorワークスペース設定完了
echo.

REM Cursor起動
echo [LAUNCH] Cursor起動中...
start "" "%CURSOR_PATH%" "%~dp0.cursor\workspace.code-workspace"
echo ✅ Cursor起動完了
echo.

REM Cursor起動待機
echo [WAIT] Cursor起動待機中...
timeout /t 15 /nobreak >nul
echo ✅ Cursor起動待機完了
echo.

REM MOONSHOT無人運用開始
echo [MOONSHOT] 完全無人運用開始...
echo このシステムは完全に無人運用可能です。
echo 学習中に電源が切れても自動復旧します。
echo.

REM 無人運用スクリプト実行
call auto_moonshot_unattended.bat

if %errorlevel% equ 0 (
    echo ✅ MOONSHOT無人運用完了
    goto :success
) else (
    echo ❌ MOONSHOT無人運用失敗
    goto :error
)

:success
echo [SUCCESS] Cursor + MOONSHOT完全無人運用完了
echo ============================================
echo 完了時刻: %date% %time%
echo.
echo [INTEGRATION FEATURES]
echo ✅ Cursor自動起動
echo ✅ ワークスペース自動オープン
echo ✅ MOONSHOT無人運用連携
echo ✅ 3分間隔チェックポイント
echo ✅ 電源断自動復旧
echo ✅ エラー自動検知・修正

REM 完了通知
powershell -ExecutionPolicy Bypass -Command "Start-Job -ScriptBlock { try { & 'scripts\utils\play_audio_notification.ps1' } catch { Write-Host 'Audio notification failed but continuing...' } } | Wait-Job -Timeout 10 | Out-Null"

REM ログ保存
echo [%date% %time%] Cursor+MOONSHOT無人運用完了 >> cursor_moonshot_integration.log
goto :end

:error
echo [CRITICAL ERROR] Cursor + MOONSHOT統合失敗
echo ============================================
echo エラーレベル: %errorlevel%
echo 時刻: %date% %time%
echo.

REM エラー通知
powershell -ExecutionPolicy Bypass -Command "try { [System.Console]::Beep(800, 1000) } catch { Write-Host 'Beep failed but continuing...' }"

REM エラーログ保存
echo [%date% %time%] Cursor+MOONSHOT統合失敗 (ErrorLevel: %errorlevel%) >> cursor_moonshot_integration_error.log

echo [ERROR RECOVERY]
echo 1. Cursorが正常に起動しているか確認してください
echo 2. MOONSHOTのエラーログを確認してください: moonshot_unattended_error.log
echo 3. 必要に応じて手動修正を行ってください
echo 4. 再度 cursor_moonshot_unattended_launcher.bat を実行してください
echo.
exit /b 1

:end
echo [SYSTEM SHUTDOWN] 正常終了
exit /b 0
