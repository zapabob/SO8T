@echo off
chcp 65001 >nul
echo ================================================================================
echo Chrome DevTools リモートデバッグ接続ガイド
echo ================================================================================
echo.
echo [INFO] 自分のChromeブラウザからリモートデバッグポートに接続する方法
echo.

REM リモートデバッグポートの確認
set "DEBUG_PORT=9222"
echo [INFO] リモートデバッグポート: %DEBUG_PORT%
echo.

echo [STEP 1] Chromeブラウザを起動します
echo [STEP 2] 以下のURLにアクセスしてください:
echo.
echo     chrome://inspect
echo.
echo [STEP 3] "Discover network targets" セクションで以下のポートを追加:
echo.
echo     localhost:%DEBUG_PORT%
echo.
echo [STEP 4] または、Chromeをリモートデバッグモードで起動:
echo.
echo     chrome.exe --remote-debugging-port=%DEBUG_PORT% --user-data-dir="%TEMP%\chrome_devtools"
echo.
echo [INFO] 接続後、DevToolsでスクレイピング中のブラウザを監視できます
echo.

REM Chromeをリモートデバッグモードで起動するオプション
set /p LAUNCH_CHROME="Chromeをリモートデバッグモードで起動しますか？ (y/n): "
if /i "%LAUNCH_CHROME%"=="y" (
    echo [INFO] Chromeをリモートデバッグモードで起動中...
    
    REM Chromeのパスを探す
    set "CHROME_PATH="
    if exist "%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe" (
        set "CHROME_PATH=%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"
    ) else if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" (
        set "CHROME_PATH=C:\Program Files\Google\Chrome\Application\chrome.exe"
    ) else if exist "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" (
        set "CHROME_PATH=C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
    )
    
    if defined CHROME_PATH (
        echo [INFO] Chromeパス: %CHROME_PATH%
        echo [INFO] リモートデバッグポート: %DEBUG_PORT%
        echo.
        echo [INFO] Chromeを起動します...
        start "" "%CHROME_PATH%" --remote-debugging-port=%DEBUG_PORT% --user-data-dir="%TEMP%\chrome_devtools_%DEBUG_PORT%"
        echo [OK] Chromeを起動しました
        echo.
        echo [INFO] 次に、別のChromeウィンドウで chrome://inspect にアクセスしてください
    ) else (
        echo [ERROR] Chromeが見つかりません
        echo [INFO] 手動でChromeを起動してください:
        echo     chrome.exe --remote-debugging-port=%DEBUG_PORT% --user-data-dir="%TEMP%\chrome_devtools"
    )
)

echo.
echo ================================================================================
echo [INFO] 接続方法まとめ
echo ================================================================================
echo.
echo 方法1: chrome://inspect を使用
echo   1. Chromeブラウザで chrome://inspect を開く
echo   2. "Configure..." をクリック
echo   3. "localhost:%DEBUG_PORT%" を追加
echo   4. 接続されたブラウザが表示されます
echo.
echo 方法2: 直接接続
echo   1. Chromeをリモートデバッグモードで起動
echo   2. 同じポート (%DEBUG_PORT%) を使用
echo   3. DevToolsで接続されたブラウザを監視
echo.
echo ================================================================================

pause










