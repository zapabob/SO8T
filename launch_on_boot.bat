@echo off
:: ==========================================
:: AEGIS Auto-Launcher (Boot Hook)
:: ==========================================

:: 1. プロジェクトディレクトリに移動（絶対パス）
cd /d "C:\Users\downl\Desktop\SO8T"

:: 2. ログファイルに起動時刻を記録
echo. >> boot_history.log
echo [BOOT] Windows Started at %DATE% %TIME% >> boot_history.log
echo [BOOT] Initializing AEGIS Pipeline... >> boot_history.log

:: 3. 少し待つ（Windowsのネットワークとかが安定するまで30秒待機）
timeout /t 30 /nobreak >nul

:: 4. メインのパイプラインを呼び出す
:: (新しいウィンドウを開いて実行、落ちてもログは残る)
start "AEGIS-v2.0 Auto-Training" cmd /k "auto_aegis_pipeline.bat"

:: ランチャー自体は終了
exit
