@echo off
:: ==========================================
:: AEGIS Auto-Launcher (Boot Hook)
:: 完全自律トレーニングシステム起動
:: ==========================================

:: 1. プロジェクトディレクトリに移動（絶対パス）
cd /d "C:\Users\downl\Desktop\SO8T"

:: 2. ログファイルに起動時刻を記録
echo. >> boot_history.log
echo [%DATE% %TIME%] ===== AEGIS BOOT LAUNCH ===== >> boot_history.log
echo [BOOT] Windows Started at %DATE% %TIME% >> boot_history.log
echo [BOOT] Initializing AEGIS Autonomous Training System... >> boot_history.log

:: 3. 少し待つ（Windowsのネットワークとかが安定するまで30秒待機）
echo [BOOT] Waiting for system stabilization... >> boot_history.log
timeout /t 30 /nobreak >nul

:: 4. システム状態確認
echo [BOOT] Checking system status... >> boot_history.log
py -3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')" >> boot_history.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [BOOT] System check failed, retrying in 60 seconds... >> boot_history.log
    timeout /t 60 /nobreak >nul
)

:: 5. メインのパイプラインを呼び出す
:: (新しいウィンドウを開いて実行、落ちてもログは残る)
echo [BOOT] Launching AEGIS Autonomous Pipeline... >> boot_history.log
start "AEGIS-v2.0 Autonomous Training" cmd /k "auto_aegis_pipeline.bat"

:: 6. 起動成功を記録
echo [%DATE% %TIME%] AEGIS Autonomous System Launched Successfully >> boot_history.log

:: ランチャー自体は終了（バックグラウンドでパイプラインが実行される）
exit
