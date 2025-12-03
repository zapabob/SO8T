@echo off
:: ==========================================
:: AEGIS Autonomous Training Pipeline
:: 電源投入時に自動起動するメインスクリプト
:: ==========================================

:: ログ開始
echo. >> auto_training.log
echo [%DATE% %TIME%] Starting AEGIS Autonomous Training >> auto_training.log

:: カレントディレクトリ設定
cd /d "C:\Users\downl\Desktop\SO8T"

:: Python仮想環境アクティベート（必要に応じて）
:: call venv\Scripts\activate.bat

:: SO8Tサンシャイン実験実行
echo [%DATE% %TIME%] Launching Sunshine SO8T Experiment... >> auto_training.log
py -3 scripts/pipeline/sunshine_pipeline.py so8t >> auto_training.log 2>&1

:: 終了コードチェック
if %ERRORLEVEL% EQU 0 (
    echo [%DATE% %TIME%] Training completed successfully >> auto_training.log
) else (
    echo [%DATE% %TIME%] Training failed with error code %ERRORLEVEL% >> auto_training.log
    echo [%DATE% %TIME%] Will retry on next boot... >> auto_training.log
)

:: 終了（自動再開はWindowsのスタートアップが担う）
echo [%DATE% %TIME%] AEGIS session ended >> auto_training.log
pause
