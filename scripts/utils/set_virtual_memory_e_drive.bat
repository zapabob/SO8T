@echo off
chcp 65001 >nul
echo ========================================
echo E Drive Page File Configuration (220GB)
echo ========================================
echo.
echo [INFO] Eドライブに220GBの仮想メモリを設定します
echo [WARNING] この操作には管理者権限が必要です
echo [WARNING] 管理者として実行してください
echo.
echo [INFO] PowerShellスクリプトを実行中...
powershell -ExecutionPolicy Bypass -File "%~dp0set_virtual_memory_e_drive.ps1"
echo.
echo [AUDIO] 完了通知音を再生します...
powershell -ExecutionPolicy Bypass -File "%~dp0play_audio_notification.ps1"
pause

