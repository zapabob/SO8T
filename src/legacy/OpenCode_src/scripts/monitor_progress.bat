@echo off
chcp 65001 >nul
echo [SO8T] プロジェクト進捗監視を開始します...
echo.

REM PowerShellスクリプトの実行権限確認と実行
powershell -ExecutionPolicy Bypass -File "scripts\utils\monitor_so8t_progress.ps1"

echo.
echo [SO8T] 進捗監視が完了しました
echo [AUDIO] 完了通知を再生します...

REM オーディオ通知
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

pause
