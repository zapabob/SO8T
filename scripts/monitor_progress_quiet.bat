@echo off
chcp 65001 >nul

REM 静かなモードで進捗を表示（ログなし）
powershell -ExecutionPolicy Bypass -File "scripts\utils\monitor_so8t_progress.ps1" -Quiet

echo.
echo [AUDIO] 進捗確認完了通知を再生します...

REM 完了通知（静かに）
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"
