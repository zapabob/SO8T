@echo off
chcp 65001 >nul
echo [MONITOR] Starting training log monitor...
echo ========================================

REM 学習ログを監視し、完了後にパイプラインを再開
py -3 scripts\utils\monitor_training_and_resume_pipeline.py --interval 60 --auto-resume

echo [AUDIO] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

