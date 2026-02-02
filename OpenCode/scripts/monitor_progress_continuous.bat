@echo off
chcp 65001 >nul
echo [SO8T] 連続プロジェクト進捗監視を開始します...
echo 監視間隔: 30秒
echo Ctrl+Cで停止できます
echo.

REM 連続監視モードで実行
powershell -ExecutionPolicy Bypass -File "scripts\utils\monitor_so8t_progress.ps1" -Continuous

echo.
echo [SO8T] 連続監視が停止しました
pause
