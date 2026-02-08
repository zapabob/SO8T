@echo off
REM Wikipediaクローラー バックグラウンド実行バッチファイル
REM 
REM Usage:
REM   scripts\data\run_wikipedia_crawler_background.bat [target_samples] [output_dir]

chcp 65001 >nul
setlocal enabledelayedexpansion

REM パラメータ設定
set TARGET=1000
set OUTPUT=D:\webdataset
set SEED=42

if not "%~1"=="" set TARGET=%~1
if not "%~2"=="" set OUTPUT=%~2

echo ========================================
echo Wikipedia Chromium Crawler
echo Background Execution
echo ========================================
echo Target samples per domain: %TARGET%
echo Output directory: %OUTPUT%
echo ========================================

REM プロジェクトルートに移動
cd /d "%~dp0\..\.."

REM ログディレクトリ作成
if not exist "logs" mkdir logs

REM 出力ディレクトリ作成
if not exist "%OUTPUT%" mkdir "%OUTPUT%"

REM ログファイル
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set logfile=logs\wikipedia_crawler_background_%datetime:~0,8%_%datetime:~8,6%.log
set errorlog=logs\wikipedia_crawler_background_errors_%datetime:~0,8%_%datetime:~8,6%.log

echo [INFO] Starting crawler in background...
echo [INFO] Log file: %logfile%
echo [INFO] Error log: %errorlog%

REM PowerShellでバックグラウンド実行
powershell -ExecutionPolicy Bypass -Command ^
    "$job = Start-Job -ScriptBlock { ^
        Set-Location '%CD%'; ^
        py -3 scripts\data\wikipedia_chromium_crawler.py --output '%OUTPUT%' --target %TARGET% --seed %SEED% ^
    } -Name 'WikipediaCrawler'; ^
    Write-Host '[OK] Crawler started in background (Job: ' $job.Name ')'; ^
    Write-Host '[INFO] Monitor with: Get-Job -Name WikipediaCrawler ^| Receive-Job'; ^
    $jobInfo = @{ ^
        job_name = $job.Name; ^
        job_id = $job.Id; ^
        start_time = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss'); ^
        target_samples = %TARGET%; ^
        output_dir = '%OUTPUT%' ^
    } ^| ConvertTo-Json; ^
    $jobInfo ^| Out-File -FilePath '%OUTPUT%\crawler_job_info.json' -Encoding UTF8"

echo.
echo [OK] Crawler started in background
echo.
echo To check status:
echo   powershell -Command "Get-Job -Name WikipediaCrawler"
echo.
echo To view output:
echo   powershell -Command "Get-Job -Name WikipediaCrawler ^| Receive-Job"
echo.
echo To stop:
echo   powershell -Command "Stop-Job -Name WikipediaCrawler; Remove-Job -Name WikipediaCrawler"
echo.

REM 音声通知
powershell -ExecutionPolicy Bypass -File .cursor\play_audio_enhanced.ps1





















