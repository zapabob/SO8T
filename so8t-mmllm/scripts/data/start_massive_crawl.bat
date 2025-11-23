@echo off
chcp 65001 >nul
echo ========================================
echo SO8T MASSIVE PARALLEL WEB CRAWLER
echo ========================================
echo.
echo [CONFIG]
echo   Target: 17,000,000 samples (~200 GB raw data)
echo   Workers: 16 parallel processes
echo   Languages: JA(70%%) EN(20%%) ZH(10%%)
echo   Sources: NDL, Wikipedia, Gov, eGov, Nikkei225, NSFW sources, etc.
echo   Output: D:\webdataset
echo.
echo [ESTIMATE]
echo   Duration: 50-80 hours (2-4 days)
echo   Raw data: ~200 GB
echo   Cleaned data: ~100 GB (after pipeline)
echo   Disk space required: 400 GB+ (D: drive)
echo.
echo [WARNING]
echo   This is a LONG-RUNNING background task
echo   Make sure you have:
echo   - Stable internet connection
echo   - Sufficient disk space (150GB+)
echo   - Power supply (UPS recommended)
echo.
pause

echo.
echo [START] Launching massive crawl in background...
echo   Output directory: D:\webdataset
echo.

REM Dドライブに移動して出力ディレクトリ作成
D:
mkdir D:\webdataset 2>nul
cd C:\Users\downl\Desktop\SO8T\so8t-mmllm

REM バックグラウンド実行（200GB対応）
start /MIN "SO8T-MassiveCrawl" py -3 scripts\data\massive_parallel_crawler.py ^
  --target 17000000 ^
  --output-dir D:\webdataset ^
  --workers 16 ^
  --ja-weight 0.7 ^
  --en-weight 0.2 ^
  --zh-weight 0.1

timeout /t 5 /nobreak >nul

echo.
echo [OK] Massive crawl started in background!
echo.
echo [MONITOR]
echo   Log: logs\massive_crawl.log
echo   Progress: Get-Content logs\massive_crawl.log -Tail 30 -Wait
echo.
echo [STOP]
echo   Find process: Get-Process python
echo   Kill process: Stop-Process -Name python
echo.

REM 音声通知
if exist "C:\SO8T\so8t-mmllm\.cursor\marisa_owattaze.wav" (
    powershell -Command "(New-Object System.Media.SoundPlayer 'C\SO8T\so8t-mmllm\.cursor\marisa_owattaze.wav').PlaySync()"
    echo [AUDIO] Crawl started notification played
)

echo.
echo ========================================
echo MASSIVE CRAWL RUNNING IN BACKGROUND
echo ========================================
echo.
pause

