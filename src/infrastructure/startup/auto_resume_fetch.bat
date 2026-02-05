@echo off
REM == Auto-Resume Arxiv Citation Fetcher ==
REM このスクリプトをWindowsスタートアップに配置して電源投入時自動再開

cd /d "C:\Users\downl\Desktop\SO8T"

REM ログファイル設定
set LOG_DIR=logs
set LOG_FILE=%LOG_DIR%\arxiv_100k_fetch.log
set CHECKPOINT_FILE=data\sunset_pipeline\raw\arxiv_citations\arxiv_top_100k_2024-2026_checkpoint.json

REM チェックポイントが存在し、完了していない場合のみ実行
if exist "%CHECKPOINT_FILE%" (
    echo [%date% %time%] Auto-resume started >> %LOG_FILE%
    start "Arxiv Fetch" /MIN py -3.12 scripts\data_processing\citation_fetcher.py --source arxiv --max-papers 100000 --output data\sunset_pipeline\raw\arxiv_citations\arxiv_top_100k_2024-2026.jsonl --verbose
) else (
    echo [%date% %time%] No checkpoint found, skipping auto-resume >> %LOG_FILE%
)
