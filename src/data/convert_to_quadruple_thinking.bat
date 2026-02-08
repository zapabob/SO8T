@echo off
REM 既存データセットを四重推論形式に変換するバッチスクリプト

chcp 65001 >nul
echo [CONVERT] Converting dataset to quadruple thinking format
echo ==========================================================
echo.

REM デフォルトの入力・出力パス
set INPUT_DIR=D:\webdataset\processed
set OUTPUT_DIR=D:\webdataset\processed\thinking_quadruple

REM 入力ファイルの検索パターン
set FOUR_CLASS_PATTERN=%INPUT_DIR%\four_class\four_class_*.jsonl
set DOMAIN_KNOWLEDGE_PATTERN=%INPUT_DIR%\domain_knowledge\domain_knowledge_*.jsonl

echo [INFO] Searching for input datasets...
echo.

REM four_classデータセットの処理
echo [STEP 1] Processing four_class datasets...
powershell -Command "$files = Get-ChildItem '%FOUR_CLASS_PATTERN%' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName; if ($files) { Write-Host '[OK] Found four_class datasets' -ForegroundColor Green; $files | ForEach-Object { Write-Host \"  - $_\" } } else { Write-Host '[WARNING] No four_class datasets found' -ForegroundColor Yellow }"

REM domain_knowledgeデータセットの処理
echo [STEP 2] Processing domain_knowledge datasets...
powershell -Command "$files = Get-ChildItem '%DOMAIN_KNOWLEDGE_PATTERN%' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName; if ($files) { Write-Host '[OK] Found domain_knowledge datasets' -ForegroundColor Green; $files | ForEach-Object { Write-Host \"  - $_\" } } else { Write-Host '[WARNING] No domain_knowledge datasets found' -ForegroundColor Yellow }"

echo.
echo [STEP 3] Converting datasets to quadruple thinking format...

REM 出力ディレクトリの作成
if not exist "%OUTPUT_DIR%" (
    mkdir "%OUTPUT_DIR%"
)

REM タイムスタンプ付き出力ファイル名
set TIMESTAMP=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set OUTPUT_FILE=%OUTPUT_DIR%\quadruple_thinking_%TIMESTAMP%.jsonl

REM 複数ファイルをマージして変換
echo [INFO] Merging and converting datasets...
powershell -Command "$fourClassFiles = Get-ChildItem '%FOUR_CLASS_PATTERN%' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName; $domainFiles = Get-ChildItem '%DOMAIN_KNOWLEDGE_PATTERN%' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName; $allFiles = @($fourClassFiles) + @($domainFiles); if ($allFiles.Count -gt 0) { $fileArgs = $allFiles -join ' '; py -3 scripts\data\create_quadruple_thinking_dataset.py --inputs $allFiles --output '%OUTPUT_FILE%' --validate; if ($LASTEXITCODE -eq 0) { Write-Host '[SUCCESS] Conversion completed' -ForegroundColor Green } else { Write-Host '[ERROR] Conversion failed' -ForegroundColor Red; exit 1 } } else { Write-Host '[ERROR] No input files found' -ForegroundColor Red; exit 1 }"

if errorlevel 1 (
    echo [ERROR] Conversion failed
    exit /b 1
)

echo.
echo [SUCCESS] Conversion completed successfully!
echo [INFO] Output file: %OUTPUT_FILE%
echo.
echo [AUDIO] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

