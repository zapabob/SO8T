@echo off
REM 四値分類データセット存在確認スクリプト

chcp 65001 >nul
echo [CHECK] Checking four_class datasets...
echo =====================================================================
echo.

powershell -Command "$files = Get-ChildItem 'D:\webdataset\processed\four_class\four_class_*.jsonl' -ErrorAction SilentlyContinue; if ($files) { Write-Host '[OK] Found four_class datasets:' -ForegroundColor Green; $files | ForEach-Object { Write-Host $_.FullName -ForegroundColor Cyan; $count = (Get-Content $_.FullName | Measure-Object -Line).Lines; Write-Host \"Samples: $count\" -ForegroundColor Cyan } } else { Write-Host '[WARNING] No four_class datasets found' -ForegroundColor Yellow }"
echo.

echo =====================================================================
echo [INFO] Four_class dataset check completed
echo.







REM 四値分類データセット存在確認スクリプト

chcp 65001 >nul
echo [CHECK] Checking four_class datasets...
echo =====================================================================
echo.

powershell -Command "$files = Get-ChildItem 'D:\webdataset\processed\four_class\four_class_*.jsonl' -ErrorAction SilentlyContinue; if ($files) { Write-Host '[OK] Found four_class datasets:' -ForegroundColor Green; $files | ForEach-Object { Write-Host $_.FullName -ForegroundColor Cyan; $count = (Get-Content $_.FullName | Measure-Object -Line).Lines; Write-Host \"Samples: $count\" -ForegroundColor Cyan } } else { Write-Host '[WARNING] No four_class datasets found' -ForegroundColor Yellow }"
echo.

echo =====================================================================
echo [INFO] Four_class dataset check completed
echo.







REM 四値分類データセット存在確認スクリプト

chcp 65001 >nul
echo [CHECK] Checking four_class datasets...
echo =====================================================================
echo.

powershell -Command "$files = Get-ChildItem 'D:\webdataset\processed\four_class\four_class_*.jsonl' -ErrorAction SilentlyContinue; if ($files) { Write-Host '[OK] Found four_class datasets:' -ForegroundColor Green; $files | ForEach-Object { Write-Host $_.FullName -ForegroundColor Cyan; $count = (Get-Content $_.FullName | Measure-Object -Line).Lines; Write-Host \"Samples: $count\" -ForegroundColor Cyan } } else { Write-Host '[WARNING] No four_class datasets found' -ForegroundColor Yellow }"
echo.

echo =====================================================================
echo [INFO] Four_class dataset check completed
echo.


































































































