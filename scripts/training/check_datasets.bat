@echo off
REM データセット存在確認スクリプト

chcp 65001 >nul
echo [CHECK] Checking datasets...
echo =====================================================================
echo.

REM 四重推論形式データセットの確認
echo [CHECK 1] Quadruple thinking datasets...
powershell -Command "$files = Get-ChildItem 'D:\webdataset\processed\thinking_quadruple\quadruple_thinking_*.jsonl' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($files) { Write-Host '[OK] Found:' -ForegroundColor Green; Write-Host $files.FullName -ForegroundColor Cyan; $count = (Get-Content $files.FullName | Measure-Object -Line).Lines; Write-Host \"Samples: $count\" -ForegroundColor Cyan } else { Write-Host '[WARNING] Not found' -ForegroundColor Yellow }"
echo.

REM thinking_sftデータセットの確認
echo [CHECK 2] thinking_sft dataset...
powershell -Command "if (Test-Path 'D:\webdataset\processed\thinking_sft\thinking_sft_dataset.jsonl') { $file = Get-Item 'D:\webdataset\processed\thinking_sft\thinking_sft_dataset.jsonl'; Write-Host '[OK] Found:' -ForegroundColor Green; Write-Host $file.FullName -ForegroundColor Cyan; $count = (Get-Content $file.FullName | Measure-Object -Line).Lines; Write-Host \"Samples: $count\" -ForegroundColor Cyan } else { Write-Host '[WARNING] Not found' -ForegroundColor Yellow }"
echo.

echo =====================================================================
echo [INFO] Dataset check completed
echo.







REM データセット存在確認スクリプト

chcp 65001 >nul
echo [CHECK] Checking datasets...
echo =====================================================================
echo.

REM 四重推論形式データセットの確認
echo [CHECK 1] Quadruple thinking datasets...
powershell -Command "$files = Get-ChildItem 'D:\webdataset\processed\thinking_quadruple\quadruple_thinking_*.jsonl' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($files) { Write-Host '[OK] Found:' -ForegroundColor Green; Write-Host $files.FullName -ForegroundColor Cyan; $count = (Get-Content $files.FullName | Measure-Object -Line).Lines; Write-Host \"Samples: $count\" -ForegroundColor Cyan } else { Write-Host '[WARNING] Not found' -ForegroundColor Yellow }"
echo.

REM thinking_sftデータセットの確認
echo [CHECK 2] thinking_sft dataset...
powershell -Command "if (Test-Path 'D:\webdataset\processed\thinking_sft\thinking_sft_dataset.jsonl') { $file = Get-Item 'D:\webdataset\processed\thinking_sft\thinking_sft_dataset.jsonl'; Write-Host '[OK] Found:' -ForegroundColor Green; Write-Host $file.FullName -ForegroundColor Cyan; $count = (Get-Content $file.FullName | Measure-Object -Line).Lines; Write-Host \"Samples: $count\" -ForegroundColor Cyan } else { Write-Host '[WARNING] Not found' -ForegroundColor Yellow }"
echo.

echo =====================================================================
echo [INFO] Dataset check completed
echo.







REM データセット存在確認スクリプト

chcp 65001 >nul
echo [CHECK] Checking datasets...
echo =====================================================================
echo.

REM 四重推論形式データセットの確認
echo [CHECK 1] Quadruple thinking datasets...
powershell -Command "$files = Get-ChildItem 'D:\webdataset\processed\thinking_quadruple\quadruple_thinking_*.jsonl' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($files) { Write-Host '[OK] Found:' -ForegroundColor Green; Write-Host $files.FullName -ForegroundColor Cyan; $count = (Get-Content $files.FullName | Measure-Object -Line).Lines; Write-Host \"Samples: $count\" -ForegroundColor Cyan } else { Write-Host '[WARNING] Not found' -ForegroundColor Yellow }"
echo.

REM thinking_sftデータセットの確認
echo [CHECK 2] thinking_sft dataset...
powershell -Command "if (Test-Path 'D:\webdataset\processed\thinking_sft\thinking_sft_dataset.jsonl') { $file = Get-Item 'D:\webdataset\processed\thinking_sft\thinking_sft_dataset.jsonl'; Write-Host '[OK] Found:' -ForegroundColor Green; Write-Host $file.FullName -ForegroundColor Cyan; $count = (Get-Content $file.FullName | Measure-Object -Line).Lines; Write-Host \"Samples: $count\" -ForegroundColor Cyan } else { Write-Host '[WARNING] Not found' -ForegroundColor Yellow }"
echo.

echo =====================================================================
echo [INFO] Dataset check completed
echo.


































































































