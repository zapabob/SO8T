@echo off
chcp 65001 >nul
REM SO8T完全パイプライン実行スクリプト
REM 全Phase自動実行（データ収集→学習→評価→配備）

echo ========================================
echo [START] SO8T Full Pipeline Execution
echo ========================================
echo.

set START_TIME=%TIME%
set LOG_DIR=logs\pipeline_%DATE:~0,4%%DATE:~5,2%%DATE:~8,2%
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM =============================================
REM Phase 1: データ収集・生成（6-12時間）
REM =============================================
echo [PHASE 1] Data Collection and Generation
echo ========================================
echo.

echo [1.1] Collecting public Japanese data (100k samples)...
py -3 scripts\data\collect_japanese_data.py --target 100000 > "%LOG_DIR%\collect_data.log" 2>&1
if errorlevel 1 (
    echo [ERROR] Data collection failed
    goto ERROR_EXIT
)
echo [OK] Data collection completed
echo.

echo [1.2] Generating synthetic data (25k per domain)...
py -3 scripts\data\generate_synthetic_data.py --samples 25000 > "%LOG_DIR%\generate_synthetic.log" 2>&1
if errorlevel 1 (
    echo [ERROR] Synthetic generation failed
    goto ERROR_EXIT
)
echo [OK] Synthetic generation completed
echo.

echo [1.3] Generating multimodal synthetic data...
py -3 scripts\data\generate_multimodal_synthetic.py --samples 25000 > "%LOG_DIR%\generate_multimodal.log" 2>&1
if errorlevel 1 (
    echo [ERROR] Multimodal generation failed
    goto ERROR_EXIT
)
echo [OK] Multimodal generation completed
echo.

echo [1.4] Validating data quality...
py -3 scripts\data\validate_data_quality.py > "%LOG_DIR%\validate_data.log" 2>&1
if errorlevel 1 (
    echo [ERROR] Data validation failed
    goto ERROR_EXIT
)
echo [OK] Data validation completed
echo.

echo [PHASE 1] COMPLETED
echo ========================================
echo.

REM 音声通知
powershell -Command "if (Test-Path 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav') { [System.Media.SoundPlayer]::new('C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav').Play(); Start-Sleep -Seconds 2 }"

REM =============================================
REM Phase 2: 学習実行（33時間）
REM =============================================
echo [PHASE 2] Training Execution (33 hours)
echo ========================================
echo.

set /p TRAIN_NOW="Start training now? This will take ~33 hours (Y/N): "
if /i not "%TRAIN_NOW%"=="Y" (
    echo [SKIP] Training skipped. Run start_training.bat manually later.
    goto PHASE_3
)

echo [2.1] Starting training (background mode)...
cd scripts\training
start "SO8T Training" /MIN start_training.bat
cd ..\..
echo [OK] Training started in background
echo [INFO] Monitor: logs\training_*.log
echo.

echo [WAIT] Waiting for training to complete...
echo [INFO] This will take approximately 33 hours
echo [INFO] You can stop this script and training will continue
echo.

set /p WAIT_TRAIN="Wait for training to complete? (Y/N): "
if /i not "%WAIT_TRAIN%"=="Y" (
    echo [SKIP] Continuing to next phases (training runs in background)
    goto PHASE_3
)

:WAIT_LOOP
REM 学習完了チェック（outputs/so8t_ja_finetuned/final_model存在確認）
if exist "outputs\so8t_ja_finetuned\final_model\config.json" (
    echo [OK] Training completed!
    goto PHASE_3
)
echo [WAIT] Training in progress... (checking every 5 minutes)
timeout /t 300 /nobreak >nul
goto WAIT_LOOP

:PHASE_3
echo [PHASE 2] COMPLETED (or SKIPPED)
echo ========================================
echo.

powershell -Command "if (Test-Path 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav') { [System.Media.SoundPlayer]::new('C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav').Play(); Start-Sleep -Seconds 2 }"

REM =============================================
REM Phase 3: GGUF変換・Ollama配備
REM =============================================
echo [PHASE 3] GGUF Conversion and Ollama Deployment
echo ========================================
echo.

echo [3.1] Converting models to GGUF...
py -3 scripts\convert_to_gguf_full.py > "%LOG_DIR%\convert_gguf.log" 2>&1
if errorlevel 1 (
    echo [WARNING] GGUF conversion had errors, check log
) else (
    echo [OK] GGUF conversion completed
)
echo.

echo [PHASE 3] COMPLETED
echo ========================================
echo.

powershell -Command "if (Test-Path 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav') { [System.Media.SoundPlayer]::new('C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav').Play(); Start-Sleep -Seconds 2 }"

REM =============================================
REM Phase 4: モデル比較評価
REM =============================================
echo [PHASE 4] Model Comparison Evaluation
echo ========================================
echo.

echo [4.1] Running model comparison...
py -3 scripts\evaluation\compare_models.py > "%LOG_DIR%\compare_models.log" 2>&1
if errorlevel 1 (
    echo [WARNING] Model comparison had errors, check log
) else (
    echo [OK] Model comparison completed
)
echo.

echo [4.2] Running comprehensive evaluation...
py -3 scripts\evaluation\comprehensive_evaluation.py > "%LOG_DIR%\comprehensive_eval.log" 2>&1
if errorlevel 1 (
    echo [WARNING] Comprehensive evaluation had errors, check log
) else (
    echo [OK] Comprehensive evaluation completed
)
echo.

echo [PHASE 4] COMPLETED
echo ========================================
echo.

powershell -Command "if (Test-Path 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav') { [System.Media.SoundPlayer]::new('C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav').Play(); Start-Sleep -Seconds 2 }"

REM =============================================
REM Phase 5: Windows MCPサービス配備
REM =============================================
echo [PHASE 5] Windows MCP Service Deployment
echo ========================================
echo.

set /p INSTALL_SERVICE="Install Windows MCP service? (Y/N): "
if /i "%INSTALL_SERVICE%"=="Y" (
    echo [5.1] Installing Windows MCP service...
    powershell -ExecutionPolicy Bypass -File scripts\windows\install_service.ps1 -Install
    
    echo [5.2] Starting service...
    powershell -ExecutionPolicy Bypass -File scripts\windows\install_service.ps1 -Start
    
    echo [OK] Service deployed
) else (
    echo [SKIP] Service installation skipped
)
echo.

echo [PHASE 5] COMPLETED
echo ========================================
echo.

powershell -Command "if (Test-Path 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav') { [System.Media.SoundPlayer]::new('C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav').Play(); Start-Sleep -Seconds 2 }"

REM =============================================
REM 完了
REM =============================================
set END_TIME=%TIME%

echo.
echo ========================================
echo [OK] SO8T Full Pipeline COMPLETED!
echo ========================================
echo.
echo Start Time: %START_TIME%
echo End Time: %END_TIME%
echo.
echo [INFO] Generated Reports:
dir /b _docs\*.md
echo.
echo [INFO] Deployed Models:
ollama list 2>nul
echo.
echo [INFO] Service Status:
powershell -ExecutionPolicy Bypass -File scripts\windows\install_service.ps1 -Status 2>nul
echo.

REM 最終音声通知（3回）
powershell -Command "$player = [System.Media.SoundPlayer]::new('C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav'); for ($i=0; $i -lt 3; $i++) { $player.PlaySync(); Start-Sleep -Milliseconds 500 }"

echo [END] Pipeline completed successfully
pause
exit /b 0

:ERROR_EXIT
echo.
echo [ERROR] Pipeline failed. Check log files in %LOG_DIR%
echo.
pause
exit /b 1

