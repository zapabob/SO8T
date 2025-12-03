@echo off
:: ==========================================
:: AEGIS A/Bテスト完全自律パイプライン
:: 電源投入時に自動起動し、すべてのフェーズに3分毎ローリングチェックポイント適用
:: ==========================================

:: システム監視デーモン起動（バックグラウンド）
echo [%DATE% %TIME%] Starting system monitor daemon...
start /B py -3 "%~dp0scripts\utils\system_monitor.py" --daemon

:: ログ開始
echo. >> ab_test_automation.log
echo [%DATE% %TIME%] ===== AEGIS A/B TEST AUTOMATION START ===== >> ab_test_automation.log
echo [%DATE% %TIME%] System monitor daemon started >> ab_test_automation.log

:: カレントディレクトリ設定
cd /d "C:\Users\downl\Desktop\SO8T"

:: ==========================================
:: フェーズ0: 環境チェック
:: ==========================================
echo [%DATE% %TIME%] Phase 0: Environment Check >> ab_test_automation.log
py -3 simple_rlpo_test.py >> ab_test_automation.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%DATE% %TIME%] Environment check failed >> ab_test_automation.log
    goto :error
)
echo [%DATE% %TIME%] Environment OK >> ab_test_automation.log

:: ==========================================
:: フェーズ1: AEGIS高品質データセット作成
:: ==========================================
echo [%DATE% %TIME%] Phase 1: AEGIS High-Quality Dataset Creation >> ab_test_automation.log
py -3 "%~dp0scripts\data\create_aegis_high_quality_dataset.py" >> ab_test_automation.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%DATE% %TIME%] Dataset creation failed >> ab_test_automation.log
    goto :error
)
echo [%DATE% %TIME%] AEGIS dataset created >> ab_test_automation.log

:: ==========================================
:: フェーズ2: lm-eval-harnessとELYZA-100セットアップ
:: ==========================================
echo [%DATE% %TIME%] Phase 2: lm-eval and ELYZA-100 Setup >> ab_test_automation.log
py -3 "%~dp0scripts\evaluation\setup_lm_eval_elyza.py" >> ab_test_automation.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%DATE% %TIME%] lm-eval setup failed >> ab_test_automation.log
    goto :error
)
echo [%DATE% %TIME%] lm-eval and ELYZA-100 setup completed >> ab_test_automation.log

:: ==========================================
:: フェーズ3: AEGIS RLPO学習（3分毎チェックポイント）
:: ==========================================
echo [%DATE% %TIME%] Phase 3: AEGIS RLPO Training with 3min Rolling Checkpoints >> ab_test_automation.log
echo Starting AEGIS RLPO training with NKAT SO(8) theory... >> ab_test_automation.log

py -3 scripts/training/rlpo_science_nsfw_automated.py --max_steps 10000 --checkpoint_interval 180 >> ab_test_automation.log 2>&1

:: RLPOの終了コードに関わらず、次のフェーズへ
echo [%DATE% %TIME%] AEGIS RLPO training session completed >> ab_test_automation.log

:: ==========================================
:: フェーズ4: ベースラインGGUF変換
:: ==========================================
echo [%DATE% %TIME%] Phase 4: Baseline GGUF Conversion >> ab_test_automation.log

:: ベースラインGGUF変換（BF16）
if exist "models\phi-3.5-mini-instruct" (
    echo Converting baseline model to BF16 GGUF... >> ab_test_automation.log
    py -3 scripts/conversion/convert_hf_to_gguf.py models\phi-3.5-mini-instruct --outfile D:/webdataset/gguf_models/baseline_phi35_bf16/baseline_phi35_bf16.gguf --outtype bf16 >> ab_test_automation.log 2>&1
    echo [%DATE% %TIME%] Baseline GGUF conversion completed >> ab_test_automation.log
) else (
    echo No baseline model found, skipping GGUF conversion >> ab_test_automation.log
)

:: ==========================================
:: フェーズ5: AEGIS GGUF変換
:: ==========================================
echo [%DATE% %TIME%] Phase 5: AEGIS GGUF Conversion >> ab_test_automation.log

:: AEGIS学習済みモデルが存在する場合、GGUF変換を実行
if exist "checkpoints\rlpo_science_nsfw_automated\final_model" (
    echo Converting AEGIS trained model to GGUF format... >> ab_test_automation.log

    :: Q8_0変換
    echo [%DATE% %TIME%] Converting to Q8_0 quantization... >> ab_test_automation.log
    py -3 scripts/utils/task_manager.py gguf --model_path "checkpoints\rlpo_science_nsfw_automated\final_model" --quantization q8_0 --output_file "D:/webdataset/gguf_models/aegis_phi35_so8t/aegis_phi35_so8t_Q8_0.gguf" >> ab_test_automation.log 2>&1

    :: Q4_K_M変換（追加）
    echo [%DATE% %TIME%] Converting to Q4_K_M quantization... >> ab_test_automation.log
    py -3 scripts/utils/task_manager.py gguf --model_path "checkpoints\rlpo_science_nsfw_automated\final_model" --quantization q4_k_m --output_file "D:/webdataset/gguf_models/aegis_phi35_so8t/aegis_phi35_so8t_Q4_K_M.gguf" >> ab_test_automation.log 2>&1

    echo [%DATE% %TIME%] AEGIS GGUF conversion completed >> ab_test_automation.log
) else (
    echo No AEGIS trained model found, skipping GGUF conversion >> ab_test_automation.log
)

:: ==========================================
:: フェーズ6: A/Bテスト実行（llama.cpp.python）
:: ==========================================
echo [%DATE% %TIME%] Phase 6: A/B Test Execution with llama.cpp >> ab_test_automation.log
echo Running comprehensive A/B test evaluation... >> ab_test_automation.log

py -3 scripts/evaluation/run_llama_cpp_ab_test.py >> ab_test_automation.log 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo [%DATE% %TIME%] A/B test execution failed >> ab_test_automation.log
    goto :error
)
echo [%DATE% %TIME%] A/B test completed >> ab_test_automation.log

:: ==========================================
:: フェーズ7: 統計解析（ANOVA、効果量、p値）
:: ==========================================
echo [%DATE% %TIME%] Phase 7: Statistical Analysis >> ab_test_automation.log
echo Performing ANOVA, effect size, and p-value analysis... >> ab_test_automation.log

py -3 scripts/evaluation/analyze_ab_test_stats.py >> ab_test_automation.log 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo [%DATE% %TIME%] Statistical analysis failed >> ab_test_automation.log
    goto :error
)
echo [%DATE% %TIME%] Statistical analysis completed >> ab_test_automation.log

:: ==========================================
:: フェーズ8: HFアップロード準備
:: ==========================================
echo [%DATE% %TIME%] Phase 8: HF Upload Preparation >> ab_test_automation.log
echo Preparing complete A/B test results for HF upload... >> ab_test_automation.log

py -3 scripts/evaluation/prepare_hf_upload.py >> ab_test_automation.log 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo [%DATE% %TIME%] HF upload preparation failed >> ab_test_automation.log
    goto :error
)
echo [%DATE% %TIME%] HF upload package ready >> ab_test_automation.log

:: ==========================================
:: フェーズ9: 完了処理と自動終了
:: ==========================================
echo [%DATE% %TIME%] ===== AEGIS A/B TEST AUTOMATION COMPLETED ===== >> ab_test_automation.log
echo [%DATE% %TIME%] All checkpoints saved (3min rolling stock throughout all phases) >> ab_test_automation.log
echo [%DATE% %TIME%] A/B test results ready for HF upload >> ab_test_automation.log
echo [%DATE% %TIME%] Upload package location: hf_upload_package/ >> ab_test_automation.log

:: システム状態レポート生成
echo [%DATE% %TIME%] Generating final system status report... >> ab_test_automation.log
py -3 scripts/utils/system_monitor.py --status >> system_status_final.log 2>&1

:: 完了音を鳴らす（3回）
powershell -c "[console]::beep(1000,300); [console]::beep(1200,300); [console]::beep(1500,500); [console]::beep(1200,300); [console]::beep(1500,500); [console]::beep(1800,700)"

:: ==========================================
:: 自動終了処理（タスクキル＆自動起動削除）
:: ==========================================
echo [%DATE% %TIME%] Initiating automatic completion cleanup... >> ab_test_automation.log

:: Windowsタスクスケジューラから自動起動タスクを削除
schtasks /delete /tn "AEGIS_Autonomous_Pipeline" /f >> ab_test_automation.log 2>&1
echo [%DATE% %TIME%] Removed Windows Task Scheduler entry >> ab_test_automation.log

:: スタートアップショートカットを削除
set STARTUP_DIR=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup
set SHORTCUT_FILE=%STARTUP_DIR%\AEGIS_Autonomous_Launch.lnk
if exist "%SHORTCUT_FILE%" (
    del "%SHORTCUT_FILE%" >> ab_test_automation.log 2>&1
    echo [%DATE% %TIME%] Removed startup shortcut >> ab_test_automation.log
)

:: システム監視デーモンを終了
taskkill /f /im python.exe /fi "WINDOWTITLE eq AEGIS System Monitor*" >> ab_test_automation.log 2>&1
echo [%DATE% %TIME%] Terminated system monitor daemon >> ab_test_automation.log

:: 完了ログ
echo [%DATE% %TIME%] ===== AUTOMATION CLEANUP COMPLETED ===== >> ab_test_automation.log
echo [%DATE% %TIME%] AEGIS A/B test automation fully completed and cleaned up >> ab_test_automation.log

:: 最終完了音
powershell -c "[console]::beep(800,500); [console]::beep(1000,500); [console]::beep(1200,700)"

goto :end

:error
echo [%DATE% %TIME%] ===== ERROR OCCURRED ===== >> ab_test_automation.log
echo [%DATE% %TIME%] Check logs above for details >> ab_test_automation.log

:: エラー音を鳴らす
powershell -c "[console]::beep(400,500); [console]::beep(300,500); [console]::beep(400,500)"

:: エラー時は少し待ってから終了（デバッグ用）
timeout /t 10 /nobreak >nul

:end
echo [%DATE% %TIME%] AEGIS A/B test automation session ended >> ab_test_automation.log
