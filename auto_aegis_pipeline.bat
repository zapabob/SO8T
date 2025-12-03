@echo off
:: ==========================================
:: AEGIS完全自律トレーニングパイプライン
:: 電源投入時に自動起動し、すべてのタスクにチェックポイント適用
:: ==========================================

:: ログ開始
echo. >> auto_training.log
echo [%DATE% %TIME%] ===== AEGIS AUTONOMOUS TRAINING START ===== >> auto_training.log

:: カレントディレクトリ設定
cd /d "C:\Users\downl\Desktop\SO8T"

:: ==========================================
:: フェーズ1: 環境チェック
:: ==========================================
echo [%DATE% %TIME%] Phase 1: Environment Check >> auto_training.log
py -3 simple_rlpo_test.py >> auto_training.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%DATE% %TIME%] Environment check failed >> auto_training.log
    goto :error
)
echo [%DATE% %TIME%] Environment OK >> auto_training.log

:: ==========================================
:: フェーズ2: データセット更新（チェックポイント対応）
:: ==========================================
echo [%DATE% %TIME%] Phase 2: Dataset Update >> auto_training.log
echo Checking for new datasets... >> auto_training.log

:: データセットの存在確認
if not exist "data\science_reasoning_dataset_final.jsonl" (
    echo [%DATE% %TIME%] Creating science dataset... >> auto_training.log
    py -3 create_science_dataset.py >> auto_training.log 2>&1
)

if not exist "data\nsfw_drug_detection\nsfw_drug_mixed_dataset.jsonl" (
    echo [%DATE% %TIME%] Creating NSFW drug dataset... >> auto_training.log
    py -3 scripts/data/create_nsfw_drug_dataset.py >> auto_training.log 2>&1
)

echo [%DATE% %TIME%] Dataset check completed >> auto_training.log

:: ==========================================
:: フェーズ3: RLPO学習（完全自律）
:: ==========================================
echo [%DATE% %TIME%] Phase 3: RLPO Training with Full Checkpointing >> auto_training.log
echo Starting RLPO training with 3min checkpoints and 5-rolling stock... >> auto_training.log

py -3 scripts/training/rlpo_science_nsfw_automated.py --max_steps 10000 >> auto_training.log 2>&1

:: RLPOの終了コードに関わらず、次のフェーズへ
echo [%DATE% %TIME%] RLPO training session completed >> auto_training.log

:: ==========================================
:: フェーズ4: GGUF変換実行
:: ==========================================
echo [%DATE% %TIME%] Phase 4: GGUF Conversion >> auto_training.log

:: 学習済みモデルが存在する場合、GGUF変換を実行
if exist "checkpoints\rlpo_science_nsfw_automated\final_model" (
    echo Converting trained model to GGUF format... >> auto_training.log

    :: Q8_0変換
    echo [%DATE% %TIME%] Converting to Q8_0 quantization... >> auto_training.log
    py -3 scripts/utils/task_manager.py gguf --model_path "checkpoints\rlpo_science_nsfw_automated\final_model" --quantization q8_0 --output_file "D:/webdataset/gguf_models/rlpo_science_nsfw/rlpo_science_nsfw_Q8_0.gguf" >> auto_training.log 2>&1

    :: Q4_K_M変換（追加）
    echo [%DATE% %TIME%] Converting to Q4_K_M quantization... >> auto_training.log
    py -3 scripts/utils/task_manager.py gguf --model_path "checkpoints\rlpo_science_nsfw_automated\final_model" --quantization q4_k_m --output_file "D:/webdataset/gguf_models/rlpo_science_nsfw/rlpo_science_nsfw_Q4_K_M.gguf" >> auto_training.log 2>&1

    echo [%DATE% %TIME%] GGUF conversion completed >> auto_training.log
) else (
    echo No trained model found, skipping GGUF conversion >> auto_training.log
)

:: ==========================================
:: フェーズ5: 評価実行
:: ==========================================
if exist "checkpoints\rlpo_science_nsfw_automated\final_model" (
    echo Running evaluation on trained model... >> auto_training.log
    py -3 run_rlpo_evaluation.py >> auto_training.log 2>&1
) else (
    echo No trained model found, skipping evaluation >> auto_training.log
)

:: ==========================================
:: フェーズ6: レポート生成
:: ==========================================
echo [%DATE% %TIME%] Phase 6: Report Generation >> auto_training.log
py -3 generate_training_report.py >> auto_training.log 2>&1

:: ==========================================
:: 完了処理
:: ==========================================
echo [%DATE% %TIME%] ===== AEGIS TRAINING CYCLE COMPLETED ===== >> auto_training.log
echo [%DATE% %TIME%] All checkpoints saved (RLPO: 3min intervals, GGUF: conversion progress) >> auto_training.log
echo [%DATE% %TIME%] Next cycle will start on next boot >> auto_training.log

:: 成功音を鳴らす
powershell -c "[console]::beep(800,300); [console]::beep(1000,300); [console]::beep(1200,500)"

goto :end

:error
echo [%DATE% %TIME%] ===== ERROR OCCURRED ===== >> auto_training.log
echo [%DATE% %TIME%] Check logs above for details >> auto_training.log

:: エラー音を鳴らす
powershell -c "[console]::beep(400,500); [console]::beep(300,500)"

:: エラー時は少し待ってから終了（デバッグ用）
timeout /t 10 /nobreak >nul

:end
echo [%DATE% %TIME%] AEGIS autonomous session ended >> auto_training.log
