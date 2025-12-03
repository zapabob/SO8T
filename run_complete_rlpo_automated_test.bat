@echo off
:: ==========================================
:: RLPO学習完全自動化テスト実行スクリプト
:: 科学・数学SFT + NKAT理論 + 薬物NSFWデータ
:: ==========================================

echo 🚀 Starting Complete RLPO Automated Test Suite...
echo ===================================================

:: ログファイル設定
set LOG_FILE=test_results\rlpo_complete_test_%DATE:~-4,4%%DATE:~-10,2%%DATE:~-7,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%.log
set LOG_FILE=%LOG_FILE: =0%

:: ディレクトリ作成
if not exist test_results mkdir test_results

:: ログ開始
echo [%DATE% %TIME%] Starting RLPO Complete Automated Test > %LOG_FILE%
echo [%DATE% %TIME%] Test Suite: Science + NSFW + NKAT Integration >> %LOG_FILE%

:: 1. 環境チェック
echo [STEP 1/5] Environment Check...
echo [%DATE% %TIME%] Step 1: Environment Check >> %LOG_FILE%
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')" >> %LOG_FILE% 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Environment check failed!
    echo [%DATE% %TIME%] FAILED: Environment check >> %LOG_FILE%
    goto :error
)
echo ✅ Environment OK
echo [%DATE% %TIME%] PASSED: Environment check >> %LOG_FILE%

:: 2. データセットチェック
echo [STEP 2/5] Dataset Validation...
echo [%DATE% %TIME%] Step 2: Dataset Validation >> %LOG_FILE%
python -c "
import os
from pathlib import Path
science = Path('data/science_reasoning_dataset_final.jsonl')
nsfw = Path('data/nsfw_drug_detection/nsfw_drug_mixed_dataset.jsonl')
print(f'Science dataset: {science.exists()} ({sum(1 for _ in open(science)) if science.exists() else 0} lines)')
print(f'NSFW dataset: {nsfw.exists()} ({sum(1 for _ in open(nsfw)) if nsfw.exists() else 0} lines)')
" >> %LOG_FILE% 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Dataset check failed!
    echo [%DATE% %TIME%] FAILED: Dataset check >> %LOG_FILE%
    goto :error
)
echo ✅ Datasets OK
echo [%DATE% %TIME%] PASSED: Dataset check >> %LOG_FILE%

:: 3. NKAT統合テスト
echo [STEP 3/5] NKAT Integration Test...
echo [%DATE% %TIME%] Step 3: NKAT Integration Test >> %LOG_FILE%
python -c "
from scripts.models.so8t_residual_adapter import SO8ResidualAdapter
import torch
adapter = SO8ResidualAdapter(1024)
x = torch.randn(2, 10, 1024)
out = adapter(x)
stats = adapter.get_adapter_stats()
print(f'Adapter created: ✓')
print(f'Forward pass: ✓ (shape: {out.shape})')
print(f'Stats: Alpha={stats[\"alpha\"]:.4f}, LieNorm={stats[\"lie_norm\"]:.6f}')
" >> %LOG_FILE% 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ NKAT integration failed!
    echo [%DATE% %TIME%] FAILED: NKAT integration >> %LOG_FILE%
    goto :error
)
echo ✅ NKAT Integration OK
echo [%DATE% %TIME%] PASSED: NKAT integration >> %LOG_FILE%

:: 4. 完全テストスイート実行
echo [STEP 4/5] Complete Test Suite...
echo [%DATE% %TIME%] Step 4: Complete Test Suite >> %LOG_FILE%
python run_rlpo_science_nsfw_automated_test.py >> %LOG_FILE% 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Complete test suite failed!
    echo [%DATE% %TIME%] FAILED: Complete test suite >> %LOG_FILE%
    goto :error
)
echo ✅ Complete Test Suite OK
echo [%DATE% %TIME%] PASSED: Complete test suite >> %LOG_FILE%

:: 5. ミニトレーニング実行
echo [STEP 5/5] Mini Training Test...
echo [%DATE% %TIME%] Step 5: Mini Training Test >> %LOG_FILE%
python scripts/training/rlpo_science_nsfw_automated.py --max_steps 5 --batch_size 1 --output_dir test_results/mini_rlpo_test >> %LOG_FILE% 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Mini training failed!
    echo [%DATE% %TIME%] FAILED: Mini training >> %LOG_FILE%
    goto :error
)
echo ✅ Mini Training OK
echo [%DATE% %TIME%] PASSED: Mini training >> %LOG_FILE%

:: 成功完了
echo.
echo 🎉 ALL TESTS PASSED SUCCESSFULLY!
echo ===================================
echo [%DATE% %TIME%] SUCCESS: All RLPO tests completed >> %LOG_FILE%
echo.
echo 📋 Test Results:
echo    - Environment: ✅
echo    - Datasets: ✅
echo    - NKAT Integration: ✅
echo    - Complete Test Suite: ✅
echo    - Mini Training: ✅
echo.
echo 📁 Logs saved to: %LOG_FILE%
echo 📊 Detailed reports in: test_results\
echo.
echo 🚀 Ready for full RLPO training!
echo    Run: python scripts/training/rlpo_science_nsfw_automated.py
echo.
goto :end

:error
echo.
echo ❌ TEST SUITE FAILED!
echo ====================
echo [%DATE% %TIME%] FAILED: Test suite encountered errors >> %LOG_FILE%
echo.
echo 📁 Check logs: %LOG_FILE%
echo 🔍 Check test results in: test_results\
echo.
exit /b 1

:end
echo [%DATE% %TIME%] Test suite completed >> %LOG_FILE%
