@echo off
chcp 65001 >nul
echo ========================================
echo 軽量モデル複雑テスト実行スクリプト
echo ========================================
echo.

echo [INFO] 現在の日時: %date% %time%
echo [INFO] 作業ディレクトリ: %CD%
echo.

echo [STEP 1] 環境確認中...
py -3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
if %errorlevel% neq 0 (
    echo [ERROR] Python環境の確認に失敗しました
    pause
    exit /b 1
)
echo [OK] Python環境確認完了
echo.

echo [STEP 2] Ollama環境確認中...
ollama --version
if %errorlevel% neq 0 (
    echo [ERROR] Ollama環境の確認に失敗しました
    pause
    exit /b 1
)
echo [OK] Ollama環境確認完了
echo.

echo [STEP 3] 軽量モデルGGUF変換中...
echo [INFO] 蒸留された軽量モデルをGGUF形式に変換中...
py -3 scripts\convert_lightweight_to_gguf.py --model "models\qwen_so8t_lightweight\checkpoints\student_model_final.pt" --output "models\qwen_so8t_lightweight_gguf"

if %errorlevel% neq 0 (
    echo [ERROR] 軽量モデルGGUF変換に失敗しました
    pause
    exit /b 1
)
echo [OK] 軽量モデルGGUF変換完了
echo.

echo [STEP 4] Ollamaモデル作成中...
echo [INFO] 軽量モデルをOllamaに登録中...
ollama create so8t-lightweight -f modelfiles\Modelfile-SO8T-Lightweight-Distilled

if %errorlevel% neq 0 (
    echo [ERROR] Ollamaモデル作成に失敗しました
    pause
    exit /b 1
)
echo [OK] Ollamaモデル作成完了
echo.

echo [STEP 5] 軽量モデル動作確認中...
echo [INFO] 軽量モデルの基本動作を確認中...
ollama run so8t-lightweight "こんにちは！あなたは何ができますか？"

if %errorlevel% neq 0 (
    echo [ERROR] 軽量モデル動作確認に失敗しました
    pause
    exit /b 1
)
echo [OK] 軽量モデル動作確認完了
echo.

echo [STEP 6] 複雑なテスト実行中...
echo [INFO] 数学的推論、科学的概念、論理的推論、創造的作文、倫理的推論のテストを実行中...
echo [INFO] このテストは時間がかかる場合があります...
echo.

py -3 scripts\test_lightweight_model_complex.py --model "so8t-lightweight" --output "_docs\lightweight_model_test_results.json"

if %errorlevel% neq 0 (
    echo [ERROR] 複雑なテスト実行に失敗しました
    pause
    exit /b 1
)
echo [OK] 複雑なテスト実行完了
echo.

echo [STEP 7] 結果確認中...
if exist "_docs\lightweight_model_test_results.json" (
    echo [OK] テスト結果ファイル確認完了
    echo [INFO] テスト結果の概要:
    py -3 -c "
import json
with open('_docs/lightweight_model_test_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)
print(f'総テスト数: {results[\"total_tests\"]}')
print(f'成功テスト数: {results[\"total_successful\"]}')
print(f'総合成功率: {results[\"overall_success_rate\"]:.2%}')
print(f'総実行時間: {results[\"total_execution_time\"]:.2f}秒')
"
) else (
    echo [WARNING] テスト結果ファイルが見つかりません
)
echo.

echo [STEP 8] 最終確認中...
echo [INFO] 生成されたファイル一覧:
dir "models\qwen_so8t_lightweight_gguf" /b
echo.

echo ========================================
echo 軽量モデル複雑テスト完了！
echo ========================================
echo [OK] 軽量モデル: so8t-lightweight
echo [OK] GGUF変換: 完了
echo [OK] Ollama登録: 完了
echo [OK] 複雑なテスト: 完了
echo.

echo [AUDIO] 完了通知音を再生中...
powershell -Command "if (Test-Path 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav') { Add-Type -AssemblyName System.Windows.Forms; [System.Media.SoundPlayer]::new('C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav').Play(); Write-Host '[OK] 完了通知音再生成功' -ForegroundColor Green } else { Write-Host '[WARNING] 完了通知音ファイルが見つかりません' -ForegroundColor Yellow }"

echo.
echo [INFO] 軽量モデルの複雑なテストが正常に完了しました
echo [INFO] テスト結果は _docs\lightweight_model_test_results.json に保存されています
echo.
pause
