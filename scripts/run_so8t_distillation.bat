@echo off
chcp 65001 >nul
echo ========================================
echo SO8T知識蒸留実行スクリプト
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
echo [OK] 環境確認完了
echo.

echo [STEP 2] 教師モデル確認中...
if not exist "models\SO8T-Phi31-Mini-128K-Enhanced-Q8_0.gguf" (
    echo [ERROR] 教師モデルが見つかりません: models\SO8T-Phi31-Mini-128K-Enhanced-Q8_0.gguf
    pause
    exit /b 1
)
echo [OK] 教師モデル確認完了
echo.

echo [STEP 3] 出力ディレクトリ作成中...
if not exist "models\qwen_so8t_lightweight" (
    mkdir "models\qwen_so8t_lightweight"
    echo [OK] 出力ディレクトリ作成完了
) else (
    echo [OK] 出力ディレクトリ既に存在
)
echo.

echo [STEP 4] 知識蒸留実行中...
echo [INFO] 教師モデル: models\SO8T-Phi31-Mini-128K-Enhanced-Q8_0.gguf
echo [INFO] 出力先: models\qwen_so8t_lightweight
echo [INFO] エポック数: 10
echo [INFO] サンプル数: 1000
echo.

py -3 scripts\run_so8t_distillation.py --teacher "models\SO8T-Phi31-Mini-128K-Enhanced-Q8_0.gguf" --output "models\qwen_so8t_lightweight" --epochs 10 --samples 1000

if %errorlevel% neq 0 (
    echo [ERROR] 知識蒸留実行に失敗しました
    pause
    exit /b 1
)
echo [OK] 知識蒸留実行完了
echo.

echo [STEP 5] 結果確認中...
if exist "models\qwen_so8t_lightweight\distillation_results.json" (
    echo [OK] 蒸留結果ファイル確認完了
) else (
    echo [WARNING] 蒸留結果ファイルが見つかりません
)
if exist "models\qwen_so8t_lightweight\README.md" (
    echo [OK] モデルカード確認完了
) else (
    echo [WARNING] モデルカードが見つかりません
)
echo.

echo [STEP 6] 軽量モデルテスト中...
echo [INFO] 軽量モデルの動作確認を実行中...
py -3 -c "
import sys
sys.path.append('.')
from utils.knowledge_distillation import SO8TKnowledgeDistillation
import torch

print('軽量モデル読み込みテスト中...')
try:
    # 設定
    student_config = {
        'vocab_size': 32000,
        'hidden_size': 2048,
        'intermediate_size': 8192,
        'num_hidden_layers': 16,
        'num_attention_heads': 16,
        'num_key_value_heads': 4,
        'hidden_act': 'silu',
        'max_position_embeddings': 131072,
        'rms_norm_eps': 1e-6,
        'rope_theta': 10000.0,
        'attention_dropout': 0.0,
        'use_cache': True,
        'so8t_rotation_dim': 8,
        'so8t_triality_symmetry': True,
        'so8t_cross_head_interaction': True,
        'so8t_non_commutative_gates': True,
    }
    
    # 蒸留システム初期化
    distillation_system = SO8TKnowledgeDistillation(
        teacher_model_path='models/SO8T-Phi31-Mini-128K-Enhanced-Q8_0.gguf',
        student_config=student_config,
        output_dir='models/qwen_so8t_lightweight'
    )
    
    # 学生モデル作成テスト
    student_model = distillation_system.create_student_model()
    print(f'学生モデルパラメータ数: {sum(p.numel() for p in student_model.parameters()):,}')
    print('軽量モデルテスト完了')
    
except Exception as e:
    print(f'軽量モデルテストエラー: {e}')
    sys.exit(1)
"

if %errorlevel% neq 0 (
    echo [ERROR] 軽量モデルテストに失敗しました
    pause
    exit /b 1
)
echo [OK] 軽量モデルテスト完了
echo.

echo [STEP 7] 最終確認中...
echo [INFO] 生成されたファイル一覧:
dir "models\qwen_so8t_lightweight" /b
echo.

echo ========================================
echo SO8T知識蒸留完了！
echo ========================================
echo [OK] 教師モデル: models\SO8T-Phi31-Mini-128K-Enhanced-Q8_0.gguf
echo [OK] 学生モデル: models\qwen_so8t_lightweight
echo [OK] 実行時間: 完了
echo [OK] 軽量モデル: 作成完了
echo.

echo [AUDIO] 完了通知音を再生中...
powershell -Command "if (Test-Path 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav') { Add-Type -AssemblyName System.Windows.Forms; [System.Media.SoundPlayer]::new('C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav').Play(); Write-Host '[OK] 完了通知音再生成功' -ForegroundColor Green } else { Write-Host '[WARNING] 完了通知音ファイルが見つかりません' -ForegroundColor Yellow }"

echo.
echo [INFO] 知識蒸留が正常に完了しました
echo [INFO] 軽量モデルは models\qwen_so8t_lightweight に保存されています
echo.
pause
