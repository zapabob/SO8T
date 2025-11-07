@echo off
chcp 65001 >nul
echo ========================================
echo 簡易SO8T知識蒸留実行スクリプト
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

echo [STEP 4] 簡易知識蒸留実行中...
echo [INFO] 教師モデル: models\SO8T-Phi31-Mini-128K-Enhanced-Q8_0.gguf
echo [INFO] 出力先: models\qwen_so8t_lightweight
echo [INFO] エポック数: 5
echo [INFO] サンプル数: 100
echo.

py -3 scripts\simple_so8t_distillation.py

if %errorlevel% neq 0 (
    echo [ERROR] 簡易知識蒸留実行に失敗しました
    pause
    exit /b 1
)
echo [OK] 簡易知識蒸留実行完了
echo.

echo [STEP 5] 結果確認中...
if exist "models\qwen_so8t_lightweight\checkpoints" (
    echo [OK] チェックポイントディレクトリ確認完了
    dir "models\qwen_so8t_lightweight\checkpoints" /b
) else (
    echo [WARNING] チェックポイントディレクトリが見つかりません
)
echo.

echo [STEP 6] 軽量モデルテスト中...
echo [INFO] 軽量モデルの動作確認を実行中...
py -3 -c "
import sys
sys.path.append('.')
import torch
import torch.nn as nn

print('軽量モデル読み込みテスト中...')
try:
    # チェックポイント読み込みテスト
    checkpoint_path = 'models/qwen_so8t_lightweight/checkpoints/student_model_final.pt'
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f'チェックポイント読み込み成功: {checkpoint_path}')
    print(f'エポック: {checkpoint.get(\"epoch\", \"N/A\")}')
    print(f'損失: {checkpoint.get(\"loss\", \"N/A\")}')
    print(f'タイムスタンプ: {checkpoint.get(\"timestamp\", \"N/A\")}')
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
echo 簡易SO8T知識蒸留完了！
echo ========================================
echo [OK] 教師モデル: models\SO8T-Phi31-Mini-128K-Enhanced-Q8_0.gguf
echo [OK] 学生モデル: models\qwen_so8t_lightweight
echo [OK] 実行時間: 完了
echo [OK] 軽量モデル: 作成完了
echo.

echo [AUDIO] 完了通知音を再生中...
powershell -Command "if (Test-Path 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav') { Add-Type -AssemblyName System.Windows.Forms; [System.Media.SoundPlayer]::new('C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav').Play(); Write-Host '[OK] 完了通知音再生成功' -ForegroundColor Green } else { Write-Host '[WARNING] 完了通知音ファイルが見つかりません' -ForegroundColor Yellow }"

echo.
echo [INFO] 簡易知識蒸留が正常に完了しました
echo [INFO] 軽量モデルは models\qwen_so8t_lightweight に保存されています
echo.
pause
