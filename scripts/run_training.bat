@echo off
REM Windows 11 での自動再開用バッチ
REM 電源投入時にタスクスケジューラから実行されることを想定

REM UTF-8エンコーディング設定
chcp 65001 >nul

REM conda環境の有効化（必要に応じてパスを調整）
REM 例: call "%USERPROFILE%\miniconda3\Scripts\activate.bat" so8t-think
REM または: call "%USERPROFILE%\anaconda3\Scripts\activate.bat" so8t-think

REM プロジェクトディレクトリに移動
cd /d C:\Users\downl\Desktop\SO8T

REM ログファイルに出力を記録（オプション）
REM python scripts\training\train_so8t_lora.py --output_dir D:\webdataset\checkpoints\training\so8t_lora --auto-resume >> logs\training_auto_resume.log 2>&1

REM auto-resume を有効にして学習を再開
REM 実際の実行時は、以下の引数を適切に設定してください:
REM   --base_model: ベースモデルのパス
REM   --dataset: 学習データセットのパス
REM   --output_dir: チェックポイント保存先（D:\webdataset\checkpoints\training\so8t_lora を推奨）
REM   --auto-resume: 自動再開フラグ（必須）

python scripts\training\train_so8t_lora.py ^
    --base_model models/Borea-Phi-3.5-mini-Instruct-Jp ^
    --dataset data/train.jsonl ^
    --output_dir D:\webdataset\checkpoints\training\so8t_lora ^
    --lora_r 16 ^
    --lora_alpha 32 ^
    --lora_dropout 0.05 ^
    --batch_size 1 ^
    --gradient_accumulation_steps 8 ^
    --learning_rate 2e-4 ^
    --num_epochs 3 ^
    --max_length 2048 ^
    --save_steps 500 ^
    --logging_steps 10 ^
    --load_in_4bit ^
    --auto-resume

REM エラーコードを返す（タスクスケジューラでエラー検出に使用）
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Training failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo [OK] Training completed successfully



