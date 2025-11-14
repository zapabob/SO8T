@echo off
REM Borea-Phi-3.5 SO8T/thinking再学習スクリプト（RTX3060最適化版）
REM 電源断対応機能付き（TimeBasedCheckpointCallback + auto-resume）

REM 変数展開を有効化（最初に設定）
setlocal enabledelayedexpansion

chcp 65001 >nul
echo [RETRAIN] Borea-Phi-3.5 SO8T/thinking Retraining (RTX3060 Optimized)
echo =====================================================================
echo.

REM 設定ファイルパス
set CONFIG=configs\train_borea_phi35_so8t_thinking_rtx3060.yaml

REM 設定ファイルの確認
echo [CHECK] Verifying config file...
if not exist "%CONFIG%" (
    echo [ERROR] Config file not found: %CONFIG%
    exit /b 1
)
echo [OK] Config file found: %CONFIG%

REM 設定ファイルからモデルパスと出力ディレクトリを読み取る（Pythonスクリプトを使用）
echo [INFO] Reading configuration from %CONFIG%...
for /f "delims=" %%a in ('py -3 -c "import yaml, sys; config = yaml.safe_load(open(r'%CONFIG%', encoding='utf-8')); print(config.get('model', {}).get('base_model', 'C:/Users/downl/Desktop/SO8T/models/Borea-Phi-3.5-mini-Instruct-Jp'))"') do set MODEL_PATH=%%a
for /f "delims=" %%a in ('py -3 -c "import yaml, sys; config = yaml.safe_load(open(r'%CONFIG%', encoding='utf-8')); print(config.get('training', {}).get('output_dir', 'D:/webdataset/checkpoints/training/borea_phi35_so8t_thinking_rtx3060'))"') do set OUTPUT_DIR=%%a

REM デフォルト値の設定（設定ファイルから読み取れない場合）
if not defined MODEL_PATH (
    set MODEL_PATH=C:\Users\downl\Desktop\SO8T\models\Borea-Phi-3.5-mini-Instruct-Jp
)
if not defined OUTPUT_DIR (
    set OUTPUT_DIR=D:\webdataset\checkpoints\training\borea_phi35_so8t_thinking_rtx3060
)

REM パスの正規化（スラッシュをバックスラッシュに変換）
set MODEL_PATH=!MODEL_PATH:/=\!
set OUTPUT_DIR=!OUTPUT_DIR:/=\!

echo [INFO] Model path: %MODEL_PATH%
echo [INFO] Config: %CONFIG%
echo [INFO] Output directory: %OUTPUT_DIR%
echo.

REM モデルパスの確認
echo [CHECK] Verifying model path...
if not exist "%MODEL_PATH%" (
    echo [ERROR] Model not found: %MODEL_PATH%
    exit /b 1
)
if not exist "%MODEL_PATH%\config.json" (
    echo [ERROR] Model config.json not found: %MODEL_PATH%\config.json
    exit /b 1
)
echo [OK] Model found: %MODEL_PATH%
echo.

REM データセットの確認・準備
echo [CHECK] Verifying dataset...

REM 設定ファイルからデータセット設定を読み取る（Pythonスクリプトを使用）
for /f "delims=" %%a in ('py -3 -c "import yaml, sys; config = yaml.safe_load(open(r'%CONFIG%', encoding='utf-8')); print(config.get('data', {}).get('train_data_dir', 'D:/webdataset/processed/thinking_quadruple'))"') do set QUADRUPLE_DATASET_DIR=%%a
for /f "delims=" %%a in ('py -3 -c "import yaml, sys; config = yaml.safe_load(open(r'%CONFIG%', encoding='utf-8')); print(config.get('data', {}).get('train_data_pattern', 'quadruple_thinking_*.jsonl'))"') do set QUADRUPLE_DATASET_PATTERN=%%a
for /f "delims=" %%a in ('py -3 -c "import yaml, sys; config = yaml.safe_load(open(r'%CONFIG%', encoding='utf-8')); print(config.get('data', {}).get('fallback_train_data', 'D:/webdataset/processed/thinking_sft/thinking_sft_dataset.jsonl'))"') do set THINKING_SFT_DATASET=%%a

REM デフォルト値の設定
if not defined QUADRUPLE_DATASET_DIR (
    set QUADRUPLE_DATASET_DIR=D:\webdataset\processed\thinking_quadruple
)
if not defined QUADRUPLE_DATASET_PATTERN (
    set QUADRUPLE_DATASET_PATTERN=quadruple_thinking_*.jsonl
)
if not defined THINKING_SFT_DATASET (
    set THINKING_SFT_DATASET=D:\webdataset\processed\thinking_sft\thinking_sft_dataset.jsonl
)

REM パスの正規化
set QUADRUPLE_DATASET_DIR=!QUADRUPLE_DATASET_DIR:/=\!
set THINKING_SFT_DATASET=!THINKING_SFT_DATASET:/=\!

set DATASET=

REM 四重推論形式データセットを優先的に検索
echo [INFO] Searching for quadruple thinking datasets...
echo [INFO] Directory: %QUADRUPLE_DATASET_DIR%
echo [INFO] Pattern: %QUADRUPLE_DATASET_PATTERN%
REM PowerShellコマンドを2つに分離: 1つ目で警告メッセージを標準エラー出力に、2つ目でデータセットパスのみを標準出力に
powershell -Command "$files = Get-ChildItem '%QUADRUPLE_DATASET_DIR%\%QUADRUPLE_DATASET_PATTERN%' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName; if ($files) { Write-Host '[OK] Found quadruple thinking dataset:' -ForegroundColor Green; Write-Host \"  $files\" -ForegroundColor Cyan } else { Write-Warning 'No quadruple thinking datasets found' }" 2>&1
for /f "delims=" %%f in ('powershell -Command "$files = Get-ChildItem '%QUADRUPLE_DATASET_DIR%\%QUADRUPLE_DATASET_PATTERN%' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName; if ($files) { Write-Output $files }"') do set DATASET=%%f

REM 四重推論形式が見つからない場合は、thinking_sftデータセットを確認
if not defined DATASET (
    if exist "%THINKING_SFT_DATASET%" (
        set DATASET=%THINKING_SFT_DATASET%
        echo [OK] Using thinking_sft dataset: %DATASET%
    ) else (
        echo [WARNING] Dataset not found: %THINKING_SFT_DATASET%
        echo [INFO] Attempting to create dataset from existing data...
        
        REM 四重推論形式データセットを作成
        call scripts\data\convert_to_quadruple_thinking.bat
        
        if errorlevel 1 (
            echo [WARNING] Failed to create quadruple thinking dataset, trying thinking_sft...
            
            REM thinking_sftデータセットを作成
            powershell -Command "$files = Get-ChildItem 'D:\webdataset\processed\four_class\four_class_*.jsonl' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName; if ($files) { py -3 scripts\data\create_thinking_sft_dataset.py --inputs $files --output '%THINKING_SFT_DATASET%'; if ($LASTEXITCODE -eq 0) { Write-Host '[OK] Created thinking_sft dataset' -ForegroundColor Green; Write-Output '%THINKING_SFT_DATASET%' } else { Write-Host '[ERROR] Failed to create thinking_sft dataset' -ForegroundColor Red; exit 1 } } else { Write-Host '[ERROR] No four_class datasets found' -ForegroundColor Red; exit 1 }" > "%TEMP%\dataset_path.txt"
            if exist "%TEMP%\dataset_path.txt" (
                for /f "delims=" %%f in ('type "%TEMP%\dataset_path.txt"') do set DATASET=%%f
                del "%TEMP%\dataset_path.txt"
            )
        ) else (
            REM 作成されたデータセットを検索
            powershell -Command "$files = Get-ChildItem 'D:\webdataset\processed\thinking_quadruple\quadruple_thinking_*.jsonl' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName; if ($files) { Write-Host '[OK] Created dataset:' -ForegroundColor Green; Write-Host \"  $files\" -ForegroundColor Cyan } else { Write-Error 'Dataset creation failed'; exit 1 }" 2>&1
            for /f "delims=" %%f in ('powershell -Command "$files = Get-ChildItem 'D:\webdataset\processed\thinking_quadruple\quadruple_thinking_*.jsonl' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName; if ($files) { Write-Output $files }"') do set DATASET=%%f
        )
        
        if not defined DATASET (
            echo [ERROR] Dataset not found after creation
            exit /b 1
        )
    )
)

if not defined DATASET (
    echo [ERROR] No dataset available
    exit /b 1
)

echo.
echo [INFO] Dataset: %DATASET%
echo [INFO] Auto-resume: Enabled
echo [INFO] Time-based checkpoint: Every 3 minutes
echo [INFO] RTX3060 optimization: Enabled
echo.

REM 再学習実行（--auto-resume付き）
echo [TRAINING] Starting retraining...
echo [INFO] Command: py -3 scripts\training\train_borea_phi35_so8t_thinking.py --config "%CONFIG%" --model-path "%MODEL_PATH%" --dataset "%DATASET%" --output-dir "%OUTPUT_DIR%" --auto-resume
echo.

py -3 scripts\training\train_borea_phi35_so8t_thinking.py ^
    --config "%CONFIG%" ^
    --model-path "%MODEL_PATH%" ^
    --dataset "%DATASET%" ^
    --output-dir "%OUTPUT_DIR%" ^
    --auto-resume

set TRAINING_EXIT_CODE=%ERRORLEVEL%

if %TRAINING_EXIT_CODE% neq 0 (
    echo.
    echo [ERROR] Training failed with error code %TRAINING_EXIT_CODE%
    echo [INFO] Check logs: logs\train_borea_phi35_so8t_thinking.log
    echo [INFO] Check output directory: %OUTPUT_DIR%
    echo [INFO] Check for checkpoint files in: %OUTPUT_DIR%\checkpoints
    exit /b %TRAINING_EXIT_CODE%
)

echo.
echo [SUCCESS] Retraining completed successfully!
echo [INFO] Model saved to: %OUTPUT_DIR%\final_model
echo.
echo [AUDIO] Playing completion notification...
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

