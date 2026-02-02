@echo off
chcp 65001 >nul
echo ================================================================================
echo Qwen2-VL-2B-Instruct → SO8T → GGUF 変換パイプライン
echo ================================================================================
echo.

REM 設定
set "PROJECT_ROOT=%~dp0.."
set "INPUT_MODEL=%PROJECT_ROOT%\models\Qwen2-VL-2B-Instruct"
set "SO8T_MODEL=%PROJECT_ROOT%\models\so8t-vl-2b-instruct"
set "GGUF_MODEL=%PROJECT_ROOT%\models\so8t-vl-2b-instruct.gguf"
set "LOG_DIR=%PROJECT_ROOT%\_docs\conversion_logs"

REM ログディレクトリの作成
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM タイムスタンプ
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "timestamp=%dt:~0,4%-%dt:~4,2%-%dt:~6,2%_%dt:~8,2%-%dt:~10,2%-%dt:~12,2%"

echo [INFO] 変換開始時刻: %timestamp%
echo [INFO] プロジェクトルート: %PROJECT_ROOT%
echo [INFO] 入力モデル: %INPUT_MODEL%
echo [INFO] SO8Tモデル: %SO8T_MODEL%
echo [INFO] GGUFモデル: %GGUF_MODEL%
echo.

REM 環境チェック
echo [PHASE 1] 環境チェック
echo [CHECK] Python環境確認中...
py --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Pythonが見つかりません
    goto :error_exit
)

echo [CHECK] PyTorch環境確認中...
py -c "import torch; print('PyTorch version:', torch.__version__)" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] PyTorchが見つかりません
    goto :error_exit
)

echo [CHECK] Transformers環境確認中...
py -c "import transformers; print('Transformers version:', transformers.__version__)" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Transformersが見つかりません
    goto :error_exit
)

echo [CHECK] SafeTensors環境確認中...
py -c "import safetensors; print('SafeTensors version:', safetensors.__version__)" >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] SafeTensorsが見つかりません。pip install safetensors を実行してください。
)

echo [OK] 環境チェック完了
echo.

REM 入力モデルの確認
echo [PHASE 2] 入力モデル確認
if not exist "%INPUT_MODEL%" (
    echo [ERROR] 入力モデルが見つかりません: %INPUT_MODEL%
    goto :error_exit
)

if not exist "%INPUT_MODEL%\config.json" (
    echo [ERROR] 設定ファイルが見つかりません: %INPUT_MODEL%\config.json
    goto :error_exit
)

if not exist "%INPUT_MODEL%\model-00001-of-00002.safetensors" (
    echo [ERROR] 重みファイルが見つかりません: %INPUT_MODEL%\model-00001-of-00002.safetensors
    goto :error_exit
)

echo [OK] 入力モデル確認完了
echo.

REM SO8T変換
echo [PHASE 3] SO8T変換実行
echo [CONVERT] Qwen2-VL-2B-Instruct → SO8T変換中...
py "%PROJECT_ROOT%\scripts\convert_qwen2vl_to_so8t.py" ^
    --input-model "%INPUT_MODEL%" ^
    --output-model "%SO8T_MODEL%" ^
    --hidden-size 1536 ^
    --rotation-dim 8 ^
    --safety-features ^
    --verbose > "%LOG_DIR%\so8t_conversion_%timestamp%.log" 2>&1

if %errorlevel% neq 0 (
    echo [ERROR] SO8T変換に失敗しました
    echo [INFO] ログファイル: %LOG_DIR%\so8t_conversion_%timestamp%.log
    goto :error_exit
)

echo [OK] SO8T変換完了
echo.

REM SO8Tモデルの確認
echo [PHASE 4] SO8Tモデル確認
if not exist "%SO8T_MODEL%" (
    echo [ERROR] SO8Tモデルが作成されていません: %SO8T_MODEL%
    goto :error_exit
)

if not exist "%SO8T_MODEL%\config.json" (
    echo [ERROR] SO8T設定ファイルが見つかりません: %SO8T_MODEL%\config.json
    goto :error_exit
)

if not exist "%SO8T_MODEL%\model.safetensors" (
    echo [ERROR] SO8T重みファイルが見つかりません: %SO8T_MODEL%\model.safetensors
    goto :error_exit
)

echo [OK] SO8Tモデル確認完了
echo.

REM GGUF変換
echo [PHASE 5] GGUF変換実行
echo [CONVERT] SO8T → GGUF変換中...
py "%PROJECT_ROOT%\scripts\convert_so8t_to_gguf.py" ^
    --input-model "%SO8T_MODEL%" ^
    --output-gguf "%GGUF_MODEL%" ^
    --quantization Q8_0 ^
    --model-name "so8t-vl-2b-instruct" ^
    --verbose > "%LOG_DIR%\gguf_conversion_%timestamp%.log" 2>&1

if %errorlevel% neq 0 (
    echo [ERROR] GGUF変換に失敗しました
    echo [INFO] ログファイル: %LOG_DIR%\gguf_conversion_%timestamp%.log
    goto :error_exit
)

echo [OK] GGUF変換完了
echo.

REM GGUFモデルの確認
echo [PHASE 6] GGUFモデル確認
if not exist "%GGUF_MODEL%" (
    echo [ERROR] GGUFモデルが作成されていません: %GGUF_MODEL%
    goto :error_exit
)

echo [INFO] GGUFファイルサイズ:
for %%F in ("%GGUF_MODEL%") do echo   %%~nxF: %%~zF bytes (%%~zF/1024/1024/1024 GB)

echo [OK] GGUFモデル確認完了
echo.

REM 変換結果のサマリー
echo [PHASE 7] 変換結果サマリー
echo ================================================================================
echo 変換完了サマリー
echo ================================================================================
echo 開始時刻: %timestamp%
echo 完了時刻: %date% %time%
echo.
echo 入力モデル: %INPUT_MODEL%
echo SO8Tモデル: %SO8T_MODEL%
echo GGUFモデル: %GGUF_MODEL%
echo.
echo ログファイル:
echo   SO8T変換: %LOG_DIR%\so8t_conversion_%timestamp%.log
echo   GGUF変換: %LOG_DIR%\gguf_conversion_%timestamp%.log
echo.

REM ファイルサイズの比較
echo ファイルサイズ比較:
for %%F in ("%INPUT_MODEL%\model-00001-of-00002.safetensors") do echo   入力モデル: %%~zF bytes
for %%F in ("%SO8T_MODEL%\model.safetensors") do echo   SO8Tモデル: %%~zF bytes
for %%F in ("%GGUF_MODEL%") do echo   GGUFモデル: %%~zF bytes
echo.

REM 成功音声通知
echo [AUDIO] 成功通知を再生中...
powershell -Command "if (Test-Path 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav') { Add-Type -AssemblyName System.Windows.Forms; [System.Media.SoundPlayer]::new('C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav').Play(); Write-Host '[OK] 成功通知再生完了' -ForegroundColor Green }"

echo [SUCCESS] 全ての変換が正常に完了しました！
echo ================================================================================
goto :success_exit

:error_exit
echo.
echo ================================================================================
echo 変換失敗
echo ================================================================================
echo 失敗時刻: %date% %time%
echo ログディレクトリ: %LOG_DIR%
echo.

REM エラー音声通知
powershell -Command "if (Test-Path 'C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav') { Add-Type -AssemblyName System.Windows.Forms; [System.Media.SoundPlayer]::new('C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav').Play(); Write-Host '[ERROR] エラー通知再生完了' -ForegroundColor Red }"

exit /b 1

:success_exit
exit /b 0
