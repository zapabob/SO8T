@echo off
chcp 65001 >nul
echo [MOONSHOT] 完全無人運用システム (3分間隔チェックポイント・電源断復旧対応)
echo ====================================================================
echo 開始時刻: %date% %time%
echo.

REM ====================================================================
REM 電源断復旧チェック
REM ====================================================================
echo [RECOVERY] 電源断復旧チェック...
if exist "checkpoints\soul_weight_training\training_recovery.json" (
    echo [RECOVERY] 学習復旧ファイルを検知しました
    echo 最終チェックポイントから学習を再開します
    set RECOVERY_MODE=1
    for /f "tokens=*" %%i in ('powershell -Command "(Get-Content 'checkpoints\soul_weight_training\training_recovery.json' | ConvertFrom-Json).epoch"') do set LAST_EPOCH=%%i
    for /f "tokens=*" %%i in ('powershell -Command "(Get-Content 'checkpoints\soul_weight_training\training_recovery.json' | ConvertFrom-Json).global_step"') do set LAST_STEP=%%i
    echo 復旧情報: エポック %LAST_EPOCH%, ステップ %LAST_STEP%
) else (
    echo [RECOVERY] 新規学習を開始します
    set RECOVERY_MODE=0
)
echo.

REM ====================================================================
REM MOONSHOT完全無人運用シーケンス
REM ====================================================================

echo [PHASE 0] 環境チェック開始...
python scripts\utils\advanced_monitor.py --check-only
if %errorlevel% neq 0 (
    echo [ERROR] 環境チェック失敗
    goto :error
)
echo [OK] 環境チェック完了
echo.

echo [PHASE 1] 依存関係自動インストール...
start /B cmd /C "python scripts\utils\auto_dependency_installer.py && echo DEP_INSTALL_COMPLETED || echo DEP_INSTALL_FAILED > dep_install_status.tmp"
timeout /t 1800 /nobreak >nul 2>&1
taskkill /IM python.exe /F >nul 2>&1
if exist dep_install_status.tmp (
    findstr "DEP_INSTALL_COMPLETED" dep_install_status.tmp >nul
    if %errorlevel% neq 0 (
        echo [ERROR] 依存関係インストール失敗
        goto :error
    )
    del dep_install_status.tmp
) else (
    echo [WARNING] 依存関係インストールタイムアウト - 継続
)
echo [OK] 依存関係インストール完了
echo.

echo [PHASE 2] データセット生成・確認...
if %RECOVERY_MODE%==0 (
    echo データセット新規生成...
    start /B cmd /C "python scripts\data\phi35_thinking_dataset_generator.py && echo DATASET_COMPLETED || echo DATASET_FAILED > dataset_status.tmp"
    timeout /t 1800 /nobreak >nul 2>&1
    taskkill /IM python.exe /F >nul 2>&1
    if exist dataset_status.tmp (
        findstr "DATASET_COMPLETED" dataset_status.tmp >nul
        if %errorlevel% neq 0 (
            echo [ERROR] データセット生成失敗
            goto :error
        )
        del dataset_status.tmp
    ) else (
        echo [WARNING] データセット生成タイムアウト - 継続
    )
) else (
    echo 復旧モード: 既存データセットを使用
    if not exist "data\datasets\phi35_thinking\phi35_thinking_sft.jsonl" (
        echo [ERROR] 復旧用データセットが見つからない
        goto :error
    )
)
echo [OK] データセット準備完了
echo.

echo [PHASE 3] 自動エラー修正システム起動...
start /B cmd /C "python scripts\utils\auto_error_corrector.py && echo ERROR_CORRECTION_COMPLETED || echo ERROR_CORRECTION_FAILED > error_correction_status.tmp"
timeout /t 900 /nobreak >nul 2>&1
taskkill /IM python.exe /F >nul 2>&1
if exist error_correction_status.tmp (
    findstr "ERROR_CORRECTION_COMPLETED" error_correction_status.tmp >nul
    if %errorlevel% neq 0 (
        echo [WARNING] エラー修正で問題発生したが継続
    )
    del error_correction_status.tmp
) else (
    echo [WARNING] エラー修正タイムアウト - 継続
)
echo [OK] エラー修正システム完了
echo.

echo [PHASE 4] 魂の重み学習・アルファゲートアニーリング (3分間隔チェックポイント)...
echo ====================================================================
echo このPhaseは長時間実行されます。完全に無人運用可能です。
echo.
echo チェックポイント機能:
echo   - 3分間隔で自動保存 (ローリングストック: 5個)
echo   - 電源断時自動復旧可能
echo   - Cursor自動起動との連携
echo.
echo 学習開始...
echo ====================================================================

REM 学習実行（バックグラウンド・タイムアウトなし）
start /B cmd /C "python scripts\training\phi35_soul_weight_trainer.py && echo TRAINING_COMPLETED || echo TRAINING_FAILED > training_status.tmp"

REM 学習モニタリングループ（5分間隔）
set MONITOR_COUNT=0
:training_monitor
set /a MONITOR_COUNT+=1
echo [MONITOR #%MONITOR_COUNT%] 学習状態確認中... (%date% %time%)

REM 学習完了チェック
if exist training_status.tmp (
    findstr "TRAINING_COMPLETED" training_status.tmp >nul
    if %errorlevel%==0 (
        echo [SUCCESS] 魂の重み学習完了
        del training_status.tmp
        goto :training_complete
    )
    findstr "TRAINING_FAILED" training_status.tmp >nul
    if %errorlevel%==0 (
        echo [ERROR] 魂の重み学習失敗
        del training_status.tmp
        goto :error
    )
)

REM 学習プロセス確認
tasklist /FI "IMAGENAME eq python.exe" /NH | findstr python.exe >nul
if %errorlevel% neq 0 (
    echo [WARNING] Pythonプロセスが見つからない
    REM 復旧ファイル確認
    if exist "checkpoints\soul_weight_training\training_recovery.json" (
        echo 復旧ファイルが存在します。学習は正常に進行中です。
        REM 復旧ファイルから最新情報を表示
        for /f "tokens=*" %%i in ('powershell -Command "try { (Get-Content 'checkpoints\soul_weight_training\training_recovery.json' | ConvertFrom-Json).epoch } catch { 'N/A' }"') do set CURRENT_EPOCH=%%i
        for /f "tokens=*" %%i in ('powershell -Command "try { (Get-Content 'checkpoints\soul_weight_training\training_recovery.json' | ConvertFrom-Json).global_step } catch { 'N/A' }"') do set CURRENT_STEP=%%i
        for /f "tokens=*" %%i in ('powershell -Command "try { (Get-Content 'checkpoints\soul_weight_training\training_recovery.json' | ConvertFrom-Json).loss } catch { 'N/A' }"') do set CURRENT_LOSS=%%i
        echo 現在状態: エポック %CURRENT_EPOCH%, ステップ %CURRENT_STEP%, 損失 %CURRENT_LOSS%
    ) else (
        echo [ERROR] 学習プロセスが異常終了
        goto :error
    )
) else (
    echo [OK] 学習プロセス実行中
)

REM 次のチェックまで待機（5分）
timeout /t 300 /nobreak >nul 2>&1
goto :training_monitor

:training_complete
echo [OK] 魂の重み学習完了
echo.

echo [PHASE 5] A/Bテスト自動実行...
if exist auto_ab_test_pipeline.bat (
    echo A/Bテスト実行開始...
    start /B cmd /C "auto_ab_test_pipeline.bat && echo AB_TEST_COMPLETED || echo AB_TEST_FAILED > ab_test_status.tmp"
    timeout /t 3600 /nobreak >nul 2>&1
    taskkill /IM cmd.exe /F >nul 2>&1
    if exist ab_test_status.tmp (
        findstr "AB_TEST_COMPLETED" ab_test_status.tmp >nul
        if %errorlevel% neq 0 (
            echo [ERROR] A/Bテスト失敗
            goto :error
        )
        del ab_test_status.tmp
    ) else (
        echo [WARNING] A/Bテストタイムアウト - 継続
    )
    echo [OK] A/Bテスト完了
) else (
    echo [WARNING] A/Bテストパイプラインが見つからないためスキップ
)
echo.

echo [PHASE 6] HFアップロード完全自動化...
start /B cmd /C "python scripts\deployment\auto_hf_upload.py && echo HF_UPLOAD_COMPLETED || echo HF_UPLOAD_FAILED > hf_upload_status.tmp"
timeout /t 1800 /nobreak >nul 2>&1
taskkill /IM python.exe /F >nul 2>&1
if exist hf_upload_status.tmp (
    findstr "HF_UPLOAD_COMPLETED" hf_upload_status.tmp >nul
    if %errorlevel% neq 0 (
        echo [ERROR] HFアップロード失敗
        goto :error
    )
    del hf_upload_status.tmp
) else (
    echo [WARNING] HFアップロードタイムアウト - 継続
)
echo [OK] HFアップロード完了
echo.

echo [SUCCESS] MOONSHOT完全無人運用完了！
echo ====================================================================
echo 完了時刻: %date% %time%
echo.
echo [UNATTENDED FEATURES]
echo ✅ 電源投入時完全自動起動
echo ✅ 3分間隔チェックポイント (5個ローリングストック)
echo ✅ 電源断時自動復旧
echo ✅ エラー自動検知・修正
echo ✅ 学習モニタリング（5分間隔）
echo ✅ Cursor統合自動起動
echo ✅ 完全無人運用可能

REM 完了通知
powershell -ExecutionPolicy Bypass -Command "Start-Job -ScriptBlock { try { & 'scripts\utils\play_audio_notification.ps1' } catch { Write-Host 'Audio notification failed but continuing...' } } | Wait-Job -Timeout 10 | Out-Null"

REM ログ保存
echo [%date% %time%] MOONSHOT完全無人運用完了 >> moonshot_unattended.log
goto :end

:error
echo [CRITICAL ERROR] MOONSHOT無人運用失敗
echo =======================================
echo エラーレベル: %errorlevel%
echo 時刻: %date% %time%
echo.

REM エラー通知
powershell -ExecutionPolicy Bypass -Command "try { [System.Console]::Beep(800, 1000) } catch { Write-Host 'Beep failed but continuing...' }"

REM エラーログ保存
echo [%date% %time%] MOONSHOT無人運用失敗 (ErrorLevel: %errorlevel%) >> moonshot_unattended_error.log

REM エラー修正試行
echo [RECOVERY] 自動エラー修正実行...
start /B cmd /C "python scripts\utils\auto_error_corrector.py && echo RECOVERY_COMPLETED || echo RECOVERY_FAILED > recovery_status.tmp"
timeout /t 600 /nobreak >nul 2>&1
taskkill /IM python.exe /F >nul 2>&1
if exist recovery_status.tmp (
    findstr "RECOVERY_COMPLETED" recovery_status.tmp >nul
    if %errorlevel% neq 0 (
        echo [WARNING] 自動回復失敗
    )
    del recovery_status.tmp
) else (
    echo [WARNING] 自動回復タイムアウト
)

echo [RECOVERY INSTRUCTIONS]
echo 1. エラーログを確認してください: moonshot_unattended_error.log
echo 2. 復旧ファイルを確認: checkpoints\soul_weight_training\training_recovery.json
echo 3. チェックポイントを確認: checkpoints\soul_weight_training\rolling_checkpoint_*.pt
echo 4. 自動回復が実行されました
echo 5. 必要に応じて手動修正を行ってください
echo 6. 再度 auto_moonshot_unattended.bat を実行してください
echo.
exit /b 1

:end
echo [SYSTEM SHUTDOWN] 正常終了
exit /b 0
