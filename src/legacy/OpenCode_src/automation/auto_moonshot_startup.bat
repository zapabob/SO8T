@echo off
chcp 65001 >nul
echo [MOONSHOT] 電源投入時自動起動システム
echo ======================================
echo 開始時刻: %date% %time%
echo.

REM MOONSHOT完全自動化シーケンス
echo [PHASE 0] 環境チェック開始...
python scripts\utils\advanced_monitor.py --check-only
if %errorlevel% neq 0 (
    echo [ERROR] 環境チェック失敗
    goto :error
)
echo [OK] 環境チェック完了
echo.

echo [PHASE 1-2] 依存関係自動インストール・データセット生成...
python scripts\data\phi35_thinking_dataset_generator.py
if %errorlevel% neq 0 (
    echo [ERROR] データセット生成失敗
    goto :error
)
echo [OK] データセット生成完了
echo.

echo [PHASE 3] 自動エラー修正システム起動...
python scripts\utils\auto_error_corrector.py
if %errorlevel% neq 0 (
    echo [WARNING] エラー修正で問題発生したが継続
)
echo [OK] エラー修正システム完了
echo.

echo [PHASE 4-5] 魂の重み学習・アルファゲートアニーリング...
python scripts\training\phi35_soul_weight_trainer.py
if %errorlevel% neq 0 (
    echo [ERROR] 学習失敗
    goto :error
)
echo [OK] 魂の重み学習完了
echo.

echo [PHASE 6] A/Bテスト自動実行...
REM A/Bテストパイプライン実行
if exist auto_ab_test_pipeline.bat (
    call auto_ab_test_pipeline.bat
    if %errorlevel% neq 0 (
        echo [ERROR] A/Bテスト失敗
        goto :error
    )
    echo [OK] A/Bテスト完了
) else (
    echo [WARNING] A/Bテストパイプラインが見つからないためスキップ
)
echo.

echo [PHASE 7-8] HFアップロード完全自動化...
python scripts\deployment\auto_hf_upload.py
if %errorlevel% neq 0 (
    echo [ERROR] HFアップロード失敗
    goto :error
)
echo [OK] HFアップロード完了
echo.

echo [SUCCESS] MOONSHOT完全自動化完了！
echo ======================================
echo 完了時刻: %date% %time%

REM 完了通知
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"

REM ログ保存
echo [%date% %time%] MOONSHOT自動起動完了 >> moonshot_autostart.log
goto :end

:error
echo [CRITICAL ERROR] MOONSHOT自動起動失敗
echo エラーレベル: %errorlevel%
echo 時刻: %date% %time%

REM エラー通知（異なる音声）
powershell -ExecutionPolicy Bypass -Command "[System.Console]::Beep(800, 1000)"

REM エラーログ保存
echo [%date% %time%] MOONSHOT自動起動失敗 (ErrorLevel: %errorlevel%) >> moonshot_autostart_error.log

REM エラー修正試行
echo [RECOVERY] 自動エラー修正実行...
python scripts\utils\auto_error_corrector.py

exit /b 1

:end
echo [SHUTDOWN] システム正常終了
exit /b 0
