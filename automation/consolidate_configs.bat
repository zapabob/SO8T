@echo off
chcp 65001 >nul
echo [CONFIG] 設定ファイル統合を開始...

REM configs/ ディレクトリ内の全てのファイルを so8t/config/ にコピー（上書き）
if exist "configs" (
    echo configs/ から so8t/config/ へファイルを統合...
    xcopy "configs\*.*" "so8t\config\" /E /I /H /Y
    echo ファイル統合完了

    REM modelfiles/ が存在する場合、適切に統合
    if exist "configs\modelfiles" (
        echo modelfiles/ を統合...
        if not exist "so8t\config\modelfiles" mkdir "so8t\config\modelfiles"
        xcopy "configs\modelfiles\*.*" "so8t\config\modelfiles\" /E /I /H /Y
    )

    REM configs/ ディレクトリを削除
    echo 古い configs/ ディレクトリを削除...
    rmdir /S /Q "configs"
    echo configs/ ディレクトリ削除完了
) else (
    echo configs/ ディレクトリが見つかりません
)

REM 統合結果の確認
echo 統合結果:
dir /s /b "so8t\config" | find /c ".yaml" > temp_count.txt
set /p yaml_count=<temp_count.txt
dir /s /b "so8t\config" | find /c ".json" > temp_count.txt
set /p json_count=<temp_count.txt
del temp_count.txt
set /a total_count=%yaml_count% + %json_count%
echo so8t/config/ 内の設定ファイル数: %total_count%

echo [CONFIG] 設定ファイル統合完了

REM オーディオ再生
powershell -ExecutionPolicy Bypass -File "scripts\utils\play_audio_notification.ps1"
