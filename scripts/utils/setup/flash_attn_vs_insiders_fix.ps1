# Flash Attention Visual Studio Insiders 問題の解決策
# CUDA 12.8は Visual Studio 2022 Insiders (v18) をサポートしていない

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Flash Attention VS Insiders 問題解決策" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[問題] Visual Studio 2022 Insiders (v18) が検出されました" -ForegroundColor Red
Write-Host "CUDA 12.8は Visual Studio 2017-2022 (通常版) のみをサポートしています" -ForegroundColor Yellow
Write-Host ""

Write-Host "[解決策 1] Visual Studio 2022 (通常版) をインストール" -ForegroundColor Green
Write-Host "  1. Visual Studio 2022 (通常版) をダウンロード:" -ForegroundColor White
Write-Host "     https://visualstudio.microsoft.com/downloads/" -ForegroundColor Cyan
Write-Host "  2. C++ build tools をインストール" -ForegroundColor White
Write-Host "  3. Visual Studio 2022 Insiders を無効化または削除" -ForegroundColor White
Write-Host "  4. 環境変数を設定:" -ForegroundColor White
Write-Host "     `$env:VCINSTALLDIR = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\'" -ForegroundColor Cyan
Write-Host "  5. 再インストール試行:" -ForegroundColor White
Write-Host "     py -3.12 -m pip install flash_attn==2.5.8 --no-build-isolation" -ForegroundColor Cyan
Write-Host ""

Write-Host "[解決策 2] -allow-unsupported-compiler フラグを使用 (リスクあり)" -ForegroundColor Yellow
Write-Host "  警告: この方法は推奨されません。コンパイルエラーや実行時エラーが発生する可能性があります" -ForegroundColor Red
Write-Host "  環境変数を設定:" -ForegroundColor White
Write-Host "     `$env:NVCC_APPEND_FLAGS = '-allow-unsupported-compiler'" -ForegroundColor Cyan
Write-Host "  再インストール試行:" -ForegroundColor White
Write-Host "     py -3.12 -m pip install flash_attn==2.5.8 --no-build-isolation" -ForegroundColor Cyan
Write-Host ""

Write-Host "[解決策 3] Flash Attentionなしで動作 (推奨)" -ForegroundColor Green
Write-Host "  flash_attnはオプショナルな依存関係です" -ForegroundColor White
Write-Host "  インストールされていなくても、標準のattentionで動作します" -ForegroundColor White
Write-Host "  パフォーマンスは若干低下しますが、機能は正常に動作します" -ForegroundColor White
Write-Host ""

Write-Host "[推奨] 解決策1を試すか、解決策3で動作を継続してください" -ForegroundColor Green
Write-Host ""

