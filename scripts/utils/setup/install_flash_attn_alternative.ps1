# Flash Attention 代替インストール方法
# Windowsでのビルドが困難な場合の代替手段

Write-Host "[INFO] Flash Attention 代替インストール方法" -ForegroundColor Green

Write-Host "[METHOD 1] 標準のattentionを使用（推奨）" -ForegroundColor Yellow
Write-Host "  Flash Attentionはオプショナルな依存関係です。" -ForegroundColor White
Write-Host "  インストールされていなくても、標準のattentionで動作します。" -ForegroundColor White
Write-Host "  コードは既にフォールバック機能を実装しています:" -ForegroundColor White
Write-Host "    - models/so8t_attention.py: FLASH_ATTN_AVAILABLE フラグで制御" -ForegroundColor Cyan
Write-Host "    - models/Borea-Phi-3.5-mini-Instruct-Jp/modeling_phi3.py: ImportErrorをキャッチ" -ForegroundColor Cyan

Write-Host ""
Write-Host "[METHOD 2] Linux環境でビルドしてwheelファイルを作成" -ForegroundColor Yellow
Write-Host "  1. Linux環境（WSL2推奨）で以下を実行:" -ForegroundColor White
Write-Host "     pip install flash_attn==2.5.8" -ForegroundColor Cyan
Write-Host "  2. ビルドされたwheelファイルをWindowsにコピー" -ForegroundColor White
Write-Host "  3. Windowsでwheelファイルをインストール:" -ForegroundColor White
Write-Host "     pip install flash_attn-2.5.8-*.whl" -ForegroundColor Cyan

Write-Host ""
Write-Host "[METHOD 3] Visual Studio Build ToolsとCUDA Toolkitをインストール" -ForegroundColor Yellow
Write-Host "  1. Visual Studio Build Tools 2022をインストール:" -ForegroundColor White
Write-Host "     https://visualstudio.microsoft.com/downloads/" -ForegroundColor Cyan
Write-Host "     - C++ build tools" -ForegroundColor White
Write-Host "     - Windows 10/11 SDK" -ForegroundColor White
Write-Host "  2. CUDA Toolkit 12.1をインストール:" -ForegroundColor White
Write-Host "     https://developer.nvidia.com/cuda-downloads" -ForegroundColor Cyan
Write-Host "  3. 環境変数を設定:" -ForegroundColor White
Write-Host "     CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1" -ForegroundColor Cyan
Write-Host "  4. 再インストール試行:" -ForegroundColor White
Write-Host "     scripts\install_flash_attn_windows.ps1" -ForegroundColor Cyan

Write-Host ""
Write-Host "[METHOD 4] プリビルドwheelを探す（通常は存在しない）" -ForegroundColor Yellow
Write-Host "  pip search flash-attn  # 通常は存在しない" -ForegroundColor Cyan

Write-Host ""
Write-Host "[推奨] Flash Attentionなしで動作確認" -ForegroundColor Green
Write-Host "  コードは既にFlash Attentionなしで動作するように実装されています。" -ForegroundColor White
Write-Host "  パフォーマンスは若干低下しますが、機能は正常に動作します。" -ForegroundColor White







