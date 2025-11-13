# WSL2 パスワードリセットスクリプト
# Usage: .\scripts\utils\setup\reset_wsl2_password.ps1

Write-Host "[INFO] WSL2 Password Reset Guide" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "[METHOD 1] WSL2パスワードをリセット（推奨）" -ForegroundColor Yellow
Write-Host "  PowerShell (管理者)で実行:" -ForegroundColor White
Write-Host ""
Write-Host "  # デフォルトのWSLディストリビューションを確認" -ForegroundColor Cyan
Write-Host "  wsl --list --verbose" -ForegroundColor Green
Write-Host ""
Write-Host "  # デフォルトユーザーをrootに変更（一時的）" -ForegroundColor Cyan
Write-Host "  ubuntu config --default-user root" -ForegroundColor Green
Write-Host "  # または" -ForegroundColor White
Write-Host "  ubuntu2004 config --default-user root" -ForegroundColor Green
Write-Host ""
Write-Host "  # WSL2を起動してパスワードをリセット" -ForegroundColor Cyan
Write-Host "  wsl" -ForegroundColor Green
Write-Host "  passwd <username>  # ユーザー名を指定" -ForegroundColor Green
Write-Host "  # または" -ForegroundColor White
Write-Host "  passwd  # 現在のユーザー" -ForegroundColor Green
Write-Host ""
Write-Host "  # デフォルトユーザーを元に戻す" -ForegroundColor Cyan
Write-Host "  exit" -ForegroundColor Green
Write-Host "  ubuntu config --default-user <username>" -ForegroundColor Green
Write-Host ""

Write-Host "[METHOD 2] sudoなしで実行できるように設定" -ForegroundColor Yellow
Write-Host "  WSL2環境で実行:" -ForegroundColor White
Write-Host ""
Write-Host "  # sudoersファイルを編集" -ForegroundColor Cyan
Write-Host "  wsl" -ForegroundColor Green
Write-Host "  sudo visudo" -ForegroundColor Green
Write-Host "  # 以下を追加:" -ForegroundColor White
Write-Host "  # <username> ALL=(ALL) NOPASSWD: ALL" -ForegroundColor Green
Write-Host ""

Write-Host "[METHOD 3] パスワードなしでpipをインストール" -ForegroundColor Yellow
Write-Host "  WSL2環境で実行:" -ForegroundColor White
Write-Host ""
Write-Host "  # ユーザー権限でpipをインストール（--userフラグ使用）" -ForegroundColor Cyan
Write-Host "  wsl" -ForegroundColor Green
Write-Host "  python3 -m ensurepip --user" -ForegroundColor Green
Write-Host "  python3 -m pip install --user --upgrade pip" -ForegroundColor Green
Write-Host ""

Write-Host "[INFO] 最も簡単な方法:" -ForegroundColor Cyan
Write-Host "  1. WSL2を起動: wsl" -ForegroundColor White
Write-Host "  2. パスワードをリセット: passwd" -ForegroundColor White
Write-Host "  3. 新しいパスワードを設定" -ForegroundColor White








