#Requires -Version 5.1
<#
.SYNOPSIS
    SuperGemma4 llama-server デスクトップショートカット作成
.DESCRIPTION
    1. SuperGemma4 llama-server (RTX3060).lnk  - サーバー起動
    2. SuperGemma4 API Docs.url               - ブラウザでAPIドキュメントを開く
#>

$DESKTOP     = [System.Environment]::GetFolderPath("Desktop")
$LAUNCH_PS1  = "C:\Users\downl\Desktop\SO8T\scripts\start-supergemma-server.ps1"
$PWSH        = "powershell.exe"
$WScript     = New-Object -ComObject WScript.Shell

# ── ショートカット 1: llama-server 起動 ───────────────────────────────────────
$LNK1 = Join-Path $DESKTOP "SuperGemma4 llama-server (RTX3060).lnk"
$SC1  = $WScript.CreateShortcut($LNK1)
$SC1.TargetPath       = $PWSH
$SC1.Arguments        = "-ExecutionPolicy Bypass -File `"$LAUNCH_PS1`""
$SC1.WorkingDirectory = "C:\Users\downl\Desktop\SO8T"
$SC1.Description      = "SuperGemma4 E4B llama-server on RTX 3060 — OpenAI API on http://127.0.0.1:8080/v1"
# llama.cpp icon があれば使う (フォールバック: powershell icon)
$IconPath = "C:\Users\downl\AppData\Local\Programs\llama-turboquant\bin\llama-server.exe"
if (Test-Path $IconPath) {
    $SC1.IconLocation = "$IconPath,0"
} else {
    $SC1.IconLocation = "$PWSH,0"
}
$SC1.WindowStyle = 1  # 通常ウィンドウ (ログを見やすく)
$SC1.Save()
Write-Host "[OK] ショートカット作成: $LNK1" -ForegroundColor Green

# ── ショートカット 2: API ドキュメント (OpenAI swagger UI) ─────────────────────
$URL2 = Join-Path $DESKTOP "SuperGemma4 API Docs (localhost).url"
$SC2  = $WScript.CreateShortcut($URL2)
$SC2.TargetPath = "http://127.0.0.1:8080/docs"
$SC2.Save()
Write-Host "[OK] ショートカット作成: $URL2" -ForegroundColor Green

Write-Host "`n[DONE] デスクトップショートカット作成完了" -ForegroundColor Magenta
Write-Host "  1. SuperGemma4 llama-server (RTX3060).lnk" -ForegroundColor Cyan
Write-Host "  2. SuperGemma4 API Docs (localhost).url" -ForegroundColor Cyan
