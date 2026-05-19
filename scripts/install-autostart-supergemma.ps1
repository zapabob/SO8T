#Requires -Version 5.1
<#
.SYNOPSIS
    SuperGemma4 llama-server auto-start manager (Startup folder approach)
.DESCRIPTION
    install   : Startup フォルダ (管理者不要) にVBSランチャーを配置
    uninstall : Startup フォルダから削除
    status    : 登録状態 + プロセス確認
.NOTES
    無効化: C:\Users\downl\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup\
    から SuperGemma4-LlamaServer.vbs を削除するか移動する
#>
param(
    [ValidateSet("install","uninstall","status")]
    [string]$Action = "install"
)

$STARTUP_DIR = [System.Environment]::GetFolderPath("Startup")
$VBS_NAME    = "SuperGemma4-LlamaServer.vbs"
$VBS_PATH    = Join-Path $STARTUP_DIR $VBS_NAME
$LAUNCH_PS1  = "C:\Users\downl\Desktop\SO8T\scripts\start-supergemma-server.ps1"

function Write-Info($msg)  { Write-Host "[INFO]  $msg" -ForegroundColor Cyan   }
function Write-OK($msg)    { Write-Host "[OK]    $msg" -ForegroundColor Green  }
function Write-Fail($msg)  { Write-Host "[FAIL]  $msg" -ForegroundColor Red    }

switch ($Action) {
    "install" {
        Write-Host "`n[INSTALL] SuperGemma4 Startup folder 自動起動" -ForegroundColor Magenta

        if (-not (Test-Path $LAUNCH_PS1)) {
            Write-Fail "起動スクリプトが見つかりません: $LAUNCH_PS1"
            exit 1
        }

        # VBS (管理者不要、隠しウィンドウ)
        $VBS = @"
' SuperGemma4 llama-server auto-starter
' Startup folder: $STARTUP_DIR
' To disable: delete this file
Set objShell = CreateObject("WScript.Shell")
objShell.Run "powershell.exe -WindowStyle Hidden -NonInteractive -ExecutionPolicy Bypass -File ""$LAUNCH_PS1""", 0, False
"@
        [System.IO.File]::WriteAllText($VBS_PATH, $VBS, [System.Text.Encoding]::UTF8)
        Write-OK "VBS 配置: $VBS_PATH"
        Write-Info "次回ログオン時に自動起動します (管理者権限不要)"
        Write-Info ""
        Write-Info "無効化方法:"
        Write-Host "  Del `"$VBS_PATH`"" -ForegroundColor DarkCyan
        Write-Info "手動起動:"
        Write-Host "  Start-Process wscript.exe -ArgumentList `"$VBS_PATH`"" -ForegroundColor DarkCyan
    }

    "uninstall" {
        Write-Host "`n[UNINSTALL] SuperGemma4 自動起動削除" -ForegroundColor Yellow
        if (Test-Path $VBS_PATH) {
            Remove-Item $VBS_PATH -Force
            Write-OK "削除完了: $VBS_PATH"
        } else {
            Write-Info "ファイルが存在しません: $VBS_PATH"
        }
    }

    "status" {
        Write-Host "`n[STATUS] SuperGemma4 自動起動状態" -ForegroundColor Cyan
        if (Test-Path $VBS_PATH) {
            Write-OK "自動起動 VBS: $VBS_PATH (登録済)"
        } else {
            Write-Info "自動起動 VBS: 未登録"
        }

        $proc = Get-Process -Name "llama-server" -ErrorAction SilentlyContinue
        if ($proc) {
            Write-OK "llama-server 実行中 (PID: $($proc.Id))"
        } else {
            Write-Info "llama-server は実行されていません"
        }

        Write-Host ""
        Write-Info "VRAM 状態:"
        nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader 2>&1
    }
}
