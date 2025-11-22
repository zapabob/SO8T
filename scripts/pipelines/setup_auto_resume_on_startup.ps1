# NSFW・違法薬物検知目的QLoRA学習用データ生成全自動パイプライン
# 電源投入時の自動再開設定スクリプト

# UTF-8エンコーディング設定
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "NSFW・違法薬物検知目的QLoRA学習用データ生成全自動パイプライン" -ForegroundColor Cyan
Write-Host "電源投入時の自動再開設定" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# プロジェクトルートを取得
$projectRoot = Split-Path -Parent $PSScriptRoot | Split-Path -Parent
$batFile = Join-Path $projectRoot "scripts\pipelines\run_nsfw_drug_detection_qlora_training_data_pipeline.bat"

if (-not (Test-Path $batFile)) {
    Write-Host "[ERROR] バッチファイルが見つかりません: $batFile" -ForegroundColor Red
    exit 1
}

Write-Host "[INFO] バッチファイル: $batFile" -ForegroundColor Green
Write-Host ""

# タスクスケジューラに登録するか確認
Write-Host "Windowsタスクスケジューラに登録しますか？" -ForegroundColor Yellow
Write-Host "  1. ログオン時に自動実行（推奨）" -ForegroundColor White
Write-Host "  2. システム起動時に自動実行" -ForegroundColor White
Write-Host "  3. スタートアップフォルダにショートカットを作成" -ForegroundColor White
Write-Host "  4. キャンセル" -ForegroundColor White
Write-Host ""
$choice = Read-Host "選択 (1-4)"

switch ($choice) {
    "1" {
        # ログオン時に自動実行
        Write-Host "[INFO] ログオン時に自動実行するタスクを作成中..." -ForegroundColor Green
        
        $taskName = "NSFW-Drug-Detection-QLoRA-Training-Data-Pipeline-Auto-Resume"
        $taskDescription = "NSFW・違法薬物検知目的QLoRA学習用データ生成全自動パイプライン（電源投入時自動再開）"
        
        # 既存のタスクを削除（存在する場合）
        $existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
        if ($existingTask) {
            Write-Host "[INFO] 既存のタスクを削除中..." -ForegroundColor Yellow
            Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
        }
        
        # タスクアクションを作成
        $action = New-ScheduledTaskAction -Execute $batFile -WorkingDirectory $projectRoot
        
        # タスクトリガーを作成（ログオン時）
        $trigger = New-ScheduledTaskTrigger -AtLogOn
        
        # タスク設定を作成
        $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
        
        # タスクを登録
        Register-ScheduledTask -TaskName $taskName -Description $taskDescription -Action $action -Trigger $trigger -Settings $settings -RunLevel Highest
        
        Write-Host "[SUCCESS] タスクスケジューラに登録しました: $taskName" -ForegroundColor Green
        Write-Host "[INFO] 次回ログオン時に自動実行されます" -ForegroundColor Cyan
    }
    "2" {
        # システム起動時に自動実行
        Write-Host "[INFO] システム起動時に自動実行するタスクを作成中..." -ForegroundColor Green
        
        $taskName = "NSFW-Drug-Detection-QLoRA-Training-Data-Pipeline-Auto-Resume-Startup"
        $taskDescription = "NSFW・違法薬物検知目的QLoRA学習用データ生成全自動パイプライン（システム起動時自動再開）"
        
        # 既存のタスクを削除（存在する場合）
        $existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
        if ($existingTask) {
            Write-Host "[INFO] 既存のタスクを削除中..." -ForegroundColor Yellow
            Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
        }
        
        # タスクアクションを作成
        $action = New-ScheduledTaskAction -Execute $batFile -WorkingDirectory $projectRoot
        
        # タスクトリガーを作成（システム起動時、30秒遅延）
        $trigger = New-ScheduledTaskTrigger -AtStartup
        $trigger.Delay = "PT30S"  # 30秒遅延
        
        # タスク設定を作成
        $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
        
        # タスクを登録
        Register-ScheduledTask -TaskName $taskName -Description $taskDescription -Action $action -Trigger $trigger -Settings $settings -RunLevel Highest
        
        Write-Host "[SUCCESS] タスクスケジューラに登録しました: $taskName" -ForegroundColor Green
        Write-Host "[INFO] 次回システム起動時に自動実行されます（30秒遅延）" -ForegroundColor Cyan
    }
    "3" {
        # スタートアップフォルダにショートカットを作成
        Write-Host "[INFO] スタートアップフォルダにショートカットを作成中..." -ForegroundColor Green
        
        $startupFolder = [Environment]::GetFolderPath("Startup")
        $shortcutPath = Join-Path $startupFolder "NSFW-Drug-Detection-QLoRA-Training-Data-Pipeline.lnk"
        
        $shell = New-Object -ComObject WScript.Shell
        $shortcut = $shell.CreateShortcut($shortcutPath)
        $shortcut.TargetPath = $batFile
        $shortcut.WorkingDirectory = $projectRoot
        $shortcut.Description = "NSFW・違法薬物検知目的QLoRA学習用データ生成全自動パイプライン（電源投入時自動再開）"
        $shortcut.Save()
        
        Write-Host "[SUCCESS] ショートカットを作成しました: $shortcutPath" -ForegroundColor Green
        Write-Host "[INFO] 次回ログオン時に自動実行されます" -ForegroundColor Cyan
    }
    "4" {
        Write-Host "[INFO] キャンセルしました" -ForegroundColor Yellow
        exit 0
    }
    default {
        Write-Host "[ERROR] 無効な選択です" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "[INFO] 設定完了" -ForegroundColor Green
Write-Host "[INFO] 電源投入時にパイプラインが自動再開されます" -ForegroundColor Cyan
Write-Host ""

























































































































