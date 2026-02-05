# 2026-02-06 AEGIS-v3.0 自動継続運転システム強化ログ

## 概要

5分間隔ローリングチェックポイント（3世代）と電源投入時自動再開機能を強化し、PowerShell による進捗管理・エラーログ表示を実装しました。

## 変更ファイル

### 1. `scripts/pipeline/run_aegis_continuous.ps1`

**主な機能強化**:

- **5分間隔ローリングチェックポイント**: 環境変数 `SO8T_CHECKPOINT_INTERVAL=300` で制御
- **3世代ローリングストック**: 環境変数 `SO8T_CHECKPOINT_ROLLING=3` で制御
- **リアルタイム進捗バー**: 5秒間隔で進捗ファイルを監視し、視覚的なプログレスバーを表示
- **チェックポイント通知**: 新しいチェックポイントが保存されるたびにコンソールに通知
- **エラーログ表示**: クラッシュ時に直近5件のエラーを色付きで表示
- **ターミナルタイトル更新**: 実行状態をウィンドウタイトルに反映

### 2. `scripts/pipeline/aegis_startup.bat`

Windows スタートアップ用のランチャースクリプト。電源投入時に自動でパイプラインを開始します。

## Windows スタートアップ登録手順

1. `Win + R` で「ファイル名を指定して実行」を開く
2. `shell:startup` と入力して Enter
3. 開いたフォルダに `aegis_startup.bat` のショートカットを作成

```powershell
# または PowerShell で自動登録
$StartupFolder = [Environment]::GetFolderPath('Startup')
$ShortcutPath = Join-Path $StartupFolder "AEGIS-v3.0.lnk"
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = "C:\Users\downl\Desktop\SO8T\scripts\pipeline\aegis_startup.bat"
$Shortcut.WorkingDirectory = "C:\Users\downl\Desktop\SO8T"
$Shortcut.Description = "AEGIS-v3.0 Auto-Resume Pipeline"
$Shortcut.Save()
```

## 確認事項

- [x] 5分間隔チェックポイント設定
- [x] 3世代ローリングストック設定
- [x] 進捗バー表示機能
- [x] エラーログ表示機能
- [x] Windows スタートアップランチャー作成
