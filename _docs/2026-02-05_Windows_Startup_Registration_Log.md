# 2026-02-05 Windows スタートアップ自動実行設定ログ

## 概要

AEGIS-v3.0 の全自動継続運転スクリプトを Windows のスタートアップに登録し、電源投入時の完全自動起動を有効化しました。

## 実施内容

### スタートアップへの登録

- **ショートカット名**: `AEGIS_AutoStart.lnk`
- **保存先**: `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup`
- **実行コマンド**:
  `powershell.exe -ExecutionPolicy Bypass -WindowStyle Minimized -File "C:\Users\downl\Desktop\SO8T\scripts\pipeline\run_aegis_continuous.ps1"`
- **設定の工夫**:
  - `-WindowStyle Minimized`: 起動時に作業を妨げないよう、ウィンドウを最小化した状態で開始。
  - **作業ディレクトリ**: `C:\Users\downl\Desktop\SO8T` に固定し、パス解決の不整合を防止。

## 検証結果

- `Test-Path` による配置確認: `True` (正常)
- これにより、ユーザーが手動でスクリプトを実行することなく、PC の電源を入れるだけで最新のチェックポイントから推論・学習・研究パイプラインが自動再開されます。
