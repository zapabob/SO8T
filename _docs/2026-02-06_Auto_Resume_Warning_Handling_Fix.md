# 2026-02-06 自動再開時の警告ハンドリング修正ログ

## 概要

電源投入時の自動再開スクリプト（`run_aegis_continuous.ps1`）が、Python の `FutureWarning` を致命的なエラーと誤認してクラッシュする問題を修正しました。

## 発生していた問題

- **現象**: PC 再起動後の自動実行時、ライブラリ（torch / pynvml）の警告が `stderr` に出力されると、PowerShell の `$ErrorActionPreference = "Stop"` 設定によりスクリプトが即座に終了していた。
- **原因**: PowerShell が `stderr` への出力を「コマンドの失敗」として厳格に扱っていたため。

## 修正内容

### PowerShell スクリプト (`run_aegis_continuous.ps1`)

1. **警告の抑制**: Python 実行前に `$env:PYTHONWARNINGS = "ignore"` を設定し、非推奨警告などが `stderr` に出ないように調整。
2. **エラー判定の緩和**: Python スクリプトを実行するブロックのみ、`$ErrorActionPreference` を一時的に `"Continue"` に変更。これにより、警告が出力されてもスクリプトの実行を継続。
3. **終了コードの確認**: 警告は無視しつつ、実際のクラッシュ（`$LASTEXITCODE` が 0 以外）は引き続き検知してリトライループを回すように維持。

## 検証結果

- スクリプトが `FutureWarning` で停止せず、メインのパイプライン処理へと正常に移行することを確認。
- `logs/pipeline_continuous_*.log` に処理が継続されていることが記録されている。
