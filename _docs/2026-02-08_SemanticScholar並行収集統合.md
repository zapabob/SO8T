# 2026-02-08 Semantic Scholar 並行収集統合

## 変更概要

`semanticscholar_fetcher.py` をパイプラインのデータ収集フェーズに統合し、Arxiv/BioRxiv の収集と同時に並行実行されるように変更しました。

## 実装詳細

- **ファイル**: `src/infrastructure/pipeline/integrated_moonshot_pipeline_2025_2026.py`
- **メソッド**: `collect_new_datasets`
- **変更内容**:
  - `subprocess.Popen` を使用して、`process_arxiv_biorxiv.py` と `semanticscholar_fetcher.py` を同時に実行するようにリファクタリングしました。
  - 両方のプロセスが終了するまで待機し、生成されたデータを一括で回収します。
  - 取得効率の向上のため、既存データのスキップ（`SO8T_SKIP_EXISTING=1`）をデフォルトで維持しています。

## 検証方法

1. パイプラインを実行し、ログに `[ARXIV] Launching parallel collection...` と `[SEMANTIC_SCHOLAR] Launching parallel collection...` が同時に表示されることを確認。
2. 両方のスクリプトが独立して進行し、最終的に `collected` ディレクトリに統合されることを確認。
