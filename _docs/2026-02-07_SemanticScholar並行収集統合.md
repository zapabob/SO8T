# 2026-02-07_SemanticScholar並行収集統合

## 概要

Semantic Scholar API を使用して論文データを自動収集する機能をパイプラインに統合しました。これにより、Arxiv/BioRxiv に加えて Semantic Scholar からも最新の論文データを収集し、VSSI（四重推論）形式で学習データに組み込むことが可能になります。

## 変更内容

### 1. 収集スクリプトの新規作成 [NEW]

- `src/data/collection/semanticscholar_fetcher.py`
  - Semantic Scholar Bulk Search API を使用した高速な論文収集。
  - 取得したアブストラクトを VSSI（Task, Analysis, Safety, Policy）形式の `<think>` タグ付きデータに変換。
  - レート制限や API キー（環境変数 `SEMANTIC_SCHOLAR_API_KEY`）に対応。

### 2. パイプラインへの統合 [MODIFY]

- `src/infrastructure/pipeline/integrated_moonshot_pipeline_2025_2026.py`
  - `collect_new_datasets` フェーズに Semantic Scholar の収集ステップを追加。
  - 環境変数で収集の有効化、クエリ、件数を制御可能。

## 追加された環境変数

`example.env` に以下の項目を追加することを推奨します：

- `SO8T_COLLECT_SEMANTIC_SCHOLAR`: 収集の有効化 (1/0)
- `SO8T_SS_QUERY`: 検索クエリ
- `SO8T_SS_COUNT`: 最大収集件数
- `SEMANTIC_SCHOLAR_API_KEY`: API キー（オプション）

## 検証結果

- 単体実行テストを行い、`data/collected_2025_2026/new_collected/test_ss.jsonl` に VSSI 形式のデータが正しく生成されることを確認しました。
