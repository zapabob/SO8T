# PDF構造化・引用データセット サンセットパイプライン統合

- **日付**: 2026-01-29
- **機能名**: PDF構造化・Arxiv/BioRxiv引用データセット統合
- **実装者**: AI Agent

## 概要

3つのPDFファイルを構造化データに変換し、Arxiv/BioRxivからの引用上位論文データセットと共にサンセットパイプラインに統合した。

## 実装内容

### 1. PDF抽出スクリプト

**ファイル**: `scripts/data_processing/pdf_extractor.py`

- PyMuPDFを使用した高品質テキスト抽出
- 構造認識（見出し、段落、ページ）
- メタデータ抽出
- DeepSeek-GLPO形式JSONL出力対応

**抽出結果**:
| ファイル | ページ数 | 語数 | サイズ |
|----------|----------|------|--------|
| r7nd-year_plan.pdf | 49 | 1,932 | 756KB |
| goal05.pdf | 34 | 1,423 | 511KB |
| R07zenpen.pdf | 546 | 63,637 | 20.7MB |

### 2. 引用データフェッチャー

**ファイル**: `scripts/data_processing/citation_fetcher.py`

- Semantic Scholar API統合
- 引用数降順ソート
- チェックポイント機能（中断・再開対応）
- レート制限対応（3秒間隔）

**取得結果**:

- Arxiv: 100件（API timeout発生）
- BioRxiv: 1件（API timeout発生）

> [!NOTE]
> 10万件の取得にはSemanticScholar APIキーの取得と長時間実行が必要です。
> 現行スクリプトはチェックポイント対応済みで中断・再開が可能です。

### 3. データセット統合

**ファイル**: `scripts/data_processing/dataset_integrator.py`

**出力**: `data/sunset_pipeline/processed/combined_training_dataset.jsonl`

- 104件のトレーニングアイテム
- PDF文書: 3件
- 論文データ: 101件

## ディレクトリ構造

```
data/sunset_pipeline/
├── raw/
│   ├── pdfs/
│   │   ├── r7nd-year_plan.json
│   │   ├── goal05.json
│   │   └── R07zenpen.json
│   ├── arxiv_citations/
│   │   └── arxiv_top_100k_2024-2026.jsonl
│   └── biorxiv_citations/
│       └── biorxiv_top_100k_2024-2026.jsonl
└── processed/
    └── combined_training_dataset.jsonl
```

## 実行コマンド

```powershell
# PDF抽出
py -3.12 scripts/data_processing/pdf_extractor.py --input "path/to.pdf" --output "output.json" -v

# 引用データ取得（バックグラウンド実行推奨）
py -3.12 scripts/data_processing/citation_fetcher.py --source arxiv --max-papers 100000 --output data/sunset_pipeline/raw/arxiv_citations/arxiv.jsonl -v

# データ統合
py -3.12 scripts/data_processing/dataset_integrator.py -v
```

## 次のステップ

1. **API制限対応**: Semantic Scholar APIキーを取得して10万件取得を再実行
2. **サンセットパイプライン実行**: `py -3.12 scripts/run_sunset_pipeline.py --phase data`
