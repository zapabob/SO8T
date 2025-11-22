# scripts/data/ ディレクトリ

## 概要

このディレクトリには、データ収集・スクレイピング・データ処理に関するスクリプトが含まれています。

## 主要スクリプト

### 並列DeepResearch Webスクレイピング
- `parallel_deep_research_scraping.py`: メインの並列DeepResearch Webスクレイピングスクリプト
  - Gemini統合対応（Computer Use Preview方式）
  - 日経225企業スクレイピング対応
  - ウィキペディアMediaWiki API統合
  - 全自動ラベル付け・4値分類・四重推論・データセット加工

### データ収集スクリプト
- `collect_training_data_with_playwright.py`: Playwrightベース学習用データ収集
- `collect_domain_knowledge_with_playwright.py`: ドメイン別知識サイトスクレイピング
- `collect_japanese_training_dataset.py`: 日本語学習用データセット収集
- `collect_drug_pharmaceutical_detection_dataset.py`: 違法薬物検知用データセット収集

### 特化スクレイピングスクリプト
- `drug_detection_deepresearch_scraping.py`: 違法薬物検知目的DeepResearch Webスクレイピング
- `nikkei225_deepresearch_scraping.py`: 日経225企業DeepResearch Webスクレイピング（統合済み）
- `human_like_web_scraping.py`: 人間を模倣したWebスクレイピング
- `so8t_controlled_browser_scraper.py`: SO8T統制ブラウザスクレイパー
- `so8t_thinking_controlled_scraping.py`: SO8T思考統制スクレイピング

### データ処理スクリプト
- `clean_japanese_dataset.py`: 日本語データセットのクレンジング
- `split_dataset.py`: データセットの分割
- `label_four_class_dataset.py`: 4値分類ラベル付け
- `label_nsfw_content.py`: NSFWコンテンツラベル付け
- `convert_to_quadruple_json.py`: 四重推論JSON形式への変換

### ユーティリティスクリプト
- `browser_coordinator.py`: ブラウザ間協調通信
- `parallel_tab_processor.py`: 並列タブ処理
- `parallel_pipeline_manager.py`: 並列パイプライン管理
- `crawler_error_handler.py`: クローラーエラーハンドラー
- `crawler_health_monitor.py`: クローラーヘルスモニター
- `retry_handler.py`: リトライハンドラー

### データ生成スクリプト
- `generation/`: 合成データ生成スクリプト
  - `generate_synthetic_data.py`: 合成データ生成
  - `generate_synthetic_japanese.py`: 日本語合成データ生成
  - `dataset_synth.py`: データセット合成
  - `prepare_data.py`: データ準備

## 実行方法

### 並列DeepResearch Webスクレイピング
```bash
# 基本実行
py -3 scripts\data\parallel_deep_research_scraping.py --output D:\webdataset\processed

# Gemini統合を使用
set GEMINI_API_KEY=your_api_key_here
py -3 scripts\data\parallel_deep_research_scraping.py --output D:\webdataset\processed --use-gemini --gemini-model gemini-2.0-flash-exp
```

### バッチスクリプト
- `run_parallel_deep_research_scraping.bat`: 並列DeepResearch Webスクレイピング実行
- `run_nikkei225_deepresearch_scraping.bat`: 日経225企業スクレイピング実行
- `run_drug_detection_deepresearch_scraping.bat`: 違法薬物検知スクレイピング実行

## 注意事項

- 各スクリプトは特定の目的で使用されるため、削除せずに維持
- 機能が重複しているスクリプトも、それぞれ異なる用途があるため統合は慎重に実施
- 新しいスクリプトを追加する場合は、このREADMEを更新





















































































































