# scripts/ ディレクトリ

## 概要

このディレクトリには、SO8Tプロジェクトの各種スクリプトが機能別に整理されています。

## ディレクトリ構造

### agents/
AIエージェント実装
- `unified_ai_agent.py`: 統合AIエージェント
- `scraping_reasoning_agent.py`: スクレイピング推論エージェント
- `domain_knowledge_integrator.py`: ドメイン知識統合エージェント

### api/
APIサーバースクリプト
- `serve_fastapi.py`: FastAPIサーバー
- `serve_think_api.py`: 思考APIサーバー
- `unified_agent_api.py`: 統合エージェントAPI

### audit/
監査ログスクリプト
- `scraping_audit_logger.py`: スクレイピング監査ロガー

### conversion/
モデル変換スクリプト
- `convert_*.py`: 各種モデル変換スクリプト
- `integrate_*.py`: モデル統合スクリプト
- `quantize_*.py`: 量子化スクリプト

### dashboard/
ダッシュボードスクリプト
- `so8t_scraping_dashboard.py`: スクレイピングダッシュボード
- `so8t_training_dashboard.py`: 訓練ダッシュボード
- `unified_pipeline_dashboard.py`: 統合パイプラインダッシュボード

### data/
データ収集・スクレイピング・データ処理スクリプト
詳細は `data/README.md` を参照

### evaluation/
評価スクリプト
- `evaluate_*.py`: 各種評価スクリプト
- `ab_test_*.py`: A/Bテストスクリプト
- `reports/`: レポート生成スクリプト
- `visualization/`: 可視化スクリプト

### inference/
推論スクリプト
- `infer.py`: 推論スクリプト
- `demo_*.py`: デモスクリプト
- `ollama/`: Ollama統合スクリプト
- `tests/`: テストスクリプト

### pipelines/
パイプラインスクリプト
- `complete_*.py`: 完全パイプラインスクリプト
- `run_*_pipeline.py`: パイプライン実行スクリプト

### training/
訓練スクリプト
- `train_*.py`: 各種訓練スクリプト
- `finetune_*.py`: ファインチューニングスクリプト
- `burnin_*.py`: バーンインスクリプト

### testing/
テストスクリプト
- `test_*.py`: 各種テストスクリプト
- `run_*_test.bat`: テスト実行バッチスクリプト

### utils/
ユーティリティスクリプト
- `audio/`: 音声通知スクリプト
- `setup/`: セットアップスクリプト（Flash Attention等）
- `notebooks/`: Jupyter Notebookファイル

## 整理方針

- 機能別にディレクトリを整理
- 各スクリプトは特定の目的で使用されるため、削除せずに維持
- 重複スクリプトも、それぞれ異なる用途があるため統合は慎重に実施
