# 引き継ぎメタプロンプト: 改良型ムーンショットパイプライン (SO8T) 強化フェーズ

## 1. プロジェクト概要

**プロジェクト名**: SO8T - Enhanced Moonshot Pipeline (AEGIS v3.0)
**主な目的**: 自己内省 (Inner Monologue) と四重推論 (Quadrality Reasoning) を備えた進化型AIモデルの全自動学習パイプラインの構築。
**主要技術**: Unsloth (RTX 3060 12GB最適化), Sakana AI Scientist/OSINT集成エージェント, GRPO (Group Relative Policy Optimization).

## 2. 実装完了した主要機能 (堅牢性強化)

### A. 堅牢なチェックポイント管理 (`src/utils/checkpoint_manager.py`)

- **RollingCheckpointManager**: 5分間隔で保存し、最新の3世代のみを保持（容量節約）。
- **EmergencyCheckpointManager**: `SIGINT` (Ctrl+C), `SIGTERM`, `SIGBREAK` (Windows) を検知し、バイナリ停止直前に強制セーブ。
- **Shared Callback**: `RollingCheckpointCallback` を実装し、Phase 5 および既存の Unsloth 訓練コードで共通利用。

### B. 常時監視・自動復旧スクリプト (`ops/`)

- **`monitor_pipeline.ps1`**: プロセス状況、CPU使用率、SFT/GRPOの進捗、エラーログをカラー表示するダッシュボード。
- **`start_pipeline_persistent.ps1`**: パイプラインがクラッシュまたは再起動した際に、無限ループで自動再開する永続化スクリプト。
- **Startup登録**: `shell:startup` への登録により「電源投入時自動復旧」をサポート。

### C. Sakana AI エージェント統合 (`src/data/phase4_data_enrichment_pipeline.py`)

- Phase 4 に Sakana AI Scientist と OSINT エージェントを統合。
- 科学的推論とOSINT分析の高品質な思考トレース（SO8Tタグ付き）を自動生成。

## 3. 現在のパイプライン状況

- **Phase 1-2**: 完了。データ収集・クレンジング。
- **Phase 3**: **実行中** (Full Retraining)。Unslothによる Borea -> AEGIS v3.0 への学習。
- **Phase 4**: **準備完了** (Sakana AI統合済み)。Phase 3完了後に実行。
- **Phase 5**: **堅牢性統合済み**。自動再学習プロセスに `RollingCheckpointManager` を適用。
- **Phase 6**: **未着手**。統計的ベンチマーク（A/B/Cテスト）。

## 4. 注意事項と解決済み課題

- **インポートエラー**: `ModuleNotFoundError: No module named 'src.utils'` が発生していたが、`train_unsloth_so8t.py` の冒頭で `sys.path` を修正して解決済み。
- **リソース制約**: RTX 3060 (12GB) を使用しているため、常に `Unsloth` の `4-bit` または `8-bit` 量子化を前提とする。
- **OSINT YAML**: `src/infrastructure/config/osint_sources.yaml` にソース設定がある。

## 5. 次のエージェントへの指示

1. **監視**: `.\ops\monitor_pipeline.ps1` で現在の Phase 3 学習の進捗を確認せよ。
2. **Phase 4 実行**: Phase 3 が正常終了（またはチェックポイントが十分）したら、`python src/run_aegis_pipeline.py --phase 4` を実行し、Sakana AI によるデータ拡充を行え。
3. **ベンチマーク**: Phase 5 完了後、Phase 6 の統計的ベンチマークを実行し、モデルの進化を定量評価せよ。
