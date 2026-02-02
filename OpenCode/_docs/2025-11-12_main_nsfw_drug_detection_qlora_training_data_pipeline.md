# NSFW・違法薬物検知目的QLoRA学習用データ生成全自動パイプライン実装ログ

## 実装情報
- **日付**: 2025-11-12
- **Worktree**: main
- **機能名**: NSFW・違法薬物検知目的QLoRA学習用データ生成全自動パイプライン
- **実装者**: AI Agent

## 実装概要

NSFW、違法薬物データを含む検知目的でのQLoRAでのSO8Tおよびドメイン別知識の学習用データ生成用の全自動パイプラインを実装しました。既存の実装を参考に、データ収集から学習用データセット生成まで全自動で実行します。

**重要**: この実装は検知目的のみで、生成目的ではない。安全判定と拒否挙動の学習を目的とする。

## 実装内容

### 1. 全自動パイプラインスクリプトの作成

**ファイル**: `scripts/pipelines/nsfw_drug_detection_qlora_training_data_pipeline.py`

**実装状況**: [実装済み]  
**動作確認**: [要確認]  
**確認日時**: 2025-11-12  
**備考**: 全自動パイプラインスクリプトを作成

**主要機能**:
- Phase 1: NSFW検知データセット収集（`collect_nsfw_detection_dataset.py`を使用）
- Phase 2: 違法薬物検知データセット収集（`collect_drug_pharmaceutical_detection_dataset.py`を使用）
- Phase 3: ドメイン別知識データセット収集（`collect_domain_knowledge_with_playwright.py`を使用）
- Phase 4: 全データセットの統合
- Phase 5: QuadrupleClassifierの初期化
- Phase 6: 4値分類・四重推論の実行
- Phase 7: 統計的データクレンジングの実行
- Phase 8: QLoRA学習用データセット形式への変換
- Phase 9: 学習用データセットの保存

**主要メソッド**:
- `collect_nsfw_detection_data()`: NSFW検知データセット収集
- `collect_drug_detection_data()`: 違法薬物検知データセット収集
- `collect_domain_knowledge_data()`: ドメイン別知識データセット収集
- `merge_all_datasets()`: 全データセットの統合
- `initialize_quadruple_classifier()`: QuadrupleClassifierの初期化
- `run_quadruple_classification()`: 4値分類・四重推論の実行
- `run_statistical_cleaning()`: 統計的データクレンジングの実行
- `convert_to_qlora_format()`: QLoRA学習用データセット形式への変換
- `save_training_dataset()`: 学習用データセットの保存
- `_statistical_data_cleaning()`: 統計的データクレンジング（重複検出、外れ値検出、品質スコアリング、信頼区間フィルタリング）
- `_calculate_statistical_quality_score()`: 統計的品質スコアの計算
- `_convert_to_training_format()`: 学習用データセット形式への変換

### 2. 設定ファイルの作成

**ファイル**: `configs/nsfw_drug_detection_qlora_training_data_pipeline_config.yaml`

**実装状況**: [実装済み]  
**動作確認**: [要確認]  
**確認日時**: 2025-11-12  
**備考**: 設定ファイルを作成

**設定内容**:
- 基本設定（セッションID形式、チェックポイントディレクトリ、出力ディレクトリ）
- NSFW検知データセット収集設定
- 違法薬物検知データセット収集設定
- ドメイン別知識データセット収集設定
- 4値分類・四重推論設定
- 統計的データクレンジング設定（重複検出、外れ値検出、品質スコアリング、信頼区間フィルタリング）
- QLoRA学習用データセット形式設定

### 3. バッチスクリプトの作成

**ファイル**: `scripts/pipelines/run_nsfw_drug_detection_qlora_training_data_pipeline.bat`

**実装状況**: [実装済み]  
**動作確認**: [要確認]  
**確認日時**: 2025-11-12  
**備考**: パイプライン実行用バッチスクリプトを作成

**機能**:
- UTF-8エンコーディング設定
- 設定ファイルの存在確認
- パイプライン実行
- エラーハンドリング
- 音声通知

### 4. 既存実装の統合

**実装状況**: [実装済み]  
**動作確認**: [要確認]  
**確認日時**: 2025-11-12  
**備考**: 既存の実装を参考に統合

**参考実装**:
- `scripts/data/collect_nsfw_detection_dataset.py`: NSFW検知データセット収集
- `scripts/data/collect_drug_pharmaceutical_detection_dataset.py`: 違法薬物検知データセット収集
- `scripts/data/collect_domain_knowledge_with_playwright.py`: ドメイン別知識データセット収集
- `scripts/data/parallel_deep_research_scraping.py`: 4値分類・四重推論・学習用データセット変換
- `scripts/pipelines/web_scraping_data_pipeline.py`: QuadrupleClassifier
- `scripts/pipelines/unified_master_pipeline.py`: パイプライン実行フレームワーク

**統合方法**:
- 既存のデータ収集スクリプトをサブプロセスとして実行
- `parallel_deep_research_scraping.py`の`process_samples_for_training_dataset()`メソッドを参考に、4値分類・四重推論・学習用データセット変換を実装
- `QuadrupleClassifier`を使用して4値分類・四重推論を実行
- `statistical_data_cleaning()`メソッドを参考に、統計的データクレンジングを実装

### 5. QLoRA学習用データセット形式への変換

**実装状況**: [実装済み]  
**動作確認**: [要確認]  
**確認日時**: 2025-11-12  
**備考**: instruction-output形式への変換を実装

**実装内容**:
- instruction-output形式への変換
- 四重推論結果（Task/Safety/Policy/Final）を含む形式
- 4値分類ラベル（ALLOW/ESCALATION/DENY/REFUSE）を含む形式
- メタデータ（カテゴリ、言語、URL等）を含む形式

**参考実装**:
- `scripts/data/parallel_deep_research_scraping.py`の`_convert_to_training_format()`メソッド

### 6. 統計的データクレンジングの実装

**実装状況**: [実装済み]  
**動作確認**: [要確認]  
**確認日時**: 2025-11-12  
**備考**: 統計的に有意な手法でデータクレンジングを実装

**実装内容**:
- 重複検出（ハッシュベース + 類似度ベース）
- 外れ値検出（Z-score、IQR）
- 統計的品質スコアリング
- 信頼区間フィルタリング

**参考実装**:
- `scripts/data/parallel_deep_research_scraping.py`の`statistical_data_cleaning()`メソッド

### 7. チェックポイント機能の実装

**実装状況**: [実装済み]  
**動作確認**: [要確認]  
**確認日時**: 2025-11-12  
**備考**: チェックポイント機能を実装

**実装内容**:
- 各フェーズの完了時にチェックポイントを保存
- 中断時は最後のチェックポイントから再開
- 進捗状況の記録

**参考実装**:
- `scripts/pipelines/unified_master_pipeline.py`のチェックポイント機能

### 8. ログ出力と進捗表示

**実装状況**: [実装済み]  
**動作確認**: [要確認]  
**確認日時**: 2025-11-12  
**備考**: ログ出力と進捗表示を実装

**実装内容**:
- 各フェーズの進捗をログ出力
- tqdmを使用した進捗バーの表示（利用可能な場合）
- 統計情報の出力（サンプル数、カテゴリ分布、言語分布等）

### 9. エラーハンドリング

**実装状況**: [実装済み]  
**動作確認**: [要確認]  
**確認日時**: 2025-11-12  
**備考**: エラーハンドリングを実装

**実装内容**:
- 各フェーズのエラーハンドリング
- リトライロジック（必要に応じて）
- エラーログの記録
- シグナルハンドラー（SIGINT、SIGTERM、SIGBREAK）

## 作成・変更ファイル

### 新規作成
- `scripts/pipelines/nsfw_drug_detection_qlora_training_data_pipeline.py`: 全自動パイプラインスクリプト
- `configs/nsfw_drug_detection_qlora_training_data_pipeline_config.yaml`: 設定ファイル
- `scripts/pipelines/run_nsfw_drug_detection_qlora_training_data_pipeline.bat`: バッチスクリプト
- `_docs/2025-11-12_main_nsfw_drug_detection_qlora_training_data_pipeline.md`: 実装ログ

## 設計判断

1. **既存実装の統合**: 既存のデータ収集スクリプトをサブプロセスとして実行し、機能を維持
2. **統計的データクレンジング**: 重複検出、外れ値検出、品質スコアリング、信頼区間フィルタリングを実装
3. **チェックポイント機能**: 各フェーズの完了時にチェックポイントを保存し、中断時の再開を可能にする
4. **エラーハンドリング**: 各フェーズでエラーハンドリングを実装し、エラー時もチェックポイントを保存
5. **QLoRA学習用データセット形式**: instruction-output形式で、四重推論結果と4値分類ラベルを含む

## 電源投入時の自動再開機能

**実装状況**: [実装済み]  
**動作確認**: [要確認]  
**確認日時**: 2025-11-12  
**備考**: 電源断からの自動再開機能を実装

**実装内容**:
- チェックポイントの自動検出機能（`_load_checkpoint()`の強化）
- 最新のチェックポイントを自動検出
- チェックポイントの整合性確認
- 中断されたフェーズから自動再開（`_determine_resume_phase()`）
- 完了済みフェーズのスキップ機能
- Windowsタスクスケジューラ用の設定スクリプト（`setup_auto_resume_on_startup.ps1`）

**使用方法**:
1. **自動再開設定**:
   ```powershell
   powershell -ExecutionPolicy Bypass -File "scripts\pipelines\setup_auto_resume_on_startup.ps1"
   ```
   - ログオン時に自動実行（推奨）
   - システム起動時に自動実行
   - スタートアップフォルダにショートカットを作成

2. **手動実行**:
   ```batch
   scripts\pipelines\run_nsfw_drug_detection_qlora_training_data_pipeline.bat
   ```
   - チェックポイントが存在する場合は自動再開
   - `--no-auto-resume`オプションで新規実行も可能

**動作仕様**:
- 電源断時は各フェーズの完了時にチェックポイントを保存
- 電源投入時は最新のチェックポイントを自動検出
- 中断されたフェーズから自動再開
- 完了済みフェーズはスキップして効率的に再開

## テスト結果

**テスト状況**: [未実施]  
**テスト日時**: -  
**備考**: パイプラインの実行テストは未実施

## 運用注意事項

### データ収集ポリシー
- 利用条件を守りつつ、高信頼ソースとして優先使用
- robots.txt遵守を徹底
- 個人情報・機密情報の除外を徹底

### NSFWコーパス運用
- **主目的**: 安全判定と拒否挙動の学習（生成目的ではない）
- モデル設計とドキュメントに明記
- 分類器は検出・拒否用途のみ

### 違法薬物データ運用
- **主目的**: 安全判定と拒否挙動の学習（生成目的ではない）
- 合法・倫理的なソースのみを使用（PMDA、GoV、WHO、UNODC、EMCDDA、Wikipedia等）
- 検知・防止・教育を目的とする

### /thinkエンドポイント運用
- 四重Thinking部（`<think-*>`）は外部非公開を徹底
- `<final>`のみ返す実装を維持
- 監査ログでThinkingハッシュを記録（内容は非公開）

### パイプライン実行
- 長時間実行される可能性があるため、チェックポイント機能を活用
- 各フェーズのタイムアウト設定を適切に設定
- エラー時はチェックポイントから再開可能


