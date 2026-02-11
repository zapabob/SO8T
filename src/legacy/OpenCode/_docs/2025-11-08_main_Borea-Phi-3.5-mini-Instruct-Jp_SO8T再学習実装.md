# Borea-Phi-3.5-mini-Instruct-Jp SO8T再学習実装ログ

## 実装情報
- **日付**: 2025-11-08
- **Worktree**: main
- **機能名**: Borea-Phi-3.5-mini-Instruct-JpをSO8Tで再学習
- **実装者**: AI Agent

## 実装内容

### 1. SO8T再学習スクリプト作成

**ファイル**: `scripts/training/retrain_borea_phi35_with_so8t.py` (新規作成)

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: Borea-Phi-3.5-mini-Instruct-JpモデルをSO8Tで再学習するスクリプトを実装

- `SO8TRetrainer`クラス: SO8T再学習クラス
- `SO8TTrainingDataset`クラス: SO8T学習用データセット
- `PowerFailureRecovery`クラス: 電源断リカバリーシステム（5分間隔チェックポイント）
- QLoRA 8bit学習対応
- LoRA設定（r=64, alpha=128）
- 四重推論形式対応（use_quadruple_thinking）
- データセット自動分割（train/val/test: 80/10/10）
- チェックポイント自動保存（5分間隔、最大10個）

### 2. 設定ファイル作成

**ファイル**: `configs/retrain_borea_phi35_so8t_config.yaml` (新規作成)

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: Borea-Phi-3.5-mini-Instruct-Jp SO8T再学習用設定ファイルを実装

- データ設定: input_path, max_seq_length, use_quadruple_thinking
- モデル設定: base_model_path（ローカルパス）
- 学習設定: num_train_epochs, batch_size, learning_rate, LoRA設定等
- チェックポイント設定: interval_seconds, max_checkpoints

## 作成・変更ファイル
- `scripts/training/retrain_borea_phi35_with_so8t.py` (新規作成)
- `configs/retrain_borea_phi35_so8t_config.yaml` (新規作成)

## 設計判断

1. **ベースモデル**: Borea-Phi-3.5-mini-Instruct-Jp（ローカルパス）を使用
2. **学習方式**: QLoRA 8bit学習を使用し、メモリ効率を重視
3. **データ形式**: 四値分類データ（four_class_*.jsonl）を使用
4. **四重推論**: 四重推論形式（think-task/think-safety/think-policy/final）に対応
5. **電源断リカバリー**: 5分間隔でチェックポイントを自動保存

## データフロー

```
Borea-Phi-3.5-mini-Instruct-Jp（ローカルモデル）
  ↓
収集・加工済みデータ（four_class_*.jsonl）
  ↓
SO8T再学習（QLoRA 8bit）
  ↓
再学習済みモデル保存（D:/webdataset/checkpoints/so8t_retrained_borea_phi35）
```

## 依存関係

### 既存実装の活用
- ✅ `Borea-Phi-3.5-mini-Instruct-Jp/` - ベースモデル（ローカル）
- ✅ `D:/webdataset/processed/four_class/four_class_*.jsonl` - 収集・加工済みデータ
- ✅ `scripts/training/finetune_borea_japanese.py` - Fine-tuning参考実装

### 外部ライブラリ
- `transformers>=4.43.0`: Hugging Face Transformers（Phi-3対応）
- `peft>=0.6.0`: Parameter-Efficient Fine-Tuning
- `bitsandbytes>=0.41.0`: 8bit量子化
- `torch>=2.0.0`: PyTorch
- `accelerate>=0.31.0`: モデル分散

## 使用方法

### 基本実行

```bash
# 設定ファイルを使用
py scripts/training/retrain_borea_phi35_with_so8t.py \
    --base-model Borea-Phi-3.5-mini-Instruct-Jp \
    --dataset D:/webdataset/processed/four_class/four_class_20251108_035137.jsonl \
    --output D:/webdataset/checkpoints/so8t_retrained_borea_phi35 \
    --config configs/retrain_borea_phi35_so8t_config.yaml
```

### カスタム設定

```bash
# 設定ファイルなしで実行
py scripts/training/retrain_borea_phi35_with_so8t.py \
    --base-model Borea-Phi-3.5-mini-Instruct-Jp \
    --dataset D:/webdataset/processed/four_class/four_class_20251108_035137.jsonl \
    --output D:/webdataset/checkpoints/so8t_retrained_borea_phi35
```

## モデル情報

### Borea-Phi-3.5-mini-Instruct-Jp
- **ベースモデル**: Phi-3.5-mini-Instruct
- **アーキテクチャ**: Phi3ForCausalLM
- **vocab_size**: 32064
- **hidden_size**: 3072
- **num_hidden_layers**: 32
- **num_attention_heads**: 32
- **intermediate_size**: 8192
- **max_position_embeddings**: 131072
- **特徴**: 日本語性能が向上

## テスト計画

1. **データ読み込みテスト**: 四値分類データの読み込み動作確認
2. **モデル読み込みテスト**: Borea-Phi-3.5-mini-Instruct-Jpモデルの読み込み動作確認
3. **学習実行テスト**: SO8T再学習の実行動作確認
4. **チェックポイントテスト**: 電源断リカバリー機能の動作確認
5. **最終モデルテスト**: 再学習済みモデルの保存と読み込み動作確認

## 運用注意事項

### データ収集ポリシー
- 利用条件を守りつつ、高信頼ソースとして優先使用
- robots.txt遵守を徹底
- 個人情報・機密情報の除外を徹底

### NSFWコーパス運用
- **主目的**: 安全判定と拒否挙動の学習（生成目的ではない）
- モデル設計とドキュメントに明記
- 分類器は検出・拒否用途のみ

### /thinkエンドポイント運用
- 四重Thinking部（`<think-*>`）は外部非公開を徹底
- `<final>`のみ返す実装を維持
- 監査ログでThinkingハッシュを記録（内容は非公開）

## 実装内容（追加）

### 3. 動作確認スクリプト作成

**ファイル**: `scripts/training/verify_borea_phi35_retraining.py` (新規作成)

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: モデル読み込み、データセット読み込み、学習設定、推論テスト、チェックポイント確認を実装

- `BoreaPhi35RetrainingVerifier`クラス: 検証クラス
- `VerificationResult`クラス: 検証結果クラス
- `verify_model_loading()`: モデル読み込み確認
- `verify_dataset_loading()`: データセット読み込み確認
- `verify_training_config()`: 学習設定確認
- `verify_inference()`: 推論テスト
- `verify_lora_setup()`: LoRA設定確認
- `verify_checkpoint_save_load()`: チェックポイント保存/読み込み確認
- 検証結果をJSON形式で保存

### 4. パフォーマンス最適化

**ファイル**: `scripts/training/retrain_borea_phi35_with_so8t.py` (拡張)

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: パフォーマンスプロファイリング機能を追加

- `PerformanceProfiler`クラス: パフォーマンスプロファイラー
- `profile_step()`: ステップごとのプロファイリング（メモリ、GPU、時間）
- `record_training_speed()`: 学習速度記録
- `get_summary()`: サマリー取得
- メモリ使用量、GPUメモリ使用量、学習速度の追跡
- パフォーマンスレポート自動生成（JSON形式）

### 5. 評価スクリプト作成

**ファイル**: `scripts/evaluation/evaluate_borea_phi35_so8t_retrained.py` (新規作成)

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: four_classデータ評価とHFベンチマーク評価を実装

- `BoreaPhi35SO8TEvaluator`クラス: 評価クラス
- `FourClassEvaluator`クラス: 四値分類評価クラス
- `HFBenchmarkEvaluator`クラス: HFベンチマーク評価クラス
- `EvaluationVisualizer`クラス: 評価結果可視化クラス
- four_classデータセット評価（accuracy, F1, confusion matrix）
- Hugging Faceベンチマーク評価（GLUE, SuperGLUE, 日本語タスク）
- 評価結果の可視化（matplotlib）
- HTMLレポート生成

### 6. A/Bテストスクリプト作成

**ファイル**: `scripts/evaluation/ab_test_borea_phi35_original_vs_so8t.py` (新規作成)

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: 元のBorea-Phi-3.5-mini-Instruct-JpとSO8T再学習済みモデルの比較評価を実装

- `ABTestEvaluator`クラス: A/Bテスト評価クラス
- `ModelLoader`クラス: モデル読み込みクラス
- `evaluate_model()`: 単一モデル評価
- `compare_models()`: モデル比較（accuracy, F1, latency）
- `visualize_comparison()`: 比較結果可視化
- `generate_html_report()`: HTMLレポート生成
- 統計的有意性検定（t検定準備）
- 比較結果の可視化とレポート生成

## 作成・変更ファイル（追加）
- `scripts/training/verify_borea_phi35_retraining.py` (新規作成)
- `scripts/training/retrain_borea_phi35_with_so8t.py` (拡張: パフォーマンスプロファイリング追加)
- `scripts/evaluation/evaluate_borea_phi35_so8t_retrained.py` (新規作成)
- `scripts/evaluation/ab_test_borea_phi35_original_vs_so8t.py` (新規作成)

## 使用方法（詳細）

### 1. 動作確認

```bash
# 基本動作確認
py scripts/training/verify_borea_phi35_retraining.py \
    --base-model Borea-Phi-3.5-mini-Instruct-Jp \
    --dataset D:/webdataset/processed/four_class/four_class_20251108_035137.jsonl \
    --config configs/retrain_borea_phi35_so8t_config.yaml \
    --output D:/webdataset/checkpoints/so8t_retrained_borea_phi35

# データセットなしで実行
py scripts/training/verify_borea_phi35_retraining.py \
    --base-model Borea-Phi-3.5-mini-Instruct-Jp \
    --output D:/webdataset/checkpoints/so8t_retrained_borea_phi35
```

### 2. SO8T再学習（パフォーマンスプロファイリング付き）

```bash
# 基本実行
py scripts/training/retrain_borea_phi35_with_so8t.py \
    --base-model Borea-Phi-3.5-mini-Instruct-Jp \
    --dataset D:/webdataset/processed/four_class/four_class_20251108_035137.jsonl \
    --output D:/webdataset/checkpoints/so8t_retrained_borea_phi35 \
    --config configs/retrain_borea_phi35_so8t_config.yaml

# パフォーマンスレポートは自動的に生成される
# D:/webdataset/checkpoints/so8t_retrained_borea_phi35/performance_report.json
```

### 3. 評価実行

```bash
# four_classデータ評価 + HFベンチマーク評価
py scripts/evaluation/evaluate_borea_phi35_so8t_retrained.py \
    --model D:/webdataset/checkpoints/so8t_retrained_borea_phi35/final_model \
    --test-data D:/webdataset/processed/four_class/four_class_20251108_035137.jsonl \
    --output-dir eval_results/borea_phi35_so8t_evaluation \
    --device cuda

# HFベンチマーク評価をスキップ
py scripts/evaluation/evaluate_borea_phi35_so8t_retrained.py \
    --model D:/webdataset/checkpoints/so8t_retrained_borea_phi35/final_model \
    --test-data D:/webdataset/processed/four_class/four_class_20251108_035137.jsonl \
    --output-dir eval_results/borea_phi35_so8t_evaluation \
    --skip-hf-benchmark
```

### 4. A/Bテスト実行

```bash
# 元のモデル vs SO8T再学習済みモデル
py scripts/evaluation/ab_test_borea_phi35_original_vs_so8t.py \
    --base-model Borea-Phi-3.5-mini-Instruct-Jp \
    --retrained-model D:/webdataset/checkpoints/so8t_retrained_borea_phi35/final_model \
    --test-data D:/webdataset/processed/four_class/four_class_20251108_035137.jsonl \
    --output-dir eval_results/ab_test_borea_phi35_original_vs_so8t \
    --device cuda
```

## 出力ファイル

### 動作確認結果
- `D:/webdataset/checkpoints/so8t_retrained_borea_phi35/verification/verification_results.json`

### パフォーマンスレポート
- `D:/webdataset/checkpoints/so8t_retrained_borea_phi35/performance_report.json`
- メモリ使用量、GPUメモリ使用量、学習速度の記録

### 評価結果
- `eval_results/borea_phi35_so8t_evaluation/evaluation_results.json`
- `eval_results/borea_phi35_so8t_evaluation/evaluation_report.html`
- `eval_results/borea_phi35_so8t_evaluation/confusion_matrix.png`
- `eval_results/borea_phi35_so8t_evaluation/metrics_comparison.png`

### A/Bテスト結果
- `eval_results/ab_test_borea_phi35_original_vs_so8t/ab_test_results.json`
- `eval_results/ab_test_borea_phi35_original_vs_so8t/ab_test_report.html`
- `eval_results/ab_test_borea_phi35_original_vs_so8t/comparison_chart.png`
- `eval_results/ab_test_borea_phi35_original_vs_so8t/confusion_matrix_comparison.png`

## 動作確認手順

1. **事前確認**: 動作確認スクリプトでモデルとデータセットの読み込みを確認
2. **再学習実行**: SO8T再学習を実行し、パフォーマンスレポートを確認
3. **評価実行**: 再学習済みモデルを評価し、評価レポートを確認
4. **A/Bテスト実行**: 元のモデルと再学習済みモデルを比較し、A/Bテストレポートを確認

## パフォーマンス最適化結果

パフォーマンスプロファイラーにより、以下のメトリクスが自動記録されます：
- メモリ使用量（MB）
- GPUメモリ使用量（MB）
- 学習速度（samples/second, tokens/second）
- 各ステップの実行時間

最適化レポートは `performance_report.json` に保存されます。

## 評価結果

評価スクリプトにより、以下のメトリクスが計算されます：
- Accuracy
- F1 Macro
- F1 per class (ALLOW, ESCALATION, DENY, REFUSE)
- Precision per class
- Recall per class
- Confusion matrix
- Average latency

評価結果はJSON形式とHTML形式で保存されます。

## A/Bテスト結果

A/Bテストにより、以下の比較メトリクスが計算されます：
- Accuracy improvement
- F1 Macro improvement
- Latency change
- Relative improvements (%)
- Statistical significance (準備中)

比較結果はJSON形式とHTML形式で保存されます。

## 実装内容（追加: 完全自動パイプライン）

### 7. A/Bテスト完全自動パイプライン実装

**ファイル**: `scripts/pipelines/complete_ab_test_post_processing_pipeline.py` (新規作成)

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: 2025-11-08  
**備考**: A/BテストからGGUF変換、Ollamaインポート、日本語パフォーマンステストまでを完全自動化

- `CompleteABTestPostProcessingPipeline`クラス: メインパイプラインクラス
- `AudioNotifier`クラス: 音声通知クラス
- `WinnerModelDeterminer`クラス: 勝者モデル判定クラス（accuracy + F1 macro比較）
- `GGUFConverter`クラス: GGUF変換クラス（F16 + Q8_0）
- `OllamaImporter`クラス: Ollamaインポートクラス
- 5つのステップを自動実行
- 各ステップ完了時に音声通知を再生

**設定ファイル**: `configs/complete_ab_test_post_processing_config.yaml` (新規作成)

**日本語パフォーマンステスト**: `scripts/testing/japanese_llm_performance_test.py` (新規作成)

詳細は `_docs/2025-11-08_main_A_Bテスト完全自動パイプライン実装.md` を参照。

## 次のステップ

1. **動作確認**: 各ステップの動作確認
2. **パフォーマンス最適化**: 学習速度とメモリ使用量の最適化
3. **評価**: 再学習済みモデルの評価
4. **A/Bテスト**: 元のモデルと再学習済みモデルの比較
5. **ドキュメント整備**: 詳細なドキュメント作成
6. **完全自動パイプライン**: A/Bテストから日本語テストまで自動実行（実装済み）

