# Borea-Phi-3.5-mini-Instruct-Common A/Bテスト実装ログ

## 実装日時
2025-11-07

## 実装概要

Borea-Phi-3.5-mini-Instruct-Commonをベースに、以下の処理を適用したモデルBを作成し、A/Bテストを実施する実装を完了しました。

**モデルBの処理フロー**:
1. 焼きこみ（SO8T Rotation Baking + Quantization）→ 学習前
2. 事後学習（言語モデル継続学習）
3. ファインチューニング（四値分類タスク特化）
4. 温度較正

**A/Bテスト評価**:
- 評価データ: `data/splits/test.jsonl`
- メトリクス: F1macro、誤検知率、正解率、混同行列、学習曲線

## 実装ファイル

### 1. モデルA（ベースライン）評価スクリプト
**ファイル**: `scripts/evaluate_model_a_baseline.py`
- Borea-Phi-3.5-mini-Instruct-Commonを直接評価
- 生成タスクとして評価し、生成テキストから分類ラベルを抽出
- メトリクス計算と保存

### 2. モデルB作成パイプライン
**ファイル**: `scripts/create_model_b_pipeline.py`
- 焼きこみ（SO8T Rotation Baking）処理
- 量子化処理（8bit）
- 事後学習実行（`scripts/finetune_borea_japanese.py`を参考）
- ファインチューニング実行（`configs/train_four_class.yaml`を使用）
- 温度較正実行（`so8t-mmllm/src/inference/temperature_calibration.py`を使用）
- 各段階のチェックポイント保存

### 3. A/Bテスト評価スクリプト
**ファイル**: `scripts/ab_test_borea_phi35.py`
- モデルA（ベースライン）とモデルBの評価実行
- 両モデルのメトリクス計算（F1macro、誤検知率、正解率、混同行列）
- 結果の比較と保存

### 4. 学習曲線可視化スクリプト
**ファイル**: `scripts/visualize_ab_test_training_curves.py`
- モデルAとモデルBの学習曲線を並列表示
- 混同行列の比較ヒートマップ
- メトリクスの比較バーグラフ
- 改善率の可視化

### 5. 統合実行スクリプト
**ファイル**: `scripts/run_ab_test_complete.py`
- 全パイプラインの統合実行
- モデルA評価 → モデルB作成 → モデルB評価 → A/Bテスト → 可視化 → レポート生成
- エラーハンドリングとログ管理

### 6. 設定ファイル
**ファイル**: `configs/ab_test_borea_phi35.yaml`
- モデルパス設定
- 学習パラメータ設定
- 評価設定
- 出力ディレクトリ設定

## 実装詳細

### ステップ1: 焼きこみ（SO8T Rotation Baking + Quantization）
- `scripts/so8t_burnin_pipeline.py`の`SO8TBurnInPipeline`を使用
- `so8t_core/burn_in.py`の`BurnInManager`を使用
- SO(8)回転行列を線形層に焼き込み（右掛け: W' = W · R）
- **注意**: SO(8)の詳細実装は機密として非開示（実装では使用）
- `utils/so8t_quantization.py`の`SO8TQuantizer`を使用して8bit量子化
- 量子化済みモデルを保存

### ステップ2: 事後学習（言語モデル継続学習）
- `scripts/finetune_borea_japanese.py`の`BoreaJapaneseFinetuner`を参考
- 量子化済みモデルをベースに継続学習
- データセット: `data/splits/train.jsonl`（言語モデルタスク）
- チェックポイント保存

### ステップ3: ファインチューニング（四値分類）
- `scripts/train_four_class_classifier.py`を使用
- 設定: `configs/train_four_class.yaml`
- データセット: `data/splits/train.jsonl`（四値分類ラベル付き）
- 学習曲線データを保存

### ステップ4: 温度較正
- `so8t-mmllm/src/inference/temperature_calibration.py`の`TemperatureCalibrator`を使用
- 検証データ: `data/splits/val.jsonl`
- 最適温度を計算して保存

### ステップ5: A/Bテスト評価
- モデルA: `Borea-Phi-3.5-mini-Instruct-Common`（そのまま）
- モデルB: 処理済みモデル
- 評価データ: `data/splits/test.jsonl`
- メトリクス計算:
  - F1macro
  - 誤検知率（DENY/REFUSEをALLOWと誤分類）
  - 正解率（Accuracy）
  - 混同行列
- 結果をJSON形式で保存

### ステップ6: 可視化
- 混同行列のヒートマップ（両モデル並列）
- 学習曲線の比較（損失、精度、F1スコア）
- メトリクスの比較バーグラフ
- 出力: `eval_results/ab_test_comparison/`

## 出力ファイル

### モデルB関連
- `checkpoints/borea_phi35_model_b/quantized/`: 量子化済みモデル
- `checkpoints/borea_phi35_model_b/post_trained/`: 事後学習済みモデル
- `checkpoints/borea_phi35_model_b/fine_tuned/`: ファインチューニング済みモデル
- `checkpoints/borea_phi35_model_b/calibrated/`: 温度較正済みモデル

### 評価結果
- `eval_results/ab_test_comparison/metrics_model_a.json`: モデルA評価結果
- `eval_results/ab_test_comparison/metrics_model_b.json`: モデルB評価結果
- `eval_results/ab_test_comparison/comparison_report.json`: 比較レポート
- `eval_results/ab_test_comparison/confusion_matrix_comparison.png`: 混同行列比較
- `eval_results/ab_test_comparison/training_curves_comparison.png`: 学習曲線比較
- `eval_results/ab_test_comparison/metrics_comparison.png`: メトリクス比較
- `eval_results/ab_test_comparison/ab_test_report.md`: 最終レポート

## 使用方法

### 1. モデルA評価のみ
```bash
python scripts/evaluate_model_a_baseline.py --model Borea-Phi-3.5-mini-Instruct-Common --test data/splits/test.jsonl
```

### 2. モデルB作成のみ
```bash
python scripts/create_model_b_pipeline.py --config configs/ab_test_borea_phi35.yaml
```

### 3. A/Bテスト比較のみ
```bash
python scripts/ab_test_borea_phi35.py --model-a Borea-Phi-3.5-mini-Instruct-Common --model-b checkpoints/borea_phi35_model_b/calibrated/final_model --test data/splits/test.jsonl
```

### 4. 可視化のみ
```bash
python scripts/visualize_ab_test_training_curves.py --metrics-a eval_results/ab_test_comparison/metrics_model_a.json --metrics-b eval_results/ab_test_comparison/metrics_model_b.json
```

### 5. 統合実行（推奨）
```bash
python scripts/run_ab_test_complete.py --config configs/ab_test_borea_phi35.yaml
```

## 注意事項

1. **SO(8)実装の機密性**: SO(8)回転行列の詳細実装は機密として非開示されていますが、実装では使用されています。

2. **メモリ要件**: モデルB作成には大量のメモリが必要です。RTX3080のCUDA12を使用することを推奨します。

3. **実行時間**: 全パイプラインの実行には数時間から数日かかる可能性があります。

4. **チェックポイント**: 各段階でチェックポイントが保存されるため、途中で中断しても再開可能です。

## 今後の拡張

- 学習曲線の詳細な可視化
- より詳細な統計的検定
- 複数の評価データセットでの評価
- ハイパーパラメータ最適化

## 実装完了

全実装が完了しました。各スクリプトは独立して実行可能で、統合スクリプトで一括実行も可能です。

## 実装済みファイル一覧

1. ✅ `scripts/evaluate_model_a_baseline.py` - モデルA（ベースライン）評価スクリプト
2. ✅ `scripts/create_model_b_pipeline.py` - モデルB作成パイプライン
3. ✅ `scripts/ab_test_borea_phi35.py` - A/Bテスト評価スクリプト
4. ✅ `scripts/visualize_ab_test_training_curves.py` - 学習曲線可視化スクリプト
5. ✅ `scripts/run_ab_test_complete.py` - 統合実行スクリプト
6. ✅ `configs/ab_test_borea_phi35.yaml` - 設定ファイル

## 実装ステータス

- ✅ モデルA（ベースライン）評価スクリプト作成
- ✅ モデルB作成パイプライン（焼きこみ→事後学習→ファインチューニング→温度較正）
- ✅ A/Bテスト評価スクリプト作成
- ✅ 学習曲線可視化スクリプト作成
- ✅ 統合実行スクリプト作成
- ✅ 設定ファイル作成
- ✅ 実装ログ作成

すべての実装が完了し、実行可能な状態です。

## convert_hf_to_gguf_update.py統合

`convert_hf_to_gguf_update.py`をSO8Tモデルの焼き込み処理に統合しました。

### 統合内容

1. **SO8Tトークナイザー情報の自動更新**
   - 焼き込み処理の後に、SO8Tトークナイザーのハッシュを自動計算
   - `convert_hf_to_gguf.py`の`get_vocab_base_pre()`関数にSO8Tトークナイザーハッシュを自動追加

2. **convert_hf_to_gguf_update.pyへの統合**
   - `convert_hf_to_gguf_update.py`の`models`リストにSO8Tモデルを自動追加
   - ローカルモデルディレクトリを参照する設定に対応

3. **フォールバック機能**
   - `convert_hf_to_gguf_update.py`が見つからない場合、`convert_hf_to_gguf.py`に直接追加

### 実装メソッド

- `_update_so8t_tokenizer_info()`: SO8Tトークナイザー情報を更新するメイン関数
- `_add_so8t_to_update_script()`: `convert_hf_to_gguf_update.py`の`models`リストにSO8Tモデルを追加
- `_add_so8t_tokenizer_directly()`: `convert_hf_to_gguf.py`に直接SO8Tトークナイザーハッシュを追加

これにより、SO8TモデルのGGUF変換時にトークナイザー情報が正しく認識されるようになりました。

