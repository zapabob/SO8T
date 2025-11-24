# SO8T (SO(8) Transformer) Project

## 📋 Project Concept

**SO(8)群構造とAlpha Gateを用いた幾何学的制約によるLLMの制御手法**

SO8Tは、8次元回転群SO(8)の数学的構造を活用し、Alpha Gate（シグモイドアニーリング）による幾何学的制約を適用することで、LLMの安全性と一貫性を確保する革新的なアプローチを実装したプロジェクトです。

### 核心技術
- **SO(8)群構造**: 非可換ゲートによる安全性の幾何学的制約
- **Alpha Gate**: 温度アニーリングによる制御パラメータ最適化
- **PET正則化**: 時系列的一貫性確保
- **四重推論システム**: 論理・倫理・実用・創造の4軸評価

## 🏗️ Model Architecture

### Base Model: Phi-3.5
- **ベースアーキテクチャ**: Microsoft Phi-3.5-mini-instruct (3.8B parameters)
- **量子化**: Q8_0, Q4_K_M, F16対応
- **最適化**: RTX 3060/3080対応GPU最適化

### SO(8)介入層
```
Input → Phi-3.5 Encoder → [SO(8) Rotation Gates] → [Alpha Gate Control] → Safety Head → Output
                          ↓
                   PET Regularization
                          ↓
                  SQLite Audit Logging
```

#### 主要コンポーネント
- **`models/so8t_group_structure.py`**: SO(8)回転行列の実装
- **`models/alpha_gate.py`**: シグモイドアニーリング制御
- **`models/so8t_safety_judge.py`**: 安全性判断ヘッド
- **`utils/so8t_compliance_logger.py`**: 完全監査ログシステム

### 学習ログと分析
- **Loss曲線**: `docs/figures/alpha_gate_loss_curve.png` - Phase Transitionを示すLossの推移
- **総合分析**: `docs/figures/alpha_gate_comprehensive_analysis.png` - Alpha Gateの詳細分析
- **学習サマリー**: `docs/figures/training_summary.txt` - 学習結果の概要

#### Phase Transitionの物理的解釈
Alpha Gateの学習中に観測されるPhase Transitionは、幾何学的制約が突然有効化される物理現象を示します。Alpha値が0.5を超えた時点で、SO(8)群構造の幾何学的制約が支配的になり、Lossが急激に減少します。これは、モデルがSO(8)群の対称性を学習し、安定した表現を獲得したことを示す証拠です。

## 🔬 Benchmark Method

### 実行環境
- **Runtime**: Python 3.12 / Ollama 0.3.0+
- **GPU**: RTX 3060/3080 (CUDA 12.1+)
- **OS**: Windows 11 / Ubuntu 22.04+

### 業界標準パイプライン

1. **lm-evaluation-harness (Open LLM Leaderboard準拠)**

```bash
   py -3 scripts/evaluation/lm_eval_benchmark.py ^
       --model-runner hf ^
       --model-name microsoft/Phi-3.5-mini-instruct ^
       --tasks gsm8k mmlu hellaswag ^
       --batch-size 4

   py -3 scripts/evaluation/lm_eval_benchmark.py ^
       --model-runner llama.cpp ^
       --model-name D:/webdataset/gguf_models/aegis-borea-phi35/aegis-borea-phi35_Q8_0.gguf ^
       --model-args n_gpu_layers=40 ^
       --batch-size 2
   ```

   - すべての結果は `D:/webdataset/benchmark_results/lm_eval/` に保存。
   - CUDAリソースは `scripts/cuda_accelerated_benchmark.py` で一括管理可能。

2. **DeepEval (倫理 / 論理 / ハルシネーション)**

   ```bash
   py -3 scripts/evaluation/deepeval_ethics_test.py ^
       --model-runner ollama ^
       --model-name aegis-borea-phi35-instinct-jp:q8_0
   ```

   - Hallucination / Bias / Answer Relevancy を自動採点。
   - 結果は `D:/webdataset/benchmark_results/deepeval/` にJSONで記録。

3. **promptfoo (A/B可視化)**

```bash
   py -3 scripts/evaluation/promptfoo_ab_test.py ^
       --config configs/promptfoo_config.yaml ^
       --use-npx --html --json
   ```

   - Node.js環境は `scripts/utils/check_nodejs.bat` で検証。
   - HTML/JSONレポートは `D:/webdataset/benchmark_results/promptfoo/` に保存。

4. **統合レポート**

   ```bash
   py -3 scripts/evaluation/industry_standard_benchmark.py
   ```

   - 上記3ツールを順次実行し、`_docs/benchmark_results/industry_standard/` にMarkdownレポートを生成。
   - Git worktree名を含む `metadata.json` で再現性を保証。

### 評価基準
- **Accuracy (lm-eval)**: MMLU / GSM8K / HellaSwag の公式スコア
- **Ethics (DeepEval)**: Hallucination / Bias / Relevancy の合格率
- **A/B差分 (promptfoo)**: HTMLレポートでモデル間スコアを比較

### 再現性の確保
- すべてのスクリプトは `py -3` 起動 & `tqdm` 進行表示
- モデル成果物は `D:/webdataset` 配下に保存
- 各ステップ完了時に `scripts/utils/play_audio_notification.ps1` を再生し、実験ログと同期

## 📊 Data Provenance

### 学習データセット

#### 1. 主要データソース
- **TFMC/imatrix-dataset-for-japanese-llm**: 日本語LLM向け重要度行列データセット
  - 出典: https://huggingface.co/datasets/TFMC/imatrix-dataset-for-japanese-llm
  - 用途: 量子化最適化のための重要度学習

#### 2. 独自生成データセット
- **`data/so8t_safety_dataset.jsonl`**: 安全性学習用データセット
  - 生成元: 倫理的ジレンマシナリオと安全応答パターン
  - サイズ: 10,000+ サンプル

- **`data/japanese_complex_dataset_enhanced.jsonl`**: 日本語複雑推論データセット
  - 生成元: 数学・科学・倫理的問題の日本語訳
  - サイズ: 5,000+ サンプル

#### 3. ファインチューニングデータ
- **`data/so8t_thinking_phi35_weighted_train.jsonl`**: SO(8)思考制御トレーニングデータ
  - 生成方法: Phi-3.5ベースの思考プロセス拡張
  - 特徴: 四重推論（logic/ethics/practical/creative）タグ付き

### データ前処理

#### 前処理スクリプト
```bash
# データクリーニング
python scripts/data_preprocessing/clean_dataset.py

# 品質チェック
python scripts/data_preprocessing/validate_dataset.py

# SO(8)適応変換
python scripts/data_preprocessing/apply_so8t_transform.py
```

#### 品質基準
- **NSFWフィルタリング**: 安全学習目的のみ使用（生成目的禁止）
- **言語品質**: 日本語・英語の両言語対応
- **多様性確保**: ドメイン偏在の排除

## 🔄 Reproduction Guide

### AEGISモデルの再学習手順

#### 1. 環境準備
```bash
# 依存関係インストール
pip install -r requirements.txt

# CUDA対応PyTorchインストール（GPU使用時）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 2. データ準備
```bash
# データセットダウンロード
python scripts/data/download_datasets.py

# データ前処理
python scripts/data_preprocessing/prepare_training_data.py
```

#### 3. SO(8)モデル学習
```bash
# Alpha Gate付き学習スクリプト
python scripts/train_so8t_alpha_gate.py \
    --model_name "microsoft/phi-3.5-mini-instruct" \
    --dataset "data/so8t_thinking_phi35_weighted_train.jsonl" \
    --output_dir "models/aegis_trained" \
    --alpha_initial 0.1 \
    --alpha_final 0.8 \
    --annealing_steps 1000 \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --num_epochs 3
```

#### 4. 安全性ファインチューニング
```bash
# 安全性ヘッド学習
python scripts/train_safety_head.py \
    --base_model "models/aegis_trained" \
    --safety_dataset "data/so8t_safety_dataset.jsonl" \
    --output_dir "models/aegis_final"
```

#### 5. GGUF変換とOllama登録
```bash
# GGUF変換
python scripts/convert_to_gguf.py \
    --model_path "models/aegis_final" \
    --output_path "D:\webdataset\gguf_models\aegis_custom\aegis_custom_Q8_0.gguf" \
    --quantization "Q8_0"

# Ollama Modelfile作成
python scripts/create_ollama_modelfile.py \
    --gguf_path "D:\webdataset\gguf_models\aegis_custom\aegis_custom_Q8_0.gguf" \
    --model_name "aegis-custom" \
    --template "aegis"

# Ollamaに登録
ollama create aegis-custom:latest -f modelfiles/aegis-custom.modelfile
```

#### 6. 検証テスト
```bash
# 機能テスト
ollama run aegis-custom:latest "AIの倫理的課題について議論してください。"

# ベンチマーク実行
python scripts/testing/run_aegis_benchmark.py
```

### ⚠️ 重要注意事項

- **計算リソース**: RTX 3060以上推奨（VRAM 12GB+）
- **学習時間**: Alpha Gateアニーリングにより通常の2-3倍の時間が必要
- **データ品質**: NSFWデータは安全学習目的のみ使用
- **再現性**: 乱数シードを固定（`--seed 42`）で結果の再現性を確保

## 主要機能

### 🔄 SO(8)群構造
- 8次元回転群の数学的構造を活用
- 非可換ゲート（R_safe → R_cmd）による安全性確保
- PET正則化による時系列一貫性の保持

### 🛡️ 安全性機能
- 安全性判断ヘッドによる倫理的推論
- SQLite監査システムによる完全な決定ログ
- 自己検証機能による出力品質保証

### 🚀 GPU最適化
- RTX 3060対応のGPU最適化
- CUDA 13.0サポート
- 効率的なメモリ使用（80%使用率）

### 📊 量子化サポート
- Q8_0, Q4_0, F16量子化
- GGUF形式でのモデル配布
- llama.cpp互換性

### 🧠 AEGIS: 四値分類・四重推論システム
- **論理的正確性**: 数学的・論理的検証 (`<think-logic>`)
- **倫理的妥当性**: 道徳的・倫理的評価 (`<think-ethics>`)
- **実用的価値**: 現実世界での実現可能性 (`<think-practical>`)
- **創造的洞察**: 革新的アイデアと視点 (`<think-creative>`)
- **構造化応答**: XMLタグによる明確な思考プロセス

## 統合開発フロー

SO8Tプロジェクトは、開発を直線関係にするために統合されたモジュール構造を採用しています。

### 📋 開発ステップ（線形フロー）

1. **環境セットアップ**: `python scripts/setup.py`
2. **データ準備**: `python scripts/train.py --prepare-data`
3. **モデル学習**: `python scripts/train.py`
4. **評価実行**: `python scripts/eval.py`
5. **デプロイ**: `python scripts/deploy.py`

### 🏗️ 統合モジュール構造

```
SO8T/
├── so8t/                           # 統合SO8Tパッケージ
│   ├── core/                       # SO(8)コアコンポーネント
│   ├── training/                   # 学習関連
│   ├── inference/                  # 推論関連
│   ├── data/                       # データ処理
│   ├── safety/                     # 安全性機能
│   ├── utils/                      # 汎用ユーティリティ
│   └── config/                     # 統合設定ファイル
├── scripts/                        # 実行スクリプト（線形フロー）
│   ├── setup.py                    # 環境セットアップ
│   ├── train.py                    # 学習パイプライン
│   ├── eval.py                     # 評価パイプライン
│   └── deploy.py                   # デプロイパイプライン
├── _docs/                          # プロジェクトドキュメント
│   └── test_so8t_ollama_complex.bat     # 複雑テストスクリプト
├── tests/                           # テストファイル
│   ├── test_so8_operations_comprehensive.py  # SO(8)演算テスト
│   ├── test_pytorch_comparison.py           # PyTorch比較テスト
│   └── test_so8t_quantization.py           # 量子化テスト
├── utils/                           # ユーティリティ
│   ├── so8t_quantization.py         # 量子化機能
│   ├── weight_stability_manager.py  # 重み安定性管理
│   └── ocr_processor.py             # OCR処理
├── so8t-mmllm/                      # メイン実装
│   ├── src/                         # ソースコード
│   ├── configs/                     # 設定ファイル
│   └── outputs/                     # 出力ファイル
├── external/                        # 外部ライブラリ
│   └── llama.cpp-master/            # llama.cpp
├── _docs/                           # 実装ログ
├── archive/                         # アーカイブファイル
└── test_images/                     # テスト画像
```

## クイックスタート

### 1. 環境セットアップ

```bash
# 依存関係のインストール
pip install -r requirements.txt

# CUDA対応PyTorch（オプション）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. モデルの実行

```bash
# Ollamaでモデルを実行
ollama run so8t-lightweight "あなたのプロンプト"

# GPU最適化モデル（要CUDA）
ollama run so8t-vl-2b-instruct-gpu "あなたのプロンプト"

# AEGISモデル（四重推論）
ollama run agiasi-phi35-golden-sigmoid:q8_0 "AIの未来についてどう思いますか？"
```

### 3. 複雑なテストの実行

```bash
# 複雑な数学的推論テスト
scripts\test_so8t_ollama_complex.bat

# 包括的なテストスイート
scripts\run_comprehensive_tests.bat
```

## 主要コンポーネント

### SO(8)群構造実装
- `models/so8t_group_structure.py`: SO(8)回転行列の実装
- `models/so8t_mlp.py`: SO(8)群構造を持つMLP
- `models/so8t_attention.py`: SO(8)回転埋め込み

### 安全性機能
- `models/so8t_safety_judge.py`: 安全性判断ヘッド
- `utils/so8t_compliance_logger.py`: コンプライアンスログ
- `so8t-mmllm/src/audit/sqlite_logger.py`: SQLite監査

### GPU最適化
- `scripts/convert_so8t_to_gguf_gpu.py`: GPU最適化GGUF変換
- `models/Modelfile-gpu-simple`: GPU最適化Modelfile

## テスト

### 単体テスト
```bash
# SO(8)演算テスト
python -m pytest tests/test_so8_operations_comprehensive.py

# PyTorch比較テスト
python -m pytest tests/test_pytorch_comparison.py

# 量子化テスト
python -m pytest tests/test_so8t_quantization.py
```

### 統合テスト
```bash
# 包括的テスト
scripts\run_comprehensive_tests.bat

# Ollama統合テスト
scripts\test_so8t_ollama_complex.bat
```

## パフォーマンス

### GPU最適化設定
- **GPU**: RTX 3060 (12GB VRAM)
- **CUDA**: 13.0
- **メモリ使用率**: 80%
- **並列処理**: 8スレッド

### ベンチマーク結果
- **数学的推論**: 4次元超立方体問題の解決
- **科学的概念**: 量子力学のSO(8)群理論説明
- **論理的推論**: パラドックス解析
- **倫理的推論**: AI安全性判断

## ライセンス

MIT License - 詳細は[LICENSE](LICENSE)を参照してください。

## 貢献

プロジェクトへの貢献を歓迎します。プルリクエストやイシューの報告をお待ちしています。

## 更新履歴

- **2025-10-29**: GPU最適化実装完了
- **2025-10-29**: リポジトリ整理整頓完了
- **2025-10-29**: 複雑テスト実装完了
- **2025-10-29**: GGUF変換パイプライン完成

## サポート

質問や問題がある場合は、GitHubのIssuesページで報告してください。

---

**SO8T Project** - SO(8)群構造による先進的AI実装 🚀