# SO8T (SO(8) Transformer) Project

## プロジェクト概要

SO8Tは、SO(8)群構造を活用した先進的なマルチモーダル大規模言語モデル（LLM）プロジェクトです。安全性機能とローカル処理機能を備え、GPU最適化された推論を提供します。

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

### 🧠 AGIASI: 四値分類・四重推論システム
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

# AGIASIモデル（四重推論）
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