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

## ディレクトリ構造

```
SO8T/
├── models/                          # モデルファイル
│   ├── so8t-vl-2b-instruct-main.gguf    # メインモデル
│   ├── so8t-vl-2b-instruct-gpu-optimized.gguf  # GPU最適化モデル
│   ├── so8t-vl-2b-instruct-lightweight.gguf    # 軽量版モデル
│   ├── Modelfile-main                # メインModelfile
│   ├── Modelfile-gpu-simple          # GPU最適化Modelfile
│   └── *.py                         # モデル実装ファイル
├── scripts/                         # 実行スクリプト
│   ├── convert_so8t_to_gguf_fixed.py    # GGUF変換（修正版）
│   ├── convert_so8t_to_gguf_gpu.py      # GPU最適化GGUF変換
│   ├── convert_so8t_to_gguf_llama.py    # llama.cpp互換GGUF変換
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