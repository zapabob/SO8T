# SO8T (SO(8) Transformer) Project

## 概要

SO8Tは、SO(8)群構造とTriality対称性を活用した高度なAI推論システムです。自己検証機能、マルチパス推論、安全性評価を統合した革新的なTransformerアーキテクチャを提供します。

## 特徴

### 🧠 高度な推論能力
- **SO(8)群構造**: 数学的対称性を活用した効率的な推論
- **Triality対称性**: 3つの表現（Vector, Spinor+, Spinor-, Verifier）による多角的分析
- **自己検証システム**: リアルタイムでの一貫性検証と品質保証

### 🔒 安全性と倫理
- **マルチレイヤー安全性フィルタリング**: 有害な出力の防止
- **倫理推論エンジン**: 複数の倫理フレームワークによる分析
- **透明性プロトコル**: 推論プロセスの可視化

### ⚡ 高性能
- **マルチパス生成**: 3-5つのアプローチによる多様な解決策
- **インテリジェント選択**: 最適な解決策の自動選択
- **自己再試行メカニズム**: エラー時の自動修正

## プロジェクト構造

```
SO8T/
├── agents/                 # エージェント実装
├── configs/               # 設定ファイル
├── docs/                  # ドキュメント
├── models/                # モデルファイル
├── modelfiles/            # Ollama Modelfile
├── scripts/               # スクリプト
├── shared/                # 共有ライブラリ
├── tests/                 # テストファイル
└── _docs/                 # 実装ログ
```

## クイックスタート

### 1. 環境セットアップ

```bash
# リポジトリをクローン
git clone https://github.com/your-username/SO8T.git
cd SO8T

# 仮想環境を作成
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# または
.venv\Scripts\activate     # Windows

# 依存関係をインストール
pip install -r requirements.txt
```

### 2. モデルの実行

```bash
# Ollamaでモデルを実行
ollama run so8t-simple "SO8群構造について説明してください"

# 複雑な問題のテスト
python scripts/test_complex_problems.py

# 性能テスト
python scripts/test_so8t_performance.py
```

### 3. カスタムモデルの作成

```bash
# Modelfileを使用してモデルを作成
ollama create so8t-custom -f modelfiles/Modelfile-SO8T-Simple

# モデルを実行
ollama run so8t-custom "あなたの質問"
```

## 利用可能なモデル

### 🥇 so8t-simple
- **用途**: 日常使用
- **特徴**: 最も安定して動作、高速推論
- **推論時間**: 0.35秒
- **成功率**: 100%

### 🥈 so8t-phi3-vision-enhanced
- **用途**: 中程度の機能が必要な場合
- **特徴**: 中程度の性能、より多くの機能
- **推論時間**: 1.48秒
- **成功率**: 100%

### 🥉 so8t-ollama32-enhanced-gguf
- **用途**: 研究用途
- **特徴**: 最も高度な機能、推論速度は遅い
- **推論時間**: 2.03秒
- **成功率**: 100%

### ❌ so8t-phi31-lmstudio-enhanced
- **用途**: 高メモリ環境
- **特徴**: 最も高度な機能、51.7GB以上のメモリが必要
- **推論時間**: 実行不可（メモリ不足）

## 技術仕様

### アーキテクチャ
- **Vector Representation**: タスク実行とマルチアプローチ生成
- **Spinor+ Representation**: 安全性と倫理の推論
- **Spinor- Representation**: エスカレーションと学習
- **Verifier Representation**: 自己検証と品質保証

### 品質基準
- **信頼度閾値**: 0.75以上
- **安全性閾値**: 0.85以上
- **一貫性閾値**: 0.80以上
- **完全性閾値**: 0.80以上
- **精度閾値**: 0.85以上

## テスト結果

### 基本機能テスト
- **so8t-simple**: 5/5 (100.0%) - 最も安定
- **so8t-phi3-vision-enhanced**: 5/5 (100.0%) - 中程度の性能
- **so8t-ollama32-enhanced-gguf**: 5/5 (100.0%) - 高度な機能

### 性能テスト
- **so8t-simple**: 推論速度最速 (0.35秒)、安定性最高
- **so8t-phi3-vision-enhanced**: 中程度の性能、エラー多発
- **so8t-ollama32-enhanced-gguf**: 高度な機能、タイムアウト多発

## ドキュメント

- [実装ログ](_docs/) - 詳細な実装履歴
- [モデルカード](docs/) - 各モデルの詳細仕様
- [API仕様](docs/) - APIの使用方法
- [設定ガイド](configs/) - 設定ファイルの説明

## 貢献

1. フォークを作成
2. フィーチャーブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add some amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

## サポート

- [Issues](https://github.com/your-username/SO8T/issues) - バグレポートや機能要求
- [Discussions](https://github.com/your-username/SO8T/discussions) - 質問や議論
- [Wiki](https://github.com/your-username/SO8T/wiki) - 詳細なドキュメント

## 謝辞

- Microsoft Phi-3モデルファミリー
- Ollamaプロジェクト
- オープンソースコミュニティ

---

**SO8T - 次世代AI推論システム**
