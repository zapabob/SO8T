# SO8T×マルチモーダルLLM（ローカル）

SO(8)群回転ゲート + PET正則化 + OCR要約 + SQLite監査を統合した安全エージェント

## 🚀 概要

RTX3060 12GB環境で動作する、SO(8)群構造とTriality対称性を活用した革新的なマルチモーダルLLMです。回転ゲート、PET正則化、ローカルOCR要約、SQLite監査を統合し、安全性とプライバシーを重視した設計となっています。

## ✨ 主要機能

### 🔄 SO(8)群回転ゲート
- SDPA出力後の8次元ブロック直交回転
- 数値安定化された行列指数関数
- QLoRA 8bit学習対応

### 📊 PET正則化
- 二階差分による高周波変動抑制
- 3相スケジュール（warmup→main→anneal）
- Huber損失対応

### 🔍 ローカルOCR要約
- OpenCV + Tesseract による画像処理
- 外部送信なしのプライバシー保護
- 多言語対応（日本語・英語）

### 🗄️ SQLite監査
- WALモード + synchronous=FULL
- 判断ログ、ポリシー状態、アイデンティティ契約
- 完全な監査トレイル

## 📁 プロジェクト構造

```
so8t-mmllm/
├── src/                    # ソースコード
│   ├── modules/           # コアモジュール
│   │   └── rotation_gate.py
│   ├── losses/            # 損失関数
│   │   └── pet.py
│   ├── training/          # 学習関連
│   │   └── qlora.py
│   ├── io/               # I/O処理
│   │   └── ocr_summary.py
│   └── audit/            # 監査機能
│       └── sqlite_logger.py
├── configs/              # 設定ファイル
│   ├── model.qwen2vl-2b.json
│   └── train.qlora.json
├── sql/                  # SQLiteスキーマ
│   ├── schema.sql
│   └── init_data.sql
├── scripts/              # 実行スクリプト
│   ├── setup.ps1
│   ├── train.ps1
│   ├── eval.ps1
│   └── bake_and_convert.ps1
├── eval/                 # 評価スクリプト
│   ├── metrics.py
│   └── tasks_safety.json
├── requirements.txt      # 依存関係
└── README.md
```

## 🛠️ セットアップ

### 1. 環境構築

```powershell
# セットアップスクリプトを実行
.\scripts\setup.ps1
```

### 2. 仮想環境のアクティベート

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. 依存関係のインストール

```powershell
py -3 -m pip install -r requirements.txt
```

## 🚀 使用方法

### 学習

```powershell
# 学習を開始
.\scripts\train.ps1
```

### 評価

```powershell
# 評価を実行
.\scripts\eval.ps1
```

### 推論

```python
from src import SO8TQLoRATrainer

# 学習器を初期化
trainer = SO8TQLoRATrainer(
    model_path="../Qwen2-VL-2B-Instruct",
    config_path="configs/train.qlora.json"
)

# テキスト生成
result = trainer.generate("画像を説明してください。")
print(result)
```

## ⚙️ 設定

### モデル設定 (`configs/model.qwen2vl-2b.json`)

```json
{
  "model_name": "Qwen2-VL-2B-Instruct",
  "hidden_size": 1536,
  "num_attention_heads": 12,
  "rotation_gate_enabled": true,
  "pet_loss_enabled": true
}
```

### 学習設定 (`configs/train.qlora.json`)

```json
{
  "learning_rate": 2e-4,
  "batch_size": 1,
  "gradient_accumulation_steps": 8,
  "lora_rank": 64,
  "lora_alpha": 128,
  "pet_lambda_schedule": {
    "max_lambda": 0.1,
    "warmup_steps": 100,
    "main_steps": 1000,
    "anneal_steps": 200
  }
}
```

## 🔒 セキュリティ・プライバシー

### プライバシー保護
- **画像外部送信禁止**: 全ての画像処理はローカルで実行
- **OCR要約のみ**: 生画像ではなく要約テキストのみをモデルに投入
- **データ保持期間**: 7日間（設定可能）

### 監査機能
- **完全なログ記録**: 全ての判断とポリシー変更を記録
- **WALモード**: 電源断からの復旧対応
- **差分追跡**: ポリシーと契約の変更履歴

## 📊 性能要件

### ハードウェア要件
- **GPU**: RTX3060 12GB以上
- **RAM**: 32GB以上
- **ストレージ**: 50GB以上の空き容量

### 性能指標
- **推論速度**: 15-60秒以内
- **メモリ使用量**: 32GB以内
- **成功率**: 95%以上
- **安全性**: 98%以上の有害コンテンツ検出

## 🧪 テスト

### 基本機能テスト

```powershell
py -3 -m pytest tests/test_basic.py -v
```

### 性能テスト

```powershell
py -3 -m pytest tests/test_performance.py -v
```

### 安全性テスト

```powershell
py -3 -m pytest tests/test_safety.py -v
```

## 📈 評価指標

### 精度指標
- **BLEU**: テキスト生成品質
- **ROUGE**: 要約品質
- **CIDEr**: 画像説明品質

### 安全性指標
- **安全F1**: 有害コンテンツ検出精度
- **Refuse Recall**: 拒否率
- **誤許可率**: 誤って許可した有害コンテンツの割合

### 学習安定性
- **損失収束**: 学習曲線の安定性
- **勾配ノルム**: 勾配爆発の監視
- **重み分布**: パラメータの分布

## 🔧 トラブルシューティング

### メモリ不足
- バッチサイズを削減
- 勾配蓄積ステップ数を増加
- 量子化レベルを調整

### 学習不安定
- 学習率を調整
- ウォームアップステップ数を増加
- 勾配クリッピングを適用

### OCR精度低下
- 画像前処理パラメータを調整
- Tesseract設定を最適化
- 言語設定を確認

## 📚 参考文献

- [Qwen2-VL: A Versatile Vision-Language Model](https://arxiv.org/abs/2409.12191)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [SO(8) Group Theory and Applications](https://en.wikipedia.org/wiki/SO(8))

## 📄 ライセンス

Apache 2.0 License

## 🤝 貢献

プルリクエストやイシューの報告を歓迎します。

## 📞 サポート

問題が発生した場合は、GitHubのIssuesページで報告してください。
