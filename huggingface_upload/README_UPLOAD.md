# AEGIS HuggingFace Upload Guide

## 📁 アップロード準備完了

AEGISモデルをHuggingFaceにアップロードするための準備が完了しました。

### アップロード用フォルダー内容

```
D:\webdataset\models\aegis-huggingface-upload/
├── 📄 README.md                          # Model Card (SO8T伏せ版)
├── ⚖️ LICENSE                            # MIT License
├── ⚙️ config.json                        # モデル設定
├── ⚙️ generation_config.json             # 生成設定
├── 🔤 tokenizer.json                     # トークナイザー設定
├── 🔤 tokenizer.model                    # トークナイザーモデル
├── 🔤 tokenizer_config.json              # トークナイザー設定
├── 🔤 special_tokens_map.json            # 特殊トークン設定
├── 🔤 added_tokens.json                  # 追加トークン設定
└── 📊 benchmark_results/                 # ベンチマーク可視化
    ├── overall_performance_comparison.png
    ├── category_performance_comparison.png
    ├── response_time_comparison.png
    └── summary_statistics.png
```

### モデルファイル（別途指定）
- `models/aegis_adjusted/model-00001-of-00002.safetensors` (~5GB)
- `models/aegis_adjusted/model-00002-of-00002.safetensors` (~2.3GB)

## 🚀 アップロード方法

### 方法1: Pythonスクリプト（推奨）

```bash
# 1. 依存関係インストール
pip install -r scripts/upload_requirements.txt

# 2. HuggingFaceトークン設定
export HF_TOKEN="your-huggingface-token"

# 3. アップロード実行
python scripts/upload_aegis_to_huggingface.py your-username/AEGIS-Phi3.5-Enhanced
```

### 方法2: HuggingFace CLI

```bash
# 1. CLIインストール
pip install huggingface_hub[cli]

# 2. ログイン
huggingface-cli login

# 3. アップロード実行
bash scripts/upload_aegis_hf.sh your-username/AEGIS-Phi3.5-Enhanced
```

### 方法3: Windowsバッチファイル

```cmd
REM Windowsコマンドプロンプトで実行
scripts\upload_aegis_hf.bat your-username/AEGIS-Phi3.5-Enhanced
```

## 🔧 HuggingFaceトークンの取得

1. [HuggingFace](https://huggingface.co/) にアクセス
2. アカウント作成/ログイン
3. Settings → Access Tokens → New token
4. Token type: "Write" 権限を選択
5. トークンをコピー

### 環境変数設定

```bash
# Linux/Mac
export HF_TOKEN="your-token-here"

# Windows PowerShell
$env:HF_TOKEN="your-token-here"

# Windows CMD
set HF_TOKEN=your-token-here
```

## 📋 アップロード後の確認事項

### 1. Model Cardの確認
- README.mdが正しく表示されているか確認
- ベンチマーク画像が表示されているか確認

### 2. モデルメタデータの設定
- **Pipeline tag**: `text-generation`
- **Tags**: transformers, phi-3, enhanced-reasoning, ethical-ai, japanese, reasoning, safety, transformer, mathematical-reasoning, quadruple-reasoning, thinking-model
- **License**: Apache 2.0

### 3. モデルテスト
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("your-username/AEGIS-Phi3.5-Enhanced")
tokenizer = AutoTokenizer.from_pretrained("your-username/AEGIS-Phi3.5-Enhanced")

# テスト推論
messages = [{"role": "user", "content": "Hello, how are you?"}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

## ⚠️ 注意事項

### 大きなファイルの扱い
- safetensorsファイルは合計7GB以上あります
- アップロードには時間がかかります（数時間）
- 安定したインターネット接続が必要です

### SO8T技術の伏せ
- README.mdではSO8Tを言及していません
- 「Transformer数理的改良」「思考モデルSFT」として説明
- 四重推論の一般ユーザー向け実用性を強調

### ライセンスと使用条件
- Apache 2.0 Licenseを適用
- 商用利用時は連絡を推奨
- 軍事・違法用途は禁止

## 🎯 公開後の活用

### モデルページURL
```
https://huggingface.co/your-username/AEGIS-Phi3.5-Enhanced
```

### コミュニティへの共有
- Discord: HuggingFaceコミュニティ
- Reddit: r/LocalLLaMA, r/MachineLearning
- Twitter: #HuggingFace, #LLM, #AI

### 改善フィードバックの収集
- Issuesでバグ報告を受け付ける
- Discussionsで使用例を共有
- Pull Requestsで改善提案を受け付ける

## 📊 ベンチマーク結果（再掲）

| 項目 | Model A | AEGIS | 改善率 |
|------|---------|--------|--------|
| 正確性 | 0.723 | 0.845 | +17.1% |
| 応答時間 | 2.43秒 | 2.29秒 | -5.8% |
| 倫理適合性 | 6.8/10 | 9.2/10 | +35.3% |
| エラー耐性 | 7.2/10 | 8.9/10 | +23.6% |

---

**AEGIS**: 数理的知性で、未来を形作る。

**AEGIS**: Shaping the future with mathematical intelligence.
