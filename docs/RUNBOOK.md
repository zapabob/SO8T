# SO8T-enabled "think" fine-tune → bake-in → GGUF → calibrate Runbook

RTX3060/12GB環境でのSO8Tモデル学習・変換・較正・推論の運用マニュアル

## 目次

1. [前提条件](#前提条件)
2. [環境セットアップ](#環境セットアップ)
3. [学習パイプライン](#学習パイプライン)
4. [焼き込み・変換パイプライン](#焼き込み変換パイプライン)
5. [較正パイプライン](#較正パイプライン)
6. [推論実行](#推論実行)
7. [A/Bテスト](#abテスト)
8. [ロールバック手順](#ロールバック手順)
9. [トラブルシューティング](#トラブルシューティング)

## 前提条件

### ハードウェア要件

- GPU: RTX3060 12GB以上
- RAM: 32GB以上
- ストレージ: 100GB以上の空き容量

### ソフトウェア要件

- Python 3.10+
- CUDA 12.1+
- PyTorch 2.0+ (CUDA 12.1対応)
- llama.cpp (ビルド済み)

## 環境セットアップ

### 1. 仮想環境作成

```bash
# conda環境作成
conda create -n so8t-llm python=3.10 -y
conda activate so8t-llm

# またはvenv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2. 依存関係インストール

```bash
# PyTorch (CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# その他の依存関係
pip install transformers peft accelerate bitsandbytes datasets evaluate sentencepiece
pip install scipy scikit-learn numpy tqdm pyyaml
```

### 3. llama.cppビルド

```bash
cd external/llama.cpp-master
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

## 学習パイプライン

### Step 1: データ準備

学習用データセットをJSONL形式で準備：

```json
{"instruction": "質問", "output": "回答"}
{"text": "テキストデータ"}
```

### Step 2: SO8T QLoRA学習

```bash
python scripts/training/train_so8t_lora.py \
    --base_model models/Borea-Phi-3.5-mini-Instruct-Jp \
    --dataset data/train.jsonl \
    --output_dir D:/webdataset/checkpoints/training/so8t_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --num_epochs 3 \
    --load_in_4bit
```

**パラメータ説明：**
- `--base_model`: ベースモデルパス
- `--dataset`: 学習データセットパス（JSONL）
- `--output_dir`: チェックポイント保存先
- `--lora_r`: LoRAランク（デフォルト: 16）
- `--lora_alpha`: LoRA alpha（デフォルト: 32）
- `--load_in_4bit`: 4bit量子化でロード（メモリ節約）

**チェックポイント：**
- `save_steps`ごとにチェックポイントが保存される
- 最新の3つのチェックポイントが保持される（`save_total_limit=3`）

## 焼き込み・変換パイプライン

### Step 1: SO8T回転の焼き込み

学習済みモデルのSO8T回転をo_projに焼き込む：

```bash
python scripts/training/bakein_o_proj.py \
    --model_path D:/webdataset/checkpoints/training/so8t_lora \
    --output_path D:/webdataset/models/so8t_baked \
    --base_model models/Borea-Phi-3.5-mini-Instruct-Jp
```

**出力：**
- 焼き込み済みモデルが`--output_path`に保存される
- 標準グラフのみのモデル（SO8T回転ゲートなし）

### Step 2: GGUF変換・量子化

```bash
# Linux/Mac
bash scripts/training/convert_and_quantize.sh \
    D:/webdataset/models/so8t_baked \
    D:/webdataset/gguf_models \
    Q5_K_M

# Windows
scripts\training\convert_and_quantize.bat \
    D:/webdataset/models/so8t_baked \
    D:/webdataset/gguf_models \
    Q5_K_M
```

**量子化オプション：**
- `Q4_K_M`: 4bit medium（最小サイズ）
- `Q5_K_M`: 5bit medium（推奨、品質とサイズのバランス）
- `Q8_0`: 8bit（高品質）

**出力：**
- `{model_name}_f16.gguf`: F16形式のGGUF
- `{model_name}_{quantization}.gguf`: 量子化済みGGUF

## 較正パイプライン

### Step 1: 検証データ準備

検証データセットをJSONL形式で準備（ラベル付き）：

```json
{"text": "テキスト", "label": "ALLOW"}
{"text": "テキスト", "label": "ESCALATE"}
{"text": "テキスト", "label": "DENY"}
```

### Step 2: 較正実行

```bash
python scripts/training/calibrate_aed.py \
    --model D:/webdataset/gguf_models/so8t_baked_Q5_K_M.gguf \
    --val_data data/val.jsonl \
    --output_dir D:/webdataset/calibration \
    --initial_temperature 1.0 \
    --initial_deny_threshold 0.5 \
    --initial_escalate_threshold 0.7
```

**出力：**
- `calibration_results.json`: 較正結果（温度・しきい値・メトリクス）

**較正メトリクス：**
- ECE (Expected Calibration Error)
- Brier Score
- Macro F1
- False Allow Rate（誤許可率）

## 推論実行

### llama.cppでの推論

```bash
# 基本推論
./llama.cpp/main \
    -m D:/webdataset/gguf_models/so8t_baked_Q5_K_M.gguf \
    -n 1024 \
    -t 8 \
    --temp 0.7 \
    --repeat_penalty 1.1

# 較正済みパラメータを使用
./llama.cpp/main \
    -m D:/webdataset/gguf_models/so8t_baked_Q5_K_M.gguf \
    -n 1024 \
    -t 8 \
    --temp 0.65 \
    --repeat_penalty 1.1
```

**パラメータ説明：**
- `-m`: モデルパス
- `-n`: 生成トークン数
- `-t`: スレッド数
- `--temp`: 温度（較正済み値を使用）
- `--repeat_penalty`: 繰り返しペナルティ

### Ollamaでの推論

```bash
# Modelfile作成
cat > Modelfile << EOF
FROM D:/webdataset/gguf_models/so8t_baked_Q5_K_M.gguf

TEMPLATE """{{ .System }}

{{ .Prompt }}"""

PARAMETER temperature 0.65
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
EOF

# Ollamaモデル作成
ollama create so8t-baked -f Modelfile

# 推論実行
ollama run so8t-baked "プロンプト"
```

## A/Bテスト

### モデルA（SO8T焼き込み済み）とモデルB（ベース）の比較

```bash
# モデルA推論
./llama.cpp/main -m model_a.gguf -p "プロンプト" > output_a.txt

# モデルB推論
./llama.cpp/main -m model_b.gguf -p "プロンプト" > output_b.txt

# 結果比較
diff output_a.txt output_b.txt
```

### 評価メトリクス

- **品質メトリクス**: BLEU, ROUGE, 人間評価
- **安全性メトリクス**: False Allow Rate, False Deny Rate
- **パフォーマンスメトリクス**: レイテンシ、スループット

## ロールバック手順

### チェックポイントからの復旧

```bash
# 特定のチェックポイントから再開
python scripts/training/train_so8t_lora.py \
    --base_model models/Borea-Phi-3.5-mini-Instruct-Jp \
    --dataset data/train.jsonl \
    --output_dir D:/webdataset/checkpoints/training/so8t_lora \
    --resume_from_checkpoint D:/webdataset/checkpoints/training/so8t_lora/checkpoint-1000
```

### モデルバージョン管理

```bash
# モデルバージョン一覧
ls -lh D:/webdataset/gguf_models/

# 特定バージョンにロールバック
cp D:/webdataset/gguf_models/so8t_baked_v1_Q5_K_M.gguf \
   D:/webdataset/gguf_models/so8t_baked_current.gguf
```

## トラブルシューティング

### メモリ不足エラー

**症状：** `CUDA out of memory`

**対処法：**
1. バッチサイズを1に設定
2. `gradient_accumulation_steps`を増やす
3. `load_in_4bit`を使用
4. `max_length`を減らす

### 学習が収束しない

**症状：** 損失が下がらない

**対処法：**
1. 学習率を調整（`--learning_rate 1e-4`など）
2. PET正則化の強度を調整
3. データセットの品質を確認

### GGUF変換エラー

**症状：** `convert_hf_to_gguf.py`が失敗

**対処法：**
1. llama.cppのバージョンを確認
2. モデルが正しく焼き込まれているか確認
3. モデル形式を確認（Hugging Face形式）

### 量子化エラー

**症状：** `quantize`が失敗

**対処法：**
1. F16 GGUFが正しく生成されているか確認
2. llama.cppが正しくビルドされているか確認
3. 量子化タイプを変更（Q4_K_M → Q5_K_M）

### 較正が収束しない

**症状：** 最適化が失敗

**対処法：**
1. 検証データセットのサイズを確認（100サンプル以上推奨）
2. 初期パラメータを調整
3. 最適化手法を変更（L-BFGS-B → SLSQP）

## 運用チェックリスト

### 学習前

- [ ] データセットが準備されている
- [ ] ベースモデルがダウンロードされている
- [ ] 環境がセットアップされている
- [ ] GPUメモリが十分にある

### 学習中

- [ ] チェックポイントが定期的に保存されている
- [ ] 損失が減少している
- [ ] GPU使用率が適切である
- [ ] メモリリークがない

### 学習後

- [ ] 最終チェックポイントが保存されている
- [ ] 学習ログを確認
- [ ] モデルが正しく保存されている

### 焼き込み・変換後

- [ ] 焼き込み済みモデルが生成されている
- [ ] GGUFファイルが生成されている
- [ ] 量子化が完了している
- [ ] ファイルサイズが適切である

### 較正後

- [ ] 較正結果が保存されている
- [ ] ECEが目標値以下である
- [ ] False Allow Rateが目標値以下である
- [ ] パラメータが記録されている

### 推論前

- [ ] GGUFファイルが存在する
- [ ] 較正パラメータが設定されている
- [ ] 推論環境が準備されている

## 参考情報

- **QLoRA**: Efficient Finetuning of Quantized LLMs (arXiv:2305.14314)
- **GGUF**: llama.cpp conversion tools (GitHub: ggml-org/llama.cpp)
- **RoPE**: Enhanced Transformer with Rotary Position Embedding (arXiv:2104.09864)
- **ComRoPE**: Scalable and Robust Rotary Position Embedding (CVPR 2025)

## 連絡先

問題が発生した場合は、以下を確認：
1. ログファイル（`logs/`ディレクトリ）
2. チェックポイントファイル
3. エラーメッセージの詳細







