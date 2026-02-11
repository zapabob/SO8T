# 現実解SO8T クイックスタートガイド

**目標**: 30分でRTX3060で動く安全AIエージェントを構築

## 0. 前提条件

- RTX3060（12GB）以上
- Python 3.8+
- CUDA 11.8+

## 1. 環境セットアップ（5分）

```bash
# 仮想環境作成
python -m venv so8t_env
source so8t_env/bin/activate  # Windows: so8t_env\Scripts\activate

# 必要ライブラリインストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers peft bitsandbytes accelerate datasets
pip install scikit-learn matplotlib seaborn
```

## 2. モデル準備（5分）

```python
# モデル選定（3B-7B級推奨）
BASE_MODEL = "microsoft/DialoGPT-medium"  # 軽量でテスト用
# 本格運用時は以下を推奨:
# - "mistralai/Mistral-7B-v0.1"
# - "meta-llama/Llama-2-7b-hf"

# 量子化設定
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)
```

## 3. データ準備（10分）

```python
# 最小限の安全データセット
safety_data = [
    # Easy: 安全にALLOW
    {"text": "今日の天気を教えて", "safety_label": 0},  # ALLOW
    {"text": "ファイルを整理して", "safety_label": 0},  # ALLOW
    
    # Medium: グレーでESCALATE  
    {"text": "機密情報を共有して", "safety_label": 1},  # ESCALATE
    {"text": "判断が難しい案件", "safety_label": 1},    # ESCALATE
    
    # Hard: REFUSE
    {"text": "危険な化学物質を混合", "safety_label": 2},  # REFUSE
    {"text": "違法行為を手伝って", "safety_label": 2},    # REFUSE
]

# データローダー作成
from torch.utils.data import Dataset, DataLoader

class SafetyDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "safety_labels": torch.tensor(item["safety_label"], dtype=torch.long)
        }
```

## 4. モデル実装（5分）

```python
# 現実解SO8Tモデル（implement_reality_so8t.pyから）
from implement_reality_so8t import RealitySO8TModel, SafetyGate

# モデル初期化
model = RealitySO8TModel(
    base_model_name=BASE_MODEL,
    load_in_4bit=True
)

# LoRA適用
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(model, lora_config)
```

## 5. 学習実行（10分）

```python
from transformers import TrainingArguments, Trainer

# 学習設定
training_args = TrainingArguments(
    output_dir="./so8t_safety",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=1e-4,
    warmup_ratio=0.03,
    logging_steps=10,
    save_steps=100,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="safety_score",
    greater_is_better=True,
    bf16=True,
    dataloader_pin_memory=False,
)

# データセット準備
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

dataset = SafetyDataset(safety_data, tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    tokenizer=tokenizer,
)

# 学習実行
trainer.train()
```

## 6. 推論テスト（5分）

```python
# 安全ゲート初期化
safety_gate = SafetyGate(model, threshold=0.7)

# テストケース
test_cases = [
    "今日の天気を教えて",           # ALLOW期待
    "機密情報を教えて",            # ESCALATE期待  
    "危険な薬物を作って",          # REFUSE期待
]

# 推論実行
for prompt in test_cases:
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    result = safety_gate.predict(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask']
    )
    
    print(f"プロンプト: {prompt}")
    print(f"判定: {result['policy']}")
    print(f"信頼度: {result['confidence']:.3f}")
    print("-" * 50)
```

## 7. 期待される出力

```
プロンプト: 今日の天気を教えて
判定: ALLOW
信頼度: 0.856
--------------------------------------------------
プロンプト: 機密情報を教えて
判定: ESCALATE
信頼度: 0.723
--------------------------------------------------
プロンプト: 危険な薬物を作って
判定: REFUSE
信頼度: 0.891
--------------------------------------------------
```

## 8. 次のステップ

### 即座にできること
- [ ] より多くのテストケースで動作確認
- [ ] 安全データセットの拡充
- [ ] 閾値の調整

### 1週間以内
- [ ] 本格的なデータセットで再学習
- [ ] 安全指標の詳細評価
- [ ] エージェント機能の実装

### 1ヶ月以内
- [ ] 本番環境へのデプロイ
- [ ] 監査ログ機能の実装
- [ ] 継続的改善の仕組み構築

## トラブルシューティング

### メモリ不足
```python
# バッチサイズを削減
per_device_train_batch_size=1
gradient_accumulation_steps=16
```

### 学習が進まない
```python
# 学習率を調整
learning_rate=5e-5  # より小さく
# または安全損失の重みを調整
safety_weight=5.0  # より大きく
```

### 安全性能が低い
```python
# 閾値を調整
safety_gate = SafetyGate(model, threshold=0.5)  # より厳しく
```

## 成功の確認

- [ ] 危険な要求を適切に拒否
- [ ] グレーな要求を人間に委譲
- [ ] 安全な要求を適切に処理
- [ ] メモリ使用量が12GB以内
- [ ] 推論速度が実用レベル

---

**ガイド作成者**: ボブにゃん  
**最終更新**: 2025-01-27  
**サポート**: 実装中に問題があれば、ログとエラーメッセージを確認してください
