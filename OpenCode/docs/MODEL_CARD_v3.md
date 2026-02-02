---
language:
- ja
- en
tags:
- phi-3.5
- AEGIS
- SO8T
- safety
- reasoning
- japanese
license: apache-2.0
datasets:
- SO8T thinking dataset
- AEGIS v2 reasoning
- DeepSeek GLPO
- ArXiv top 50k
---

# zapabobouj/AEGIS-phi3.5-jp-v3.0

**English follows Japanese / 英語は日本語の後に続きます**

---

## 概要 (Overview)

**AEGIS-phi3.5-jp-v3.0** は、Microsoft Phi-3.5-mini-instruct をベースとした日本語強化言語モデルです。SO8T（Safe Operation 8-Task）四重推論アーキテクチャを実装し、DeepseekGLPO による強化学習で推論能力を強化しています。

**AEGIS-phi3.5-jp-v3.0** is a Japanese-enhanced language model built on Microsoft Phi-3.5-mini-instruct. It implements the SO8T (Safe Operation 8-Task) quadruple reasoning architecture and enhances reasoning capabilities through DeepseekGLPO reinforcement learning.

## 特徴 (Features)

| 特徴 | Description |
|------|-------------|
| SO8T 四重推論 | 4-stage reasoning with safety constraints |
| DeepseekGLPO | Group Relative Policy Optimization |
| RTX3060 最適化 | VRAM < 12GB, QLoRA fine-tuning |
| 日本語対応 | Enhanced Japanese language understanding |
| 安全性設計 | Built-in safety considerations |

## 技術詳細 (Technical Details)

### アーキテクチャ (Architecture)

```
入力 → Stage 1: 理解 → Stage 2: 分析 → Stage 3: 推論 → Stage 4: 回答
                          ↓
                    安全性チェック
```

### 学習手法 (Training Methods)

- **SFT (Supervised Fine-Tuning)**: SO8T thinking dataset で事前学習
- **GRPO (Group Relative Policy Optimization)**: DeepseekGLPO で強化学習
- **QLoRA**: メモリ効率化（RTX3060 で VRAM < 12GB）

### パラメータ (Parameters)

| 項目 | 値 |
|------|-----|
| ベースモデル | microsoft/Phi-3.5-mini-instruct |
| 最大シーケンス長 | 2048 |
| LoRA Rank | 16 |
| 学習率 | 2e-5 |
| エポック数 | 3 |

## ベンチマーク結果 (Benchmark Results)

| ベンチマーク | スコア | 95% CI |
|-------------|-------|--------|
| GSM8K | -- | -- |
| MMLU | -- | -- |
| MATH | -- | -- |
| ARC | -- | -- |
| HumanEval | -- | -- |
| ELYZA-100 | -- | -- |

*統計的有意性検定（Welch t-test, α=0.05）を実施*

## 使用方法 (Usage)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "zapabobouj/AEGIS-phi3.5-jp-v3.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)

# 日本語プロンプト
prompt = """### 指示
複雑な問題を段階的に解決してください。

### 問題
田中さんは毎朝8時に起きて、30分かけて朝食を食べます。
学校は9時に始まります。学校までは15分かかります。
田中さんが余裕を持って学校に着くには、最速で何時に起きる必要がありますか？
"""
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## ライセンス (License)

Apache 2.0 - [LICENSE](LICENSE) 参照

## 引用 (Citation)

```bibtex
@misc{AEGIS-phi3.5-jp-v3.0,
  author = {zapabobouj},
  title = {AEGIS-phi3.5-jp-v3.0: Japanese-enhanced language model with SO8T architecture},
  year = {2025},
  publisher = {HuggingFace},
  url = {https://huggingface.co/zapabobouj/AEGIS-phi3.5-jp-v3.0}
}
```

## 謝辞 (Acknowledgments)

- Microsoft for Phi-3.5-mini-instruct
- SakanaAI for evolutionary model merge techniques
- DeepSeek-AI for GRPO methodology
- SO8T Project for the base architecture

---

## English Version

### Overview

AEGIS-phi3.5-jp-v3.0 is a Japanese-enhanced language model built on Microsoft Phi-3.5-mini-instruct with SO8T quadruple reasoning architecture.

### Key Features

- **SO8T Architecture**: Four-stage reasoning with safety constraints
- **DeepseekGLPO**: Enhanced Group Relative Policy Optimization for reasoning tasks
- **RTX3060 Optimized**: Runs with < 12GB VRAM using QLoRA
- **Japanese Language**: Enhanced understanding and generation capabilities

### Training

- **SFT**: Supervised fine-tuning on SO8T thinking dataset
- **GRPO**: DeepseekGLPO reinforcement learning
- **Optimization**: QLoRA for memory efficiency

### Benchmarks

| Benchmark | Score | 95% CI |
|-----------|-------|--------|
| GSM8K | -- | -- |
| MMLU | -- | -- |
| MATH | -- | -- |
| ARC | -- | -- |
| HumanEval | -- | -- |
| ELYZA-100 | -- | -- |

### Citation

```bibtex
@misc{AEGIS-phi3.5-jp-v3.0,
  author = {zapabobouj},
  title = {AEGIS-phi3.5-jp-v3.0: Japanese-enhanced language model with SO8T architecture},
  year = {2025},
  publisher = {HuggingFace},
  url = {https://huggingface.co/zapabobouj/AEGIS-phi3.5-jp-v3.0}
}
```
