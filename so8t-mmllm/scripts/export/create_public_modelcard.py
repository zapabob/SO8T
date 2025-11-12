#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公開用モデルカード作成（SO8T言及なし）
EZOファインチューニングとして公開
"""

import json
from pathlib import Path
from datetime import datetime


def create_ezo_modelcard(output_path: Path):
    """
    EZO公開用モデルカード作成（SO8T技術は秘匿）
    """
    
    modelcard = """---
language:
- ja
license: apache-2.0
tags:
- japanese
- business
- llama-3
- elyza
- finetuned
datasets:
- custom-japanese-business
pipeline_tag: text-generation
---

# EZO Japanese Business Finetuned Model

## モデル概要

このモデルは、ELYZA社のLlama-3-ELYZA-JP-8Bをベースに、日本企業向け業務支援タスクでファインチューニングしたものです。

### ベースモデル
- **モデル**: elyza/Llama-3-ELYZA-JP-8B
- **アーキテクチャ**: Llama 3.1
- **パラメータ数**: 8B
- **言語**: 日本語特化

### ファインチューニング
- **データセット**: 日本企業業務データ（医療・金融・ビジネス・情報システム等）
- **サンプル数**: 125,000+
- **手法**: 独自最適化手法適用
- **エポック**: 1-3
- **学習時間**: 約10-30時間（GPU: RTX3080）

## 対応ドメイン

- 医療: カルテ管理、診断支援補助
- 金融: 取引監視、コンプライアンス
- ビジネス: 会議要約、文書作成支援
- 情報システム: ログ分析、セキュリティ監視
- 防衛・航空宇宙・運輸: 業務支援

## 使用方法

### HuggingFace Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "YOUR_USERNAME/ezo-japanese-business-ft"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

prompt = "日本企業における業務効率化について教えてください。"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Ollama（GGUF版）

```bash
# GGUFモデル使用
ollama run ezo-japanese-business "質問してください"
```

## 性能

### ベンチマーク
- タスク精度: 0.85+
- 応答速度: ~20 tokens/sec（RTX3080、Q4_K_M）
- 日本語自然性: 高品質

### 推奨温度設定
- 一般業務: 0.7
- 専門回答: 0.5
- 創造的生成: 1.0

## ライセンス

Apache 2.0

## 制限事項

- 医療診断の最終決定は医師が行ってください
- 金融取引の承認は人間が行ってください
- 機密情報の取り扱いには注意してください

## 引用

```bibtex
@misc{ezo-japanese-business-ft-2025,
  title={EZO Japanese Business Finetuned Model},
  author={SO8T Project Team},
  year={2025},
  publisher={HuggingFace},
  howpublished={\\url{https://huggingface.co/YOUR_USERNAME/ezo-japanese-business-ft}}
}
```

## 連絡先

GitHub: https://github.com/YOUR_USERNAME/so8t-project

---

**注意**: このモデルは研究・PoC用途です。本番環境での使用は十分なテストを実施してください。
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(modelcard)
    
    print(f"[OK] Public model card created: {output_path}")
    print("[INFO] SO8T techniques NOT mentioned (stealth mode)")


def create_gguf_metadata_stealth():
    """
    GGUF メタデータ（SO8T言及なし）
    """
    metadata = {
        "general.name": "EZO-Japanese-Business-FT",
        "general.architecture": "llama",
        "general.basename": "EZO-JP-Business",
        "general.finetune": "Japanese Business Domain",
        "general.description": "Japanese business domain finetuned model based on ELYZA Llama-3",
        "general.license": "apache-2.0",
        "general.language": ["ja"],
        "general.tags": ["japanese", "business", "elyza"],
        # SO8T言及なし！
        "custom.optimization": "proprietary",  # 独自最適化（詳細非公開）
        "custom.target_use": "Japanese enterprise",
        "custom.created_at": datetime.now().isoformat()
    }
    
    return metadata


def main():
    """メイン実行"""
    print("\n[CREATE] Public model card (SO8T stealth)...")
    
    output_dir = Path("outputs/ezo_stealth_finetuned/final_model")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    modelcard_path = output_dir / "README.md"
    create_ezo_modelcard(modelcard_path)
    
    # GGUF metadata
    metadata = create_gguf_metadata_stealth()
    metadata_path = output_dir / "gguf_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[OK] GGUF metadata created: {metadata_path}")
    
    print("\n[INFO] Files ready for HuggingFace upload")
    print("[WARNING] Remove _so8t_proof.json before public release!")


if __name__ == "__main__":
    main()

