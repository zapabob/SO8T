---
license: mit
library_name: transformers
pipeline_tag: text-generation
tags:
- conversational
---


# [BOREA model card]

![image/png](https://cdn-uploads.huggingface.co/production/uploads/657e900beaad53ff67ba84db/uPWIuw_wAOwRk9O8qlZxZ.png)


## [Model Information]
Based on phi-3.5-mini-Instruct, this model is a general-purpose model with improved performance from the base model after employing multiple tuning methods. In particular, Japanese language performance has been improved.

phi-3.5-mini-Instructをベースとして、複数のチューニング手法を採用のうえ、汎用的にベースモデルから性能を向上させたモデルです。特に日本語性能が向上しています。


### [Benchmark Results]

![image/png](https://cdn-uploads.huggingface.co/production/uploads/657e900beaad53ff67ba84db/oV2U6EnzDwNOnW-WKIMgd.png)

TODO:

### 推奨される使用ガイドライン / Recommended Usage Guidelines

1. **商用利用**: 本モデルを商用目的で使用する場合、info@axcxept.com へのメール連絡を強く推奨します。これにより、モデルの応用や改善についての協力の機会が生まれる可能性があります。

2. **クレジット表記**: 本モデルを使用または改変する際は、以下のようなクレジット表記を行うことを推奨します：
   "This project utilizes HODACHI/Borea-Phi-3.5-mini-Instruct-Jp, a model based on Phi-3.5-mini-Instruct and fine-tuned by Axcxept co., ltd."

3. **フィードバック**: モデルの使用経験に関するフィードバックを歓迎します。info@axcxept.com までご連絡ください。

これらは推奨事項であり、法的要件ではありません。

1. **Commercial Use**: If you plan to use this model for commercial purposes, we strongly encourage you to inform us via email at info@axcxept.com. This allows for potential collaboration on model applications and improvements.

2. **Attribution**: When using or adapting this model, we recommend providing attribution as follows:
   "This project utilizes HODACHI/Borea-Phi-3.5-mini-Instruct-Jp, a model based on Phi-3.5-mini-Instruct and fine-tuned by Axcxept co., ltd."

3. **Feedback**: We welcome any feedback on your experience with the model. Please feel free to email us at info@axcxept.com.

Please note that these are recommendations and not legal requirements. 


### [Usage]

Here are some code snippets to quickly get started with the model. First, run:
```bash
pip install flash_attn==2.5.8
pip install accelerate==0.31.0
pip install transformers==4.43.0
pip install -U trl
pip install pytest
```
Then, copy the snippet from the relevant section for your use case.

以下に、モデルの実行を素早く開始するためのコードスニペットをいくつか紹介します。
まず、
```bash
pip install flash_attn==2.5.8
pip install accelerate==0.31.0
pip install transformers==4.43.0
pip install -U trl
pip install pytest
```
を実行し、使用例に関連するセクションのスニペットをコピーしてください。

### [Chat Template]
```
<|system|>
あなたは日本語能力が高い高度なAIです。特別な指示がない限り日本語で返答してください。<|end|>
<|user|>
「生き物デザイナー」という職業があります。これは、自分が考えたオリジナルの生き物をデザインし、実際にDNAを編集して作り出す仕事です。あなたが生き物デザイナーである場合、どんな生き物を作りたいですか？また、その生き物が持つ特徴や能力について説明してください。
<|end|>
<|assistant|>
```


### Loading the model locally
After obtaining the Phi-3.5-mini-instruct model checkpoint, users can use this sample code for inference.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained(
    "HODACHI/Borea-Phi-3.5-mini-Instruct-Jp", 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)
tokenizer = AutoTokenizer.from_pretrained("HODACHI/Borea-Phi-3.5-mini-Instruct-Jp")

messages = [
    {"role": "system", "content": "あなたは日本語能力が高い高度なAIです。特別な指示がない限り日本語で返答してください。"},
    {"role": "user", "content": "「生き物デザイナー」という職業があります。これは、自分が考えたオリジナルの生き物をデザインし、実際にDNAを編集して作り出す仕事です。あなたが生き物デザイナーである場合、どんな生き物を作りたいですか？また、その生き物が持つ特徴や能力について説明してください。"},
]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 1024,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

output = pipe(messages, **generation_args)
print(output[0]['generated_text'])
```

Notes: If you want to use flash attention, call _AutoModelForCausalLM.from_pretrained()_ with _attn_implementation="flash_attention_2"_


### [Model Data]
#### Training Dataset]
We extracted high-quality data from Japanese Wikipedia and FineWeb to create instruction data. Our innovative training approach allows for performance improvements across various languages and domains, making the model suitable for global use despite its focus on Japanese data.

日本語のWikiデータおよび、FineWebから良質なデータのみを抽出し、Instructionデータを作成しました。このモデルでは日本語に特化させていますが、世界中のどんなユースケースでも利用可能なアプローチです。

https://huggingface.co/datasets/legacy-datasets/wikipedia
https://huggingface.co/datasets/HuggingFaceFW/fineweb

#### Data Preprocessing
We used a plain instruction tuning method to train the model on exemplary responses. This approach enhances the model's ability to understand and generate high-quality responses across various languages and contexts.

プレインストラクトチューニング手法を用いて、模範的回答を学習させました。この手法により、モデルは様々な言語やコンテキストにおいて高品質な応答を理解し生成する能力が向上しています。

#### Implementation Information
[Pre-Instruction Training] 

https://huggingface.co/instruction-pretrain/instruction-synthesizer

### [Disclaimer]
このモデルは研究開発のみを目的として提供されるものであり、実験的なプロトタイプとみなされるべきモデルです。
商業的な使用やミッションクリティカルな環境への配備を意図したものではありません。
本モデルの使用は、使用者の責任において行われるものとし、その性能および結果は保証されません。
Axcxept株式会社は、直接的、間接的、特別、偶発的、結果的な損害、または本モデルの使用から生じるいかなる損失に対しても、得られた結果にかかわらず、一切の責任を負いません。
利用者は、本モデルの使用に伴うリスクを十分に理解し、自己の判断で使用するものとします。

### [Hardware]
H100PCIe × 8(Running in 2h)

### [We are.]
[![Axcxept logo](https://cdn-uploads.huggingface.co/production/uploads/657e900beaad53ff67ba84db/8OKW86U986ywttvL2RcbG.png)](https://axcxept.com)
