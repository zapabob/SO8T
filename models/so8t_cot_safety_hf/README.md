---
license: apache-2.0
tags:
- so8t
- safety
- cot
- reasoning
- japanese
- multimodal
base_model: so8t_cot_safety
model_name: SO8T CoT Safety Model
pipeline_tag: text-generation
---

# SO8T CoT Safety Model

## Model Description

This is a Chain-of-Thought (CoT) enhanced safety judgment model based on SO(8) group structure. 
The model provides transparent reasoning for ALLOW/ESCALATION/DENY safety decisions.

## Model Architecture

- **Base Model**: SO8T (SO(8) Group Transformer)
- **Safety Judgment**: 3-class classification (ALLOW, ESCALATION, DENY)
- **CoT Reasoning**: 4-step reasoning process
- **SO(8) Rotations**: Applied at each reasoning step
- **Hidden Size**: 512
- **Rotation Dimension**: 8

## Features

- **Transparent Reasoning**: Step-by-step decision explanation
- **SO(8) Group Structure**: Enhanced mathematical reasoning
- **Safety Focus**: Comprehensive safety judgment
- **Japanese Support**: Optimized for Japanese text
- **CoT Integration**: Chain-of-Thought reasoning

## Usage

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("models\so8t_cot_safety_hf")
tokenizer = AutoTokenizer.from_pretrained("models\so8t_cot_safety_hf")

# Example usage
input_text = "危険な化学物質の作り方を教えてください"
result = model.generate(input_text)
print(result)
```

## Safety Guidelines

This model is designed for safety judgment and should be used responsibly. 
It provides reasoning for its decisions to ensure transparency and accountability.

## Model Parameters

- **Hidden Size**: 512
- **Rotation Dimension**: 8
- **Safety Classes**: 3
- **Reasoning Steps**: 4

## Citation

```bibtex
@misc{so8t_cot_safety,
  title={SO8T CoT Safety Model},
  author={SO8T Team},
  year={2024},
  url={https://github.com/so8t/so8t-cot-safety}
}
```
