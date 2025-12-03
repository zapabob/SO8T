#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データセット処理デバッグ
"""

import torch
from transformers import AutoTokenizer
from scripts.pipeline.sunshine_pipeline import SimpleDataset

# デバッグ実行
tokenizer = AutoTokenizer.from_pretrained('AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = SimpleDataset('data/aegis_v2_mathematical_enhanced_dataset.jsonl', tokenizer, max_length=128)
print(f'Dataset size: {len(dataset)}')

# サンプルデータ確認
sample = dataset[0]
print(f'Sample keys: {list(sample.keys())}')
print(f'Input IDs shape: {sample["input_ids"].shape}')
print(f'Input IDs dtype: {sample["input_ids"].dtype}')
print(f'Input IDs requires_grad: {sample["input_ids"].requires_grad}')
print(f'Attention mask shape: {sample["attention_mask"].shape}')
print(f'Labels shape: {sample["labels"].shape}')
print(f'Labels requires_grad: {sample["labels"].requires_grad}')

# テンソル作成テスト
test_tensor = torch.tensor([1, 2, 3], dtype=torch.long)
print(f'Test tensor requires_grad: {test_tensor.requires_grad}')
test_tensor.requires_grad_(True)
print(f'Test tensor requires_grad after setting: {test_tensor.requires_grad}')

# トークナイザーの出力確認
text = "Hello world, this is a test."
tokenized = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
print(f'Tokenizer output keys: {list(tokenized.keys())}')
print(f'Tokenizer input_ids requires_grad: {tokenized["input_ids"].requires_grad}')
# -*- coding: utf-8 -*-
"""
データセット処理デバッグ
"""

import torch
from transformers import AutoTokenizer
from scripts.pipeline.sunshine_pipeline import SimpleDataset

# デバッグ実行
tokenizer = AutoTokenizer.from_pretrained('AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = SimpleDataset('data/aegis_v2_mathematical_enhanced_dataset.jsonl', tokenizer, max_length=128)
print(f'Dataset size: {len(dataset)}')

# サンプルデータ確認
sample = dataset[0]
print(f'Sample keys: {list(sample.keys())}')
print(f'Input IDs shape: {sample["input_ids"].shape}')
print(f'Input IDs dtype: {sample["input_ids"].dtype}')
print(f'Input IDs requires_grad: {sample["input_ids"].requires_grad}')
print(f'Attention mask shape: {sample["attention_mask"].shape}')
print(f'Labels shape: {sample["labels"].shape}')
print(f'Labels requires_grad: {sample["labels"].requires_grad}')

# テンソル作成テスト
test_tensor = torch.tensor([1, 2, 3], dtype=torch.long)
print(f'Test tensor requires_grad: {test_tensor.requires_grad}')
test_tensor.requires_grad_(True)
print(f'Test tensor requires_grad after setting: {test_tensor.requires_grad}')

# トークナイザーの出力確認
text = "Hello world, this is a test."
tokenized = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
print(f'Tokenizer output keys: {list(tokenized.keys())}')
print(f'Tokenizer input_ids requires_grad: {tokenized["input_ids"].requires_grad}')
# -*- coding: utf-8 -*-
"""
データセット処理デバッグ
"""

import torch
from transformers import AutoTokenizer
from scripts.pipeline.sunshine_pipeline import SimpleDataset

# デバッグ実行
tokenizer = AutoTokenizer.from_pretrained('AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = SimpleDataset('data/aegis_v2_mathematical_enhanced_dataset.jsonl', tokenizer, max_length=128)
print(f'Dataset size: {len(dataset)}')

# サンプルデータ確認
sample = dataset[0]
print(f'Sample keys: {list(sample.keys())}')
print(f'Input IDs shape: {sample["input_ids"].shape}')
print(f'Input IDs dtype: {sample["input_ids"].dtype}')
print(f'Input IDs requires_grad: {sample["input_ids"].requires_grad}')
print(f'Attention mask shape: {sample["attention_mask"].shape}')
print(f'Labels shape: {sample["labels"].shape}')
print(f'Labels requires_grad: {sample["labels"].requires_grad}')

# テンソル作成テスト
test_tensor = torch.tensor([1, 2, 3], dtype=torch.long)
print(f'Test tensor requires_grad: {test_tensor.requires_grad}')
test_tensor.requires_grad_(True)
print(f'Test tensor requires_grad after setting: {test_tensor.requires_grad}')

# トークナイザーの出力確認
text = "Hello world, this is a test."
tokenized = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
print(f'Tokenizer output keys: {list(tokenized.keys())}')
print(f'Tokenizer input_ids requires_grad: {tokenized["input_ids"].requires_grad}')
# -*- coding: utf-8 -*-
"""
データセット処理デバッグ
"""

import torch
from transformers import AutoTokenizer
from scripts.pipeline.sunshine_pipeline import SimpleDataset

# デバッグ実行
tokenizer = AutoTokenizer.from_pretrained('AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = SimpleDataset('data/aegis_v2_mathematical_enhanced_dataset.jsonl', tokenizer, max_length=128)
print(f'Dataset size: {len(dataset)}')

# サンプルデータ確認
sample = dataset[0]
print(f'Sample keys: {list(sample.keys())}')
print(f'Input IDs shape: {sample["input_ids"].shape}')
print(f'Input IDs dtype: {sample["input_ids"].dtype}')
print(f'Input IDs requires_grad: {sample["input_ids"].requires_grad}')
print(f'Attention mask shape: {sample["attention_mask"].shape}')
print(f'Labels shape: {sample["labels"].shape}')
print(f'Labels requires_grad: {sample["labels"].requires_grad}')

# テンソル作成テスト
test_tensor = torch.tensor([1, 2, 3], dtype=torch.long)
print(f'Test tensor requires_grad: {test_tensor.requires_grad}')
test_tensor.requires_grad_(True)
print(f'Test tensor requires_grad after setting: {test_tensor.requires_grad}')

# トークナイザーの出力確認
text = "Hello world, this is a test."
tokenized = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
print(f'Tokenizer output keys: {list(tokenized.keys())}')
print(f'Tokenizer input_ids requires_grad: {tokenized["input_ids"].requires_grad}')
# -*- coding: utf-8 -*-
"""
データセット処理デバッグ
"""

import torch
from transformers import AutoTokenizer
from scripts.pipeline.sunshine_pipeline import SimpleDataset

# デバッグ実行
tokenizer = AutoTokenizer.from_pretrained('AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = SimpleDataset('data/aegis_v2_mathematical_enhanced_dataset.jsonl', tokenizer, max_length=128)
print(f'Dataset size: {len(dataset)}')

# サンプルデータ確認
sample = dataset[0]
print(f'Sample keys: {list(sample.keys())}')
print(f'Input IDs shape: {sample["input_ids"].shape}')
print(f'Input IDs dtype: {sample["input_ids"].dtype}')
print(f'Input IDs requires_grad: {sample["input_ids"].requires_grad}')
print(f'Attention mask shape: {sample["attention_mask"].shape}')
print(f'Labels shape: {sample["labels"].shape}')
print(f'Labels requires_grad: {sample["labels"].requires_grad}')

# テンソル作成テスト
test_tensor = torch.tensor([1, 2, 3], dtype=torch.long)
print(f'Test tensor requires_grad: {test_tensor.requires_grad}')
test_tensor.requires_grad_(True)
print(f'Test tensor requires_grad after setting: {test_tensor.requires_grad}')

# トークナイザーの出力確認
text = "Hello world, this is a test."
tokenized = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
print(f'Tokenizer output keys: {list(tokenized.keys())}')
print(f'Tokenizer input_ids requires_grad: {tokenized["input_ids"].requires_grad}')
# -*- coding: utf-8 -*-
"""
データセット処理デバッグ
"""

import torch
from transformers import AutoTokenizer
from scripts.pipeline.sunshine_pipeline import SimpleDataset

# デバッグ実行
tokenizer = AutoTokenizer.from_pretrained('AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = SimpleDataset('data/aegis_v2_mathematical_enhanced_dataset.jsonl', tokenizer, max_length=128)
print(f'Dataset size: {len(dataset)}')

# サンプルデータ確認
sample = dataset[0]
print(f'Sample keys: {list(sample.keys())}')
print(f'Input IDs shape: {sample["input_ids"].shape}')
print(f'Input IDs dtype: {sample["input_ids"].dtype}')
print(f'Input IDs requires_grad: {sample["input_ids"].requires_grad}')
print(f'Attention mask shape: {sample["attention_mask"].shape}')
print(f'Labels shape: {sample["labels"].shape}')
print(f'Labels requires_grad: {sample["labels"].requires_grad}')

# テンソル作成テスト
test_tensor = torch.tensor([1, 2, 3], dtype=torch.long)
print(f'Test tensor requires_grad: {test_tensor.requires_grad}')
test_tensor.requires_grad_(True)
print(f'Test tensor requires_grad after setting: {test_tensor.requires_grad}')

# トークナイザーの出力確認
text = "Hello world, this is a test."
tokenized = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
print(f'Tokenizer output keys: {list(tokenized.keys())}')
print(f'Tokenizer input_ids requires_grad: {tokenized["input_ids"].requires_grad}')
# -*- coding: utf-8 -*-
"""
データセット処理デバッグ
"""

import torch
from transformers import AutoTokenizer
from scripts.pipeline.sunshine_pipeline import SimpleDataset

# デバッグ実行
tokenizer = AutoTokenizer.from_pretrained('AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = SimpleDataset('data/aegis_v2_mathematical_enhanced_dataset.jsonl', tokenizer, max_length=128)
print(f'Dataset size: {len(dataset)}')

# サンプルデータ確認
sample = dataset[0]
print(f'Sample keys: {list(sample.keys())}')
print(f'Input IDs shape: {sample["input_ids"].shape}')
print(f'Input IDs dtype: {sample["input_ids"].dtype}')
print(f'Input IDs requires_grad: {sample["input_ids"].requires_grad}')
print(f'Attention mask shape: {sample["attention_mask"].shape}')
print(f'Labels shape: {sample["labels"].shape}')
print(f'Labels requires_grad: {sample["labels"].requires_grad}')

# テンソル作成テスト
test_tensor = torch.tensor([1, 2, 3], dtype=torch.long)
print(f'Test tensor requires_grad: {test_tensor.requires_grad}')
test_tensor.requires_grad_(True)
print(f'Test tensor requires_grad after setting: {test_tensor.requires_grad}')

# トークナイザーの出力確認
text = "Hello world, this is a test."
tokenized = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
print(f'Tokenizer output keys: {list(tokenized.keys())}')
print(f'Tokenizer input_ids requires_grad: {tokenized["input_ids"].requires_grad}')
# -*- coding: utf-8 -*-
"""
データセット処理デバッグ
"""

import torch
from transformers import AutoTokenizer
from scripts.pipeline.sunshine_pipeline import SimpleDataset

# デバッグ実行
tokenizer = AutoTokenizer.from_pretrained('AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = SimpleDataset('data/aegis_v2_mathematical_enhanced_dataset.jsonl', tokenizer, max_length=128)
print(f'Dataset size: {len(dataset)}')

# サンプルデータ確認
sample = dataset[0]
print(f'Sample keys: {list(sample.keys())}')
print(f'Input IDs shape: {sample["input_ids"].shape}')
print(f'Input IDs dtype: {sample["input_ids"].dtype}')
print(f'Input IDs requires_grad: {sample["input_ids"].requires_grad}')
print(f'Attention mask shape: {sample["attention_mask"].shape}')
print(f'Labels shape: {sample["labels"].shape}')
print(f'Labels requires_grad: {sample["labels"].requires_grad}')

# テンソル作成テスト
test_tensor = torch.tensor([1, 2, 3], dtype=torch.long)
print(f'Test tensor requires_grad: {test_tensor.requires_grad}')
test_tensor.requires_grad_(True)
print(f'Test tensor requires_grad after setting: {test_tensor.requires_grad}')

# トークナイザーの出力確認
text = "Hello world, this is a test."
tokenized = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
print(f'Tokenizer output keys: {list(tokenized.keys())}')
print(f'Tokenizer input_ids requires_grad: {tokenized["input_ids"].requires_grad}')
# -*- coding: utf-8 -*-
"""
データセット処理デバッグ
"""

import torch
from transformers import AutoTokenizer
from scripts.pipeline.sunshine_pipeline import SimpleDataset

# デバッグ実行
tokenizer = AutoTokenizer.from_pretrained('AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = SimpleDataset('data/aegis_v2_mathematical_enhanced_dataset.jsonl', tokenizer, max_length=128)
print(f'Dataset size: {len(dataset)}')

# サンプルデータ確認
sample = dataset[0]
print(f'Sample keys: {list(sample.keys())}')
print(f'Input IDs shape: {sample["input_ids"].shape}')
print(f'Input IDs dtype: {sample["input_ids"].dtype}')
print(f'Input IDs requires_grad: {sample["input_ids"].requires_grad}')
print(f'Attention mask shape: {sample["attention_mask"].shape}')
print(f'Labels shape: {sample["labels"].shape}')
print(f'Labels requires_grad: {sample["labels"].requires_grad}')

# テンソル作成テスト
test_tensor = torch.tensor([1, 2, 3], dtype=torch.long)
print(f'Test tensor requires_grad: {test_tensor.requires_grad}')
test_tensor.requires_grad_(True)
print(f'Test tensor requires_grad after setting: {test_tensor.requires_grad}')

# トークナイザーの出力確認
text = "Hello world, this is a test."
tokenized = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
print(f'Tokenizer output keys: {list(tokenized.keys())}')
print(f'Tokenizer input_ids requires_grad: {tokenized["input_ids"].requires_grad}')
# -*- coding: utf-8 -*-
"""
データセット処理デバッグ
"""

import torch
from transformers import AutoTokenizer
from scripts.pipeline.sunshine_pipeline import SimpleDataset

# デバッグ実行
tokenizer = AutoTokenizer.from_pretrained('AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = SimpleDataset('data/aegis_v2_mathematical_enhanced_dataset.jsonl', tokenizer, max_length=128)
print(f'Dataset size: {len(dataset)}')

# サンプルデータ確認
sample = dataset[0]
print(f'Sample keys: {list(sample.keys())}')
print(f'Input IDs shape: {sample["input_ids"].shape}')
print(f'Input IDs dtype: {sample["input_ids"].dtype}')
print(f'Input IDs requires_grad: {sample["input_ids"].requires_grad}')
print(f'Attention mask shape: {sample["attention_mask"].shape}')
print(f'Labels shape: {sample["labels"].shape}')
print(f'Labels requires_grad: {sample["labels"].requires_grad}')

# テンソル作成テスト
test_tensor = torch.tensor([1, 2, 3], dtype=torch.long)
print(f'Test tensor requires_grad: {test_tensor.requires_grad}')
test_tensor.requires_grad_(True)
print(f'Test tensor requires_grad after setting: {test_tensor.requires_grad}')

# トークナイザーの出力確認
text = "Hello world, this is a test."
tokenized = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
print(f'Tokenizer output keys: {list(tokenized.keys())}')
print(f'Tokenizer input_ids requires_grad: {tokenized["input_ids"].requires_grad}')
# -*- coding: utf-8 -*-
"""
データセット処理デバッグ
"""

import torch
from transformers import AutoTokenizer
from scripts.pipeline.sunshine_pipeline import SimpleDataset

# デバッグ実行
tokenizer = AutoTokenizer.from_pretrained('AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = SimpleDataset('data/aegis_v2_mathematical_enhanced_dataset.jsonl', tokenizer, max_length=128)
print(f'Dataset size: {len(dataset)}')

# サンプルデータ確認
sample = dataset[0]
print(f'Sample keys: {list(sample.keys())}')
print(f'Input IDs shape: {sample["input_ids"].shape}')
print(f'Input IDs dtype: {sample["input_ids"].dtype}')
print(f'Input IDs requires_grad: {sample["input_ids"].requires_grad}')
print(f'Attention mask shape: {sample["attention_mask"].shape}')
print(f'Labels shape: {sample["labels"].shape}')
print(f'Labels requires_grad: {sample["labels"].requires_grad}')

# テンソル作成テスト
test_tensor = torch.tensor([1, 2, 3], dtype=torch.long)
print(f'Test tensor requires_grad: {test_tensor.requires_grad}')
test_tensor.requires_grad_(True)
print(f'Test tensor requires_grad after setting: {test_tensor.requires_grad}')

# トークナイザーの出力確認
text = "Hello world, this is a test."
tokenized = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
print(f'Tokenizer output keys: {list(tokenized.keys())}')
print(f'Tokenizer input_ids requires_grad: {tokenized["input_ids"].requires_grad}')
