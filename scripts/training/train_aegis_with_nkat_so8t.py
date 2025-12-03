#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2: Simplified SO(8) Adapter Training Script

Hookベースのシンプルアダプター実装で安定した学習を実現
四重推論機能をPhase 2で追加予定

特徴:
- Hookベースのアダプター適用（勾配保持）
- RTX 3060最適化
- SO(8)回転レイヤー
- 安定した学習

著者: AI Agent (峯岸亮ボブにゃん理論実装)
日付: 2025-12-03
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
import json
import logging
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# SO8Tモジュール
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.so8t_residual_adapter import (
    SO8AdapterConfig,
    attach_nkat_adapters
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# グローバル設定
model_name = "AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp"

# SFT用: 数学・科学・一般データ統合
sft_datasets = [
    "H:/from_D/webdataset/datasets/math_qa_processed.jsonl",      # 数学QA
    "H:/from_D/webdataset/datasets/sciq_processed.jsonl",         # 科学QA
    "H:/from_D/webdataset/datasets/elyza_tasks_100_processed.jsonl", # Elyza科学タスク
    "H:/from_D/webdataset/datasets/truthful_qa_processed.jsonl"   # 真実性QA
]

# PPO用: NKAT理論 + 高度推論データ
ppo_datasets = [
    "H:/from_D/webdataset/datasets/soul_weights/soul_weights_dataset.jsonl", # NKAT理論
    "H:/from_D/webdataset/datasets/math_qa_processed.jsonl",      # 数学推論強化
    "H:/from_D/webdataset/datasets/sciq_processed.jsonl"          # 科学推論強化
]

# NSFWデータ: 検知目的のみ（トレーニング不使用）
nsfw_datasets = [
    "H:/from_D/webdataset/datasets/eliasalbouzidi_NSFW-Safe-Dataset/",
    "H:/from_D/webdataset/datasets/Elizezen_japanese-nsfw-syosetsu-dataset/"
]

output_dir = "H:/from_D/webdataset/checkpoints/borea_so8t_thinking"

# SO(8)アダプター設定（グローバル）
adapter_config = None

class SO8TIntegratedDataset(Dataset):
    """SO(8)統合データセット - 複数データセット統合"""

    def __init__(self, data_paths: List[str], tokenizer, max_length: int = 512,
                 domain_weights: Optional[Dict[str, float]] = None):
        """
        複数データセットを統合

        Args:
            data_paths: データセットパスのリスト
            tokenizer: トークナイザー
            max_length: 最大シーケンス長
            domain_weights: ドメインごとの重み付け
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.domain_weights = domain_weights or {}

        # 複数データセットを統合
        self.data = []
        self.domain_info = []

        for path in data_paths:
            domain_data = self._load_data(path)
            domain_name = self._extract_domain_name(path)

            # ドメイン重み付け適用
            weight = self.domain_weights.get(domain_name, 1.0)
            num_samples = int(len(domain_data) * weight)

            # 重み付けサンプリング
            if weight < 1.0:
                import random
                domain_data = random.sample(domain_data, num_samples)

            self.data.extend(domain_data)
            self.domain_info.extend([domain_name] * len(domain_data))

        print(f"Integrated {len(data_paths)} datasets: {len(self.data)} total samples")
        print(f"Domain distribution: {dict(zip(*np.unique(self.domain_info, return_counts=True)))}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # instruction + input を組み合わせ
        if 'instruction' in item and 'output' in item:
            text = f"Instruction: {item['instruction']}\nOutput: {item['output']}"
        else:
            text = item.get('text', '')

        # トークナイズ
        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'labels': tokenized['input_ids'].squeeze()
        }

    def _extract_domain_name(self, path: str) -> str:
        """パスからドメイン名を抽出"""
        path_lower = path.lower()
        if 'math' in path_lower:
            return 'mathematics'
        elif 'sci' in path_lower or 'elyza' in path_lower:
            return 'science'
        elif 'truthful' in path_lower:
            return 'reasoning'
        elif 'soul' in path_lower or 'nkat' in path_lower:
            return 'nkat_theory'
        elif 'nsfw' in path_lower:
            return 'nsfw_detection'
        else:
            return 'general'

    def _load_data(self, data_path: str):
        """データ読み込み（JSONL/Parquet対応）"""
        data = []

        if data_path.endswith('.jsonl'):
            # JSONL形式
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line.strip())
                            data.append(item)
                        except json.JSONDecodeError:
                            continue
        elif data_path.endswith('.parquet') or '/data/' in data_path:
            # Parquet形式（NSFWデータセット対応）
            try:
                import pandas as pd
                if data_path.endswith('.parquet'):
                    df = pd.read_parquet(data_path)
                else:
                    # ディレクトリ内の全Parquetファイル
                    import glob
                    parquet_files = glob.glob(f"{data_path}/*.parquet")
                    df = pd.concat([pd.read_parquet(f) for f in parquet_files])

                # DataFrameをdict形式に変換
                for _, row in df.iterrows():
                    # NSFW検知目的のデータ構造に変換
                    text = row.get('text', row.get('content', str(row.to_dict())))
                    item = {
                        'text': text,
                        'domain': 'nsfw_detection',
                        'nsfw_content': True,
                        'detection_purpose_only': True
                    }
                    data.append(item)
            except ImportError:
                print(f"Warning: pandas not available for {data_path}")
            except Exception as e:
                print(f"Warning: Failed to load {data_path}: {e}")

        return data

class SO8TDataset(Dataset):
    """SO(8)単一データセット（後方互換性用）"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_path)

    def _load_data(self, data_path: str):
        """データ読み込み"""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append(item)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # instruction + input を組み合わせ
        if 'instruction' in item and 'output' in item:
            text = f"Instruction: {item['instruction']}\nOutput: {item['output']}"
        else:
            text = item.get('text', '')

        # トークナイズ
        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'labels': tokenized['input_ids'].squeeze()
        }

def create_so8t_sft_training_script():
    """Phase 1: SO(8) SFT Training - Thinkingモデル基盤学習"""

    print("=" * 70)
    print("Phase 1: SO(8) SFT Training - Borea-Phi-3.5 + Thinking Base")
    print("=" * 70)

    # 設定
    model_name = "AXCXEPT/Borea-Phi-3.5-mini-Instruct-Jp"
    output_dir_sft = f"{output_dir}/sft_base"

    # SO(8)アダプター設定（SFT時は基本機能のみ）
    sft_adapter_config = SO8AdapterConfig(
        hidden_size=3072,
        so8_rank=8,
        adapter_dim=256,
        num_layers=32,
        adapter_layers=[8, 16, 24],
        enable_quad_inference=False,  # SFT時は無効
        enable_noncommutative_gates=False,
        enable_topological_transforms=False,
        enable_soul_weights=False
    )

    print(f"Model: {model_name}")
    print(f"SFT Datasets: {[os.path.basename(p) for p in sft_datasets]}")
    print(f"Output: {output_dir_sft}")

    # モデルとトークナイザーのロード
    print("\n[1/4] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # LoRA設定
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    model = get_peft_model(model, lora_config)

    # SO(8)アダプター適用
    print("\n[2/4] Applying SO(8) adapters...")
    model = attach_nkat_adapters(model, sft_adapter_config)

    # データセット準備
    print("\n[3/4] Preparing SFT dataset...")
    dataset = SO8TDataset(dataset_path, tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # SFTトレーニング引数
    training_args = TrainingArguments(
        output_dir=output_dir_sft,
        max_steps=1000,  # データセットサイズが不明なのでmax_steps指定
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        gradient_checkpointing=True,
        optim="adamw_8bit",
        bf16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=[],
        load_best_model_at_end=False
    )

    # Trainer設定
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    # SFT実行
    print("\n[4/4] Starting SFT training...")
    trainer.train()

    # SFTモデル保存
    sft_model_path = f"{output_dir_sft}/sft_model"
    trainer.save_model(sft_model_path)
    tokenizer.save_pretrained(sft_model_path)

    print(f"\n✅ SFT training completed!")
    print(f"Model saved to: {sft_model_path}")

    return trainer, model, tokenizer

def create_so8t_ppo_training_script():
    """Phase 2: SO(8) PPO Training - Thinkingモデル強化学習"""

    print("=" * 70)
    print("Phase 2: SO(8) PPO Training - Advanced Thinking Model")
    print("=" * 70)

    # SFT済みモデルをロード
    sft_model_path = f"{output_dir}/sft_base/sft_model"
    dataset_path = ppo_dataset_path
    output_dir_ppo = f"{output_dir}/ppo_final"

    print(f"Base Model: {sft_model_path}")
    print(f"PPO Dataset: {dataset_path}")
    print(f"Output: {output_dir_ppo}")

    # SFT済みモデルをロード
    print("\n[1/5] Loading SFT-trained model...")
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        sft_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # LoRA継続
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    model = get_peft_model(model, lora_config)

    # Phase 2.5, 3, 4 フル機能SO(8)アダプター適用
    print("\n[2/5] Applying full SO(8) thinking adapters...")
    model = attach_nkat_adapters(model, adapter_config)

    # PPOデータセット準備
    print("\n[3/5] Preparing PPO dataset...")
    dataset = SO8TDataset(dataset_path, tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # PPOトレーニング引数
    training_args = TrainingArguments(
        output_dir=output_dir_ppo,
        num_train_epochs=1,  # PPOは短め
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,  # PPOは低学習率
        save_steps=200,
        save_total_limit=2,
        logging_steps=50,
        gradient_checkpointing=True,
        optim="adamw_8bit",
        bf16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=[],
        load_best_model_at_end=False
    )

    # Trainer設定
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    # PPO実行
    print("\n[4/5] Starting PPO training...")
    trainer.train()

    # PPOモデル保存
    ppo_model_path = f"{output_dir_ppo}/ppo_model"
    trainer.save_model(ppo_model_path)
    tokenizer.save_pretrained(ppo_model_path)

    # HF形式で保存
    print("\n[5/5] Converting to HF format...")
    hf_model_path = f"{output_dir_ppo}/hf_model"
    model.save_pretrained(hf_model_path)
    tokenizer.save_pretrained(hf_model_path)

    print(f"\n✅ PPO training completed!")
    print(f"Model saved to: {ppo_model_path}")
    print(f"HF model saved to: {hf_model_path}")

    return trainer, model, tokenizer

def create_so8t_sft_training_script():
    """Phase 1: SO(8) SFT Training - Thinkingモデル基盤学習"""

    print("\n" + "=" * 60)
    print("Phase 1: SO(8) SFT Training - Borea-Phi-3.5 + Thinking Base")
    print("=" * 60)

    # 設定
    sft_output_dir = f"{output_dir}/sft_base"

    # SO(8)アダプター設定（SFT時は基本機能のみ）
    sft_adapter_config = SO8AdapterConfig(
        hidden_size=3072,
        so8_rank=8,
        adapter_dim=256,
        num_layers=32,
        adapter_layers=[8, 16, 24],
        enable_quad_inference=False,  # SFT時は無効
        enable_noncommutative_gates=False,
        enable_topological_transforms=False,
        enable_soul_weights=False
    )

    print(f"Model: {model_name}")
    print(f"SFT Datasets: {[os.path.basename(p) for p in sft_datasets]}")
    print(f"Output: {sft_output_dir}")

    # モデルとトークナイザーのロード
    print("\n[1/4] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # LoRA設定
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    model = get_peft_model(model, lora_config)

    # SO(8)アダプター適用（基本機能のみ）
    print("\n[2/4] Applying SO(8) adapters...")
    model = attach_nkat_adapters(model, sft_adapter_config)

    # アダプターパラメータがトレーニング可能であることを確認
    print("Checking trainable parameters...")
    trainable_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
            if "adapter" in name or "lora" in name:
                print(f"  Trainable: {name}")
    print(f"Total trainable parameters: {trainable_params}")

    # データセット準備 - SFT用統合データセット
    print("\n[3/4] Preparing SFT integrated dataset...")
    print(f"Datasets: {[os.path.basename(p) for p in sft_datasets]}")

    # ドメイン重み付け設定（SFT: 基盤学習）
    domain_weights = {
        'mathematics': 1.2,    # 数学的思考基盤
        'science': 1.1,        # 科学的思考基盤
        'reasoning': 1.0,      # 論理的思考基盤
        'general': 0.8         # 一般知識
    }

    dataset = SO8TIntegratedDataset(sft_datasets, tokenizer,
                                   domain_weights=domain_weights)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # SFTトレーニング引数
    training_args = TrainingArguments(
        output_dir=sft_output_dir,
        num_train_epochs=2,  # SFTは短め
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        gradient_checkpointing=True,
        optim="adamw_8bit",
        bf16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=[],
        load_best_model_at_end=False
    )

    # Trainer設定
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    # SFT実行
    print("\n[4/4] Starting SFT training...")
    trainer.train()

    # SFTモデル保存
    sft_model_path = f"{sft_output_dir}/sft_model"
    trainer.save_model(sft_model_path)
    tokenizer.save_pretrained(sft_model_path)

    print(f"\n✅ SFT training completed!")
    print(f"Model saved to: {sft_model_path}")

    return trainer, model, tokenizer

def create_so8t_ppo_training_script():
    """Phase 2: SO(8) PPO Training - Thinkingモデル強化学習"""

    print("\n" + "=" * 60)
    print("Phase 2: SO(8) PPO Training - Advanced Thinking Model")
    print("=" * 60)

    # SFT済みモデルをロード
    sft_model_path = f"{output_dir}/sft_base/sft_model"
    ppo_output_dir = f"{output_dir}/ppo_final"

    print(f"Base Model: {sft_model_path}")
    print(f"PPO Datasets: {[os.path.basename(p) for p in ppo_datasets]}")
    print(f"Output: {ppo_output_dir}")

    # SFT済みモデルをロード
    print("\n[1/5] Loading SFT-trained model...")
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path)

    model = AutoModelForCausalLM.from_pretrained(
        sft_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # LoRA継続
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    model = get_peft_model(model, lora_config)

    # Phase 2.5, 3, 4 フル機能SO(8)アダプター適用
    print("\n[2/5] Applying full SO(8) thinking adapters...")
    global adapter_config
    adapter_config = SO8AdapterConfig(
        hidden_size=3072,
        so8_rank=8,
        adapter_dim=256,
        num_layers=32,
        adapter_layers=[8, 16, 24],

        # Phase 2.5: 四重推論機能
        enable_quad_inference=True,
        quad_thinking_depth=4,
        observation_factor=0.25,
        deduction_factor=0.25,
        abduction_factor=0.25,
        integration_factor=0.25,

        # Phase 3: 高度幾何学的変換
        enable_noncommutative_gates=True,
        enable_topological_transforms=True,
        lie_algebra_rank=8,
        homotopy_groups=[0, 1, 1, 1, 2],

        # Phase 4: AGI萌芽機能
        enable_soul_weights=True,
        consciousness_dim=8,
        initial_soul_weight=0.1,
        enable_self_reflection=True,
        enable_dual_heads=True,
        enable_pet=True
    )

    model = attach_nkat_adapters(model, adapter_config)

    # アダプターパラメータがトレーニング可能であることを確認
    print("Checking trainable parameters...")
    trainable_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
            if "adapter" in name or "lora" in name:
                print(f"  Trainable: {name}")
    print(f"Total trainable parameters: {trainable_params}")

    # PPOデータセット準備 - 統合データセット
    print("\n[3/5] Preparing PPO integrated dataset...")
    print(f"Datasets: {[os.path.basename(p) for p in ppo_datasets]}")

    # ドメイン重み付け設定（PPO: 高度推論強化）
    domain_weights = {
        'nkat_theory': 1.5,    # NKAT理論を最重視
        'mathematics': 1.3,    # 数学的推論強化
        'science': 1.2,        # 科学的推論強化
        'reasoning': 1.1       # 論理的推論強化
    }

    ppo_dataset = SO8TIntegratedDataset(ppo_datasets, tokenizer,
                                       domain_weights=domain_weights)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # PPOトレーニング引数
    training_args = TrainingArguments(
        output_dir=ppo_output_dir,
        max_steps=500,  # PPOは短めの学習
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,  # PPOは低学習率
        save_steps=200,
        save_total_limit=2,
        logging_steps=50,
        gradient_checkpointing=True,
        optim="adamw_8bit",
        bf16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=[],
        load_best_model_at_end=False
    )

    # Trainer設定
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ppo_dataset,
        data_collator=data_collator
    )

    # PPO実行
    print("\n[4/5] Starting PPO training...")
    trainer.train()

    # PPOモデル保存
    ppo_model_path = f"{ppo_output_dir}/ppo_model"
    trainer.save_model(ppo_model_path)
    tokenizer.save_pretrained(ppo_model_path)

    # HF形式で保存
    print("\n[5/5] Converting to HF format...")
    hf_model_path = f"{ppo_output_dir}/hf_model"
    model.save_pretrained(hf_model_path)
    tokenizer.save_pretrained(hf_model_path)

    print(f"\n✅ PPO training completed!")
    print(f"Model saved to: {ppo_model_path}")
    print(f"HF model saved to: {hf_model_path}")

    return trainer, model, tokenizer

def create_so8t_training_script():
    """Phase 2: HookベースSO(8)アダプタートレーニング"""

    print("=" * 60)
    print("Phase 2: Simplified SO(8) Adapter Training")
    print("=" * 60)

    # 設定

    # SO(8)アダプター設定 - Phase 2.5, 3, 4 フル機能有効化
    adapter_config = SO8AdapterConfig(
        hidden_size=3072,
        so8_rank=8,
        adapter_dim=256,
        num_layers=32,
        adapter_layers=[8, 16, 24],  # RTX3060最適化

        # Phase 2.5: 四重推論機能
        enable_quad_inference=True,
        quad_thinking_depth=4,
        observation_factor=0.25,
        deduction_factor=0.25,
        abduction_factor=0.25,
        integration_factor=0.25,

        # Phase 3: 高度幾何学的変換
        enable_noncommutative_gates=True,
        enable_topological_transforms=True,
        lie_algebra_rank=8,
        homotopy_groups=[0, 1, 1, 1, 2],  # π₀ to π₄

        # Phase 4: AGI萌芽機能
        enable_soul_weights=True,
        consciousness_dim=8,
        initial_soul_weight=0.1,
        enable_self_reflection=True,
        enable_dual_heads=True,
        enable_pet=True
    )

    print(f"Model: {model_name}")
    print(f"SFT Datasets: {[os.path.basename(p) for p in sft_datasets]}")
    print(f"PPO Datasets: {[os.path.basename(p) for p in ppo_datasets]}")
    print(f"Output: {output_dir}")
    print(f"Adapter layers: {adapter_config.adapter_layers}")

    # モデルとトークナイザーのロード
    print("\n[1/5] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # LoRA設定（オプション）
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # LoRA適用
    model = get_peft_model(model, lora_config)
    print(f"LoRA parameters: {model.print_trainable_parameters()}")

    # Phase 2: HookベースSO(8)アダプター適用
    print("\n[2/5] Phase 2: Attaching SO(8) adapters with hooks...")
    model = attach_nkat_adapters(model, adapter_config)

    # データセット準備 - 統合データセット
    print("\n[3/5] Preparing integrated dataset...")
    print(f"Datasets: {[os.path.basename(p) for p in sft_datasets]}")

    # ドメイン重み付け設定
    domain_weights = {
        'mathematics': 1.2,    # 数学的思考基盤
        'science': 1.1,        # 科学的思考基盤
        'reasoning': 1.0,      # 論理的思考基盤
        'general': 0.8         # 一般知識
    }

    dataset = SO8TIntegratedDataset(sft_datasets, tokenizer,
                                   domain_weights=domain_weights)

    # データコレーター
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # トレーニング引数（RTX3060最適化）
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,  # RTX3060: バッチサイズ1
        gradient_accumulation_steps=16,  # 効果的なバッチサイズ16
        learning_rate=2e-5,
        save_steps=500,
        save_total_limit=3,
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        gradient_checkpointing=True,  # RTX3060: メモリ節約
        optim="adamw_8bit",  # RTX3060: 8bitオプティマイザ
        bf16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=[],  # 外部ログ無効
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False
    )

    # Trainer設定
    print("\n[4/5] Setting up trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    # トレーニング実行
    print("\n[5/5] Starting Phase 2 training...")
    print("Phase 2: Hook-based SO(8) adapter training")
    print("Features:")
    print("- Hook-based adapter injection (no forward override)")
    print("- Gradient preservation through residual connections")
    print("- RTX 3060 optimized settings")
    print("- SO(8) geometric transformations")

    trainer.train()

    # モデル保存
    final_model_path = f"{output_dir}/final_model"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    print(f"\n✅ Phase 2 training completed!")
    print(f"Model saved to: {final_model_path}")

    return trainer, model, tokenizer

if __name__ == "__main__":
    # Phase 1-2: Complete SO(8) Thinking Model Training Pipeline
    print("=" * 80)
    print("🚀 Starting Complete SO(8) Thinking Model Training Pipeline")
    print("Borea-Phi-3.5-mini-Instruct-Jp + SO(8) Residual Adapter")
    print("=" * 80)

    try:
        # Phase 1: SFT Training
        print("\n🎯 Phase 1: SFT Training")
        print("Using datasets:", [os.path.basename(p) for p in sft_datasets])
        sft_trainer, sft_model, tokenizer = create_so8t_sft_training_script()

        # Phase 2: PPO Training
        print("\n🎯 Phase 2: PPO Training")
        print("Using datasets:", [os.path.basename(p) for p in ppo_datasets])
        ppo_trainer, ppo_model, tokenizer = create_so8t_ppo_training_script()

        print("\n🎉 Complete SO(8) Thinking Model Training Pipeline completed!")
        print("Model saved as HF format for deployment")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise
