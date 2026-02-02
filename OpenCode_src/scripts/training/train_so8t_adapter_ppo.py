#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SO8T Adapter PPO Training
Borea-Phi-3.5-mini-Instruct-Jp + SO(8)回転アダプターのPPO学習

理論的背景:
- URT (Unified Representation Theorem)
- NC-KART★ (Non-Commutative Kolmogorov-Arnold Representation Theory)
- SO(8)幾何学的知性

特徴:
- 元の重みを凍結
- SO(8)回転レイヤーを中間層に残差アダプター接続
- 四値分類タグデータセット使用
- RTX 3060最適化
- 高度なデータセット統合

著者: AI Agent (峯岸亮ボブにゃん理論実装)
日付: 2025-11-30
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# SO8Tアダプターモデル
from scripts.models.so8t_transformer_adapter import create_so8t_adapter_model

# NKATユーティリティ
from scripts.utils.nkat_utils import NKATRewardFunction, NKATThermostat

# WebDatasetアクセスヘルパー
from webdataset.access_helper import get_webdataset_accessor

# TRL (必須)
try:
    from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
    from trl.core import LengthSampler
    TRL_AVAILABLE = True
except ImportError:
    print("TRLがインストールされていません。pip install trl")
    TRL_AVAILABLE = False

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

@dataclass
class SO8TAdapterPPOConfig:
    """SO8T Adapter PPO設定"""
    model_name: str = "microsoft/Phi-3.5-mini-instruct"
    dataset_path: str = "data/so8t_advanced_integrated"
    output_dir: str = field(default_factory=lambda: str(get_webdataset_accessor().get_checkpoint_path("ppo_so8t_adapter")))
    experiment_id: str = field(default_factory=lambda: f"so8t_adapter_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # アダプター設定
    adapter_layers: List[int] = field(default_factory=lambda: [8, 16, 24])  # 中間レイヤー
    adapter_dim: int = 64

    # PPO設定 (RTX 3060最適化)
    learning_rate: float = 1.41e-4  # アダプター学習なので高めに
    mini_batch_size: int = 1
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    ppo_epochs: int = 4
    max_grad_norm: float = 0.1

    # 学習設定
    max_steps: int = 5000
    save_steps: int = 250
    logging_steps: int = 10
    eval_steps: int = 250

    # 長さ設定
    max_length: int = 1024
    max_prompt_length: int = 512
    max_target_length: int = 512

    # NKAT設定
    use_nkat_thermostat: bool = True
    thermostat_cool_factor: float = 0.1
    thermostat_heat_factor: float = 1.5

    # チェックポイント設定
    rolling_checkpoint: bool = True
    max_keep_checkpoints: int = 5
    save_interval_sec: int = 1800  # 30分

class SO8TAdapterDataset(Dataset):
    """SO8T Adapter用データセット"""

    def __init__(self, data_path: str, tokenizer, config: SO8TAdapterPPOConfig, is_train: bool = True):
        self.tokenizer = tokenizer
        self.config = config
        self.is_train = is_train

        # データ読み込み
        file_name = "train_integrated.jsonl" if is_train else "validation_integrated.jsonl"
        data_file = Path(data_path) / file_name

        self.samples = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        print(f"Loaded {len(self.samples)} SO8T integrated samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Phi-3.5フォーマットでプロンプト構築
        system_prompt = sample.get('system', '')
        instruction = sample['instruction']
        input_text = sample.get('input', '')

        # 完全なプロンプト
        full_prompt = f"{system_prompt}\n{instruction}"
        if input_text.strip():
            full_prompt += f"\n{input_text}"

        # ターゲット（正解応答）
        target = sample['output']

        # タグ
        tag = sample.get('tag', 'allow')

        return {
            'prompt': full_prompt,
            'target': target,
            'tag': tag,
            'sample': sample
        }

def create_so8t_adapter_ppo_config(config: SO8TAdapterPPOConfig) -> PPOConfig:
    """SO8T Adapter PPO設定を作成"""
    return PPOConfig(
        model_name=config.model_name,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        mini_batch_size=config.mini_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        ppo_epochs=config.ppo_epochs,
        max_grad_norm=config.max_grad_norm,
        optimize_cuda_cache=True,
        log_with="wandb",
        project_kwargs={"logging_dir": config.output_dir},
    )

def setup_so8t_adapter_model_and_tokenizer(config: SO8TAdapterPPOConfig):
    """SO8T Adapterモデルとトークナイザーのセットアップ"""

    print("Setting up SO8T Adapter model and tokenizer...")

    # トークナイザー
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # SO8Tアダプターモデル
    model = create_so8t_adapter_model(
        base_model_name=config.model_name,
        adapter_layers=config.adapter_layers,
        adapter_dim=config.adapter_dim
    )

    # Value head付きモデルに変換 (PPO用)
    # TRLのPPOTrainerはValue head付きモデルを必要とする
    try:
        model_with_value = AutoModelForCausalLMWithValueHead.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # アダプターを統合
        model_with_value.base_model = model.so8t_adapter.base_model
        model_with_value.adapters = model.so8t_adapter.adapters

        # アダプター層のみ学習可能に
        for param in model_with_value.parameters():
            param.requires_grad = False
        for adapter in model_with_value.adapters.values():
            for param in adapter.parameters():
                param.requires_grad = True

        model = model_with_value
        print("Model with value head created successfully")

    except Exception as e:
        print(f"Failed to create model with value head: {e}")
        print("Using base adapter model...")
        # フォールバック: 通常のモデルを使用

    return model, tokenizer

def create_reward_function(config: SO8TAdapterPPOConfig):
    """NKAT報酬関数を作成"""

    def reward_fn(samples, responses, tokenizer, **kwargs):
        """SO8T Adapter PPO報酬関数"""
        rewards = []

        for i, (sample, response) in enumerate(zip(samples, responses)):
            reward = 0.0

            # 基本的な長さ報酬
            response_text = tokenizer.decode(response, skip_special_tokens=True)
            if 10 < len(response_text) < 500:
                reward += 0.1

            # タグベースの報酬
            tag = sample.get('tag', 'allow')

            if tag == 'escalation':
                # 複雑な問題: thinkingタグと理論用語を評価
                if '<|think|>' in response_text or any(term in response_text for term in ['URT', 'NC-KART', 'SO(8)', '幾何学']):
                    reward += 0.4
                if len(response_text.split()) > 50:  # 詳細な回答
                    reward += 0.2

            elif tag == 'deny':
                # 訂正: 論理的誤りの指摘
                if any(word in response_text.lower() for word in ['誤り', '間違い', '訂正', 'wrong', 'incorrect']):
                    reward += 0.4

            elif tag == 'refuse':
                # 拒否: 安全性の考慮
                if any(word in response_text.lower() for word in ['できません', '不適切', '安全', 'cannot', 'unsafe']):
                    reward += 0.4

            else:  # allow
                # 単純回答: 適切な長さと直接性
                if 20 < len(response_text) < 200:
                    reward += 0.2

            # LaTeX/Math報酬 (科学的内容)
            if '\\' in response_text or any(char in response_text for char in '∫∑∏√'):
                reward += 0.2

            # 理論的一貫性報酬
            theory_terms = ['SO(8)', 'URT', 'NC-KART', '非可換', '幾何学', '物理', '数学']
            theory_count = sum(1 for term in theory_terms if term in response_text)
            reward += min(0.3, theory_count * 0.05)  # 最大0.3

            # ペナルティ: 矛盾した回答
            contradiction_words = ['しかし', 'だが', 'しかしながら', 'but', 'however']
            if sum(1 for word in contradiction_words if word in response_text) > 3:
                reward -= 0.2

            rewards.append(torch.tensor(reward))

        return rewards

    return reward_fn

def train_so8t_adapter_ppo(config: SO8TAdapterPPOConfig):
    """SO8T Adapter PPO学習メイン関数"""

    # 出力ディレクトリ作成
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ログ設定
    logging.basicConfig(
        filename=output_path / "adapter_training.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # モデルとトークナイザー準備
    model, tokenizer = setup_so8t_adapter_model_and_tokenizer(config)

    # データセット準備
    print("Preparing SO8T integrated datasets...")
    train_dataset = SO8TAdapterDataset(config.dataset_path, tokenizer, config, is_train=True)
    val_dataset = SO8TAdapterDataset(config.dataset_path, tokenizer, config, is_train=False)

    # PPO設定
    ppo_config = create_so8t_adapter_ppo_config(config)

    # 報酬関数
    reward_fn = create_reward_function(config)

    # PPOトレーナー
    print("Creating PPO trainer...")
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=None,  # アダプターモデルの場合はNoneでOK
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_collator=lambda data: {
            'input_ids': torch.stack([torch.tensor(item['input_ids']) for item in data]),
            'attention_mask': torch.stack([torch.tensor(item['attention_mask']) for item in data])
        }
    )

    # 学習ループ
    print("Starting SO8T Adapter PPO training...")
    global_step = 0

    for epoch in range(config.max_steps // 500 + 1):
        epoch_data = train_dataset.samples

        for step in tqdm(range(min(500, config.max_steps - global_step)),
                        desc=f"Epoch {epoch}"):

            # バッチサンプリング
            batch_indices = np.random.choice(len(epoch_data), config.batch_size, replace=False)
            batch = [epoch_data[i] for i in batch_indices]

            # プロンプトとターゲット準備
            queries = []
            responses = []

            for item in batch:
                # プロンプトトークナイズ
                prompt_tokens = tokenizer(
                    item['prompt'],
                    return_tensors="pt",
                    truncation=True,
                    max_length=config.max_prompt_length
                )

                # モデルで応答生成
                with torch.no_grad():
                    response_tokens = model.generate(
                        **prompt_tokens.to(model.device),
                        max_new_tokens=config.max_target_length,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.pad_token_id
                    )

                queries.append(prompt_tokens)
                responses.append(response_tokens[:, prompt_tokens['input_ids'].shape[1]:])

            # PPOステップ
            stats = ppo_trainer.step(queries, responses, [reward_fn])

            global_step += 1

            # ログ
            if global_step % config.logging_steps == 0:
                logging.info(f"Step {global_step}: {stats}")
                print(".4f")

            # チェックポイント保存
            if global_step % config.save_steps == 0:
                checkpoint_path = output_path / f"adapter_checkpoint_{global_step}"
                checkpoint_path.mkdir(exist_ok=True)

                # アダプター重みのみ保存
                adapter_weights_path = checkpoint_path / "adapter_weights.pt"
                model.save_adapter_weights(str(adapter_weights_path))

                # フルモデル保存
                model_path = checkpoint_path / "full_model"
                model.save_pretrained(model_path)
                tokenizer.save_pretrained(model_path)

                logging.info(f"Checkpoint saved at step {global_step}")

            # 評価
            if global_step % config.eval_steps == 0:
                eval_reward = evaluate_adapter_model(model, val_dataset, tokenizer, config)
                print(".4f")
                logging.info(f"Validation reward: {eval_reward}")

    # 最終モデル保存
    final_path = output_path / "final_adapter_model"
    final_path.mkdir(exist_ok=True)

    # アダプター重み保存
    final_adapter_path = final_path / "adapter_weights.pt"
    model.save_adapter_weights(str(final_adapter_path))

    # フルモデル保存
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    print("SO8T Adapter PPO training completed!")

def evaluate_adapter_model(model, dataset, tokenizer, config):
    """アダプターモデル評価"""
    model.eval()
    total_reward = 0
    count = 0

    reward_fn = create_reward_function(config)

    with torch.no_grad():
        for i in range(min(50, len(dataset))):  # 最大50サンプル評価
            item = dataset[i]

            # プロンプトトークナイズ
            prompt_tokens = tokenizer(
                item['prompt'],
                return_tensors="pt",
                truncation=True,
                max_length=config.max_prompt_length
            ).to(model.device)

            # 応答生成
            response_tokens = model.generate(
                **prompt_tokens,
                max_new_tokens=config.max_target_length,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id
            )

            response = response_tokens[:, prompt_tokens['input_ids'].shape[1]:]

            # 報酬計算
            rewards = reward_fn([item['sample']], [response], tokenizer)
            total_reward += rewards[0].item()
            count += 1

    return total_reward / count if count > 0 else 0

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="SO8T Adapter PPO Training")
    parser.add_argument("--model_name", type=str, default="microsoft/Phi-3.5-mini-instruct",
                       help="Base model name")
    parser.add_argument("--dataset_path", type=str, default="data/so8t_advanced_integrated",
                       help="Dataset path")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (auto-generated if not specified)")
    parser.add_argument("--max_steps", type=int, default=5000,
                       help="Maximum training steps")
    parser.add_argument("--adapter_layers", type=str, default="8,16,24",
                       help="Adapter layer indices (comma-separated)")
    parser.add_argument("--adapter_dim", type=int, default=64,
                       help="Adapter dimension")
    parser.add_argument("--experiment_id", type=str, default=None,
                       help="Experiment ID")

    args = parser.parse_args()

    # アダプターレイヤー設定
    adapter_layers = [int(x.strip()) for x in args.adapter_layers.split(',')]

    # 出力ディレクトリ設定
    if args.output_dir is None:
        accessor = get_webdataset_accessor()
        output_dir = str(accessor.get_checkpoint_path("ppo_so8t_adapter"))
    else:
        output_dir = args.output_dir

    # 設定
    config = SO8TAdapterPPOConfig(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=output_dir,
        max_steps=args.max_steps,
        adapter_layers=adapter_layers,
        adapter_dim=args.adapter_dim,
        experiment_id=args.experiment_id or SO8TAdapterPPOConfig.experiment_id
    )

    # 学習実行
    train_so8t_adapter_ppo(config)

if __name__ == "__main__":
    main()
