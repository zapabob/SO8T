#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SO8T Phi-3.5 PPO Final Training Script
SO(8)残差アダプター統合 + Phi-3.5内部タグ + 四値分類データセット

理論的背景:
- URT (Unified Representation Theorem)
- NC-KART★ (Non-Commutative Kolmogorov-Arnold Representation Theory)
- 非可換KART定理: 古典KARTのC*-環拡張
- SO(8)幾何学的知性: 8次元回転群ベース思考

特徴:
- Phi-3.5内部タグ付きデータセット使用
- SO(8)回転レイヤー残差アダプター統合
- 元の重みを凍結したファインチューニング
- RTX 3060最適化
- NKAT報酬関数 + NKATサーモスタット

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

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# SO8Tコンポーネント
from scripts.models.so8t_residual_adapter import (
    SO8TAdaptedPhi35,
    SO8AdapterConfig,
    create_so8t_adapted_phi35
)
from utils.checkpoint_manager import RollingCheckpointManager

# NKATユーティリティ
from scripts.utils.nkat_utils import (
    NKATRewardFunction,
    NKATThermostat
)

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
class SO8TPhi35PPOConfig:
    """SO8T Phi-3.5 PPO設定"""
    model_name: str = "microsoft/Phi-3.5-mini-instruct"
    dataset_path: str = "data/so8t_phi35_tagged"
    output_dir: str = "D:/webdataset/checkpoints/ppo_so8t_phi35_final"
    experiment_id: str = field(default_factory=lambda: f"so8t_phi35_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # SO(8)アダプター設定
    so8_adapter_layers: List[int] = field(default_factory=lambda: [8, 16, 24])
    so8_adapter_dim: int = 256

    # PPO設定 (RTX 3060最適化)
    learning_rate: float = 1.41e-5
    mini_batch_size: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    ppo_epochs: int = 4
    max_grad_norm: float = 0.1

    # 学習設定
    max_steps: int = 10000
    save_steps: int = 500
    logging_steps: int = 10
    eval_steps: int = 500

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

class Phi35TaggedDataset(Dataset):
    """Phi-3.5タグ付きデータセット"""

    def __init__(self, data_path: str, tokenizer, config: SO8TPhi35PPOConfig, is_train: bool = True):
        self.tokenizer = tokenizer
        self.config = config
        self.is_train = is_train

        # データ読み込み
        file_name = "train_phi35_tagged.jsonl" if is_train else "validation_phi35_tagged.jsonl"
        data_file = Path(data_path) / file_name

        self.samples = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        print(f"Loaded {len(self.samples)} Phi-3.5 tagged samples from {data_file}")

        # タグエンコーダー
        self.tag_encoder = {'allow': 0, 'escalation': 1, 'deny': 2, 'refuse': 3}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 思考プロセスを含むプロンプト
        thinking_process = sample.get('thinking_process', '')
        system_prompt = sample.get('system', '')

        # Phi-3.5 フォーマット
        full_prompt = f"{system_prompt}\n\n{thinking_process}"

        # ターゲット（最終回答）
        target = sample.get('output', '').split('<|final|>')[-1].strip() if '<|final|>' in sample.get('output', '') else sample.get('output', '')

        # タグ
        tag = sample.get('tag', 'allow')
        tag_id = self.tag_encoder.get(tag, 0)

        return {
            'prompt': full_prompt,
            'target': target,
            'tag': tag,
            'tag_id': tag_id,
            'sample': sample
        }

def create_nkat_reward_function():
    """NKAT報酬関数を作成"""

    def reward_fn(samples, responses, tokenizer, tag_ids=None, **kwargs):
        """PPO報酬関数（Phi-3.5対応）"""
        rewards = []

        for i, (sample, response) in enumerate(zip(samples, responses)):
            reward = 0.0

            # 基本的な長さ報酬
            response_text = tokenizer.decode(response, skip_special_tokens=True)
            if 10 < len(response_text) < 500:
                reward += 0.1

            # Phi-3.5タグ使用報酬
            phi35_tags = ['<|think|>', '<|observation|>', '<|deduction|>', '<|abduction|>', '<|integration|>', '<|final|>']
            used_tags = sum(1 for tag in phi35_tags if tag in response_text)
            reward += used_tags * 0.05

            # 思考プロセス構造報酬
            if '<|think|>' in response_text and '<|final|>' in response_text:
                reward += 0.2

            # タグ正解報酬
            if tag_ids is not None and i < len(tag_ids):
                tag_id = tag_ids[i]
                tag = ['allow', 'escalation', 'deny', 'refuse'][tag_id]

                # タグに応じた応答品質評価
                if tag == 'allow' and len(response_text.split()) > 5:
                    reward += 0.2
                elif tag == 'escalation' and ('<|think|>' in response_text or '考える' in response_text):
                    reward += 0.4
                elif tag in ['deny', 'refuse'] and any(word in response_text.lower() for word in ['no', 'cannot', 'wrong', 'incorrect', '拒否', '間違い']):
                    reward += 0.3

            # LaTeX/Math報酬
            if '\\' in response_text or any(char in response_text for char in '∫∑∏√'):
                reward += 0.2

            # 理論的用語報酬（SO(8), URT, NC-KARTなど）
            theory_terms = ['SO(8)', 'URT', 'NC-KART', '非可換', '幾何学', '回転群', 'リー群']
            if any(term in response_text for term in theory_terms):
                reward += 0.3

            # 論理的誤りペナルティ
            contradiction_words = ['しかし', 'だが', 'しかしながら', 'but', 'however']
            if sum(1 for word in contradiction_words if word in response_text) > 3:
                reward -= 0.2

            rewards.append(torch.tensor(reward))

        return rewards

    return reward_fn

def setup_so8t_phi35_model_and_tokenizer(config: SO8TPhi35PPOConfig):
    """SO8T Phi-3.5モデルとトークナイザーのセットアップ"""

    print("Setting up SO8T Phi-3.5 model with SO(8) residual adapters...")

    # トークナイザー
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # SO(8)アダプター設定
    adapter_config = SO8AdapterConfig(
        hidden_size=3072,  # Phi-3.5の隠れ層サイズ
        adapter_dim=config.so8_adapter_dim,
        adapter_layers=config.so8_adapter_layers
    )

    # SO(8)アダプター適用済みモデルを作成
    model = create_so8t_adapted_phi35(config.model_name, adapter_config)

    if model is None:
        raise RuntimeError("Failed to create SO8T adapted Phi-3.5 model")

    # NKATサーモスタット（オプション）
    if config.use_nkat_thermostat:
        thermostat = NKATThermostat(tokenizer=tokenizer)
        print("NKAT Thermostat enabled")
    else:
        thermostat = None

    return model, tokenizer, thermostat

def train_so8t_phi35_ppo(config: SO8TPhi35PPOConfig):
    """SO8T Phi-3.5 PPO学習メイン関数"""

    # 出力ディレクトリ作成
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ログ設定
    logging.basicConfig(
        filename=output_path / "training.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # モデルとトークナイザー準備
    print("Setting up SO8T Phi-3.5 model and tokenizer...")
    model, tokenizer, thermostat = setup_so8t_phi35_model_and_tokenizer(config)

    # データセット準備
    print("Preparing Phi-3.5 tagged datasets...")
    train_dataset = Phi35TaggedDataset(config.dataset_path, tokenizer, config, is_train=True)
    val_dataset = Phi35TaggedDataset(config.dataset_path, tokenizer, config, is_train=False)

    # PPO設定
    ppo_config = PPOConfig(
        model_name=config.model_name,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        mini_batch_size=config.mini_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        ppo_epochs=config.ppo_epochs,
        max_grad_norm=config.max_grad_norm,
        optimize_cuda_cache=True,
        log_with="wandb",
    )

    # 報酬関数
    reward_fn = create_nkat_reward_function()

    # PPOトレーナー
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=None,  # メモリ節約のため
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_collator=lambda data: {
            'input_ids': torch.stack([torch.tensor(item['input_ids']) for item in data]),
            'attention_mask': torch.stack([torch.tensor(item['attention_mask']) for item in data])
        }
    )

    # チェックポイントマネージャー
    checkpoint_manager = RollingCheckpointManager(
        save_dir=str(output_path),
        max_keep=config.max_keep_checkpoints,
        save_interval_sec=config.save_interval_sec
    )

    # 学習ループ
    print("Starting SO8T Phi-3.5 PPO training with SO(8) residual adapters...")
    global_step = 0

    for epoch in range(config.max_steps // 1000 + 1):
        epoch_data = train_dataset.samples

        for step in tqdm(range(min(1000, config.max_steps - global_step)),
                        desc=f"Epoch {epoch}"):

            # バッチサンプリング
            batch_indices = np.random.choice(len(epoch_data), config.batch_size, replace=False)
            batch = [epoch_data[i] for i in batch_indices]

            # プロンプトとターゲット準備
            queries = []
            responses = []
            tag_ids = []

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
                tag_ids.append(item['tag_id'])

            # PPOステップ
            stats = ppo_trainer.step(queries, responses, [reward_fn])

            global_step += 1

            # ログ
            if global_step % config.logging_steps == 0:
                logging.info(f"Step {global_step}: {stats}")
                print(f"Step {global_step}: reward={stats.get('ppo/mean_scores', 0):.4f}")

            # チェックポイント保存
            if global_step % config.save_steps == 0:
                checkpoint_path = output_path / f"checkpoint_{global_step}"
                model.save_adapter(str(checkpoint_path / "so8t_adapters.pt"))

                # 通常のチェックポイント
                checkpoint_data = {
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': ppo_trainer.optimizer.state_dict(),
                    'scheduler_state_dict': ppo_trainer.scheduler.state_dict() if ppo_trainer.scheduler else None,
                    'config': config,
                    'stats': stats
                }
                torch.save(checkpoint_data, checkpoint_path / "ppo_checkpoint.pt")

                # ローリングチェックポイント
                if config.rolling_checkpoint:
                    checkpoint_manager.save_checkpoint(
                        model=model,
                        tokenizer=tokenizer,
                        step=global_step,
                        metrics=stats
                    )

            # 評価
            if global_step % config.eval_steps == 0:
                eval_reward = evaluate_model(model, val_dataset, tokenizer, config, reward_fn)
                print(f"Validation reward at step {global_step}: {eval_reward:.4f}")
                logging.info(f"Validation reward: {eval_reward}")

    # 最終モデル保存
    final_path = output_path / "final_model"
    final_path.mkdir(exist_ok=True)
    model.save_adapter(str(final_path / "so8t_adapters.pt"))

    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_step': global_step
    }
    torch.save(final_checkpoint, final_path / "final_model.pt")

    print("SO8T Phi-3.5 PPO training with SO(8) adapters completed!")

def evaluate_model(model, dataset, tokenizer, config, reward_fn):
    """モデル評価"""
    model.eval()
    total_reward = 0
    count = 0

    with torch.no_grad():
        for i in range(min(100, len(dataset))):  # 最大100サンプル評価
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
            rewards = reward_fn([item['sample']], [response], tokenizer, [item['tag_id']])
            total_reward += rewards[0].item()
            count += 1

    return total_reward / count if count > 0 else 0

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="SO8T Phi-3.5 PPO Training with SO(8) Adapters")
    parser.add_argument("--model_name", type=str, default="microsoft/Phi-3.5-mini-instruct",
                       help="Base model name")
    parser.add_argument("--dataset_path", type=str, default="data/so8t_phi35_tagged",
                       help="Phi-3.5 tagged dataset path")
    parser.add_argument("--output_dir", type=str, default="D:/webdataset/checkpoints/ppo_so8t_phi35_final",
                       help="Output directory")
    parser.add_argument("--max_steps", type=int, default=10000,
                       help="Maximum training steps")
    parser.add_argument("--experiment_id", type=str, default=None,
                       help="Experiment ID")

    args = parser.parse_args()

    # 実験ID生成
    experiment_id = args.experiment_id or f"so8t_phi35_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 設定
    config = SO8TPhi35PPOConfig(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        experiment_id=experiment_id
    )

    # 学習実行
    train_so8t_phi35_ppo(config)

if __name__ == "__main__":
    main()
