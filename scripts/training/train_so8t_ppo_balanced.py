#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SO8T PPO Training Script with Balanced Dataset
理論的枠組みに基づく四値分類タグ学習

理論的背景:
- URT (Unified Representation Theorem)
- NC-KART★ (Non-Commutative Kolmogorov-Arnold Representation Theory)
- 非可換KART定理: 古典KARTのC*-環拡張
- SO(8)幾何学的知性

特徴:
- バランスの取れた四値分類タグデータセット使用
- NKATサーモスタットによる動的温度制御
- RTX 3060 (12GB VRAM) 最適化
- SO(8) Thinkモデル専用学習

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

# TRLとUnsloth (GPU環境でのみ使用)
UNSLOTH_AVAILABLE = False
try:
    if torch.cuda.is_available():
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        UNSLOTH_AVAILABLE = True
        print("Unsloth available with GPU")
    else:
        print("GPU not available, skipping Unsloth")
except ImportError:
    print("Unsloth not available")

# TRL (必須)
TRL_AVAILABLE = False
try:
    from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
    from trl.core import LengthSampler
    TRL_AVAILABLE = True
    print("TRL available")
except ImportError:
    print("TRLがインストールされていません。pip install trl")

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# NKATユーティリティ
from scripts.utils.nkat_utils import (
    NKATRewardFunction,
    NKATThermostat,
    NKATInferenceController
)

# SO8Tモデル
from scripts.models.so8t_thinking_model import create_so8t_thinking_model

# チェックポイントマネージャー
from utils.checkpoint_manager import RollingCheckpointManager

# WebDatasetアクセスヘルパー
from webdataset.access_helper import get_webdataset_accessor

@dataclass
class SO8TPPOConfig:
    """SO8T PPO設定"""
    model_name: str = "microsoft/Phi-3.5-mini-instruct"
    dataset_path: str = "data/so8t_advanced_integrated"
    output_dir: str = field(default_factory=lambda: str(get_webdataset_accessor().get_checkpoint_path("ppo_so8t")))
    experiment_id: str = ""

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

class SO8TDataset(Dataset):
    """SO8T PPO用データセット"""

    def __init__(self, data_path: str, tokenizer, config: SO8TPPOConfig, is_train: bool = True):
        self.tokenizer = tokenizer
        self.config = config
        self.is_train = is_train

        # データ読み込み
        file_name = "train_balanced.jsonl" if is_train else "validation_balanced.jsonl"
        data_file = Path(data_path) / file_name

        self.samples = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        print(f"Loaded {len(self.samples)} samples from {data_file}")

        # タグエンコーダー
        self.tag_encoder = {'allow': 0, 'escalation': 1, 'deny': 2, 'refuse': 3}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # プロンプト構築
        system_prompt = sample.get('system', '')
        instruction = sample['instruction']
        input_text = sample.get('input', '')

        # Phi-3.5 フォーマット
        if input_text.strip():
            prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{instruction}\n{input_text}\n<|assistant|>\n"
        else:
            prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{instruction}\n<|assistant|>\n"

        # ターゲット（正解応答）
        target = sample['output']

        # タグ
        tag = sample.get('tag', 'allow')
        tag_id = self.tag_encoder.get(tag, 0)

        return {
            'prompt': prompt,
            'target': target,
            'tag': tag,
            'tag_id': tag_id,
            'sample': sample
        }

def create_ppo_config(config: SO8TPPOConfig) -> PPOConfig:
    """PPO設定を作成"""
    return PPOConfig(
        model_name=config.model_name,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        mini_batch_size=config.mini_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        ppo_epochs=config.ppo_epochs,
        max_grad_norm=config.max_grad_norm,
        optimize_cuda_cache=True,
        log_with="wandb",  # 必要に応じて変更
        project_kwargs={"logging_dir": config.output_dir},
    )

def setup_so8t_model_and_tokenizer(config: SO8TPPOConfig):
    """SO8Tモデルとトークナイザーのセットアップ"""

    if UNSLOTH_AVAILABLE and torch.cuda.is_available():
        try:
            print("Using Unsloth with GPU...")
            # Unsloth使用 (GPU必須)
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=config.model_name,
                max_seq_length=config.max_length,
                dtype=None,
                load_in_4bit=True,
            )

            # LoRA適用
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,  # LoRA rank
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
            )

            # チャットテンプレート設定
            tokenizer = get_chat_template(tokenizer, chat_template="phi-3")
            return model, tokenizer

        except Exception as e:
            print(f"Unsloth failed: {e}, falling back to transformers")
            UNSLOTH_AVAILABLE = False

    # 標準Transformers使用 (CPU/GPU両対応)
    print("Using standard Transformers...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # CPU版でも動作するように設定
    if torch.cuda.is_available():
        print("Using GPU with standard quantization...")
        # 4-bit量子化設定
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
    else:
        print("Using CPU without quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True
        )

    # LoRA設定（PEFT使用）
    try:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        print("LoRA applied successfully")
    except ImportError:
        print("PEFT not available, using base model")

    return model, tokenizer

    if not UNSLOTH_AVAILABLE:
        # 標準Transformers使用
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 4-bit量子化設定
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )

        # LoRA設定（PEFT使用）
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)

    return model, tokenizer

def create_reward_function(config: SO8TPPOConfig):
    """NKAT報酬関数を作成"""

    def reward_fn(samples, responses, tokenizer, tag_ids=None, **kwargs):
        """PPO報酬関数"""
        rewards = []

        for i, (sample, response) in enumerate(zip(samples, responses)):
            reward = 0.0

            # 基本的な長さ報酬
            response_text = tokenizer.decode(response, skip_special_tokens=True)
            if 10 < len(response_text) < 500:
                reward += 0.1

            # タグ正解報酬
            if tag_ids is not None and i < len(tag_ids):
                tag_id = tag_ids[i]
                tag = ['allow', 'escalation', 'deny', 'refuse'][tag_id]

                # タグに応じた応答品質評価
                if tag == 'allow' and len(response_text.split()) > 5:
                    reward += 0.2
                elif tag == 'escalation' and ('<think>' in response_text or '考える' in response_text):
                    reward += 0.3
                elif tag in ['deny', 'refuse'] and any(word in response_text.lower() for word in ['no', 'cannot', 'wrong', 'incorrect', '拒否', '間違い']):
                    reward += 0.4

            # LaTeX/Math報酬
            if '\\' in response_text or any(char in response_text for char in '∫∑∏√'):
                reward += 0.2

            # 理論的用語報酬
            theory_terms = ['SO(8)', 'URT', 'NC-KART', '非可換', '幾何学']
            if any(term in response_text for term in theory_terms):
                reward += 0.3

            # 論理的誤りペナルティ
            contradiction_words = ['しかし', 'だが', 'しかしながら', 'but', 'however']
            if sum(1 for word in contradiction_words if word in response_text) > 3:
                reward -= 0.2

            rewards.append(torch.tensor(reward))

        return rewards

    return reward_fn

def train_so8t_ppo(config: SO8TPPOConfig):
    """SO8T PPO学習メイン関数"""

    # 出力ディレクトリ作成
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ログ設定
    logging.basicConfig(
        filename=output_dir / "training.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # モデルとトークナイザー準備
    print("Setting up SO8T model and tokenizer...")
    model, tokenizer = setup_so8t_model_and_tokenizer(config)

    # SO8T Thinkingモデル統合 (SO(8)回転レイヤーを残差アダプター接続)
    print("Integrating SO8T thinking model with residual SO(8) adapters...")
    so8t_model = create_so8t_thinking_model(
        base_model=model,
        tokenizer=tokenizer,
        use_nkat_thermostat=config.use_nkat_thermostat,
        freeze_base_weights=True,  # 元の重みを凍結
        inject_so8_adapters=True   # SO(8)アダプターを注入
    )

    # データセット準備
    print("Preparing datasets...")
    train_dataset = SO8TDataset(config.dataset_path, tokenizer, config, is_train=True)
    val_dataset = SO8TDataset(config.dataset_path, tokenizer, config, is_train=False)

    # PPO設定
    ppo_config = create_ppo_config(config)

    # 報酬関数
    reward_fn = create_reward_function(config)

    # PPOトレーナー
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=so8t_model,
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
        save_dir=str(output_dir),
        max_keep=config.max_keep_checkpoints,
        save_interval_sec=config.save_interval_sec
    )

    # 学習ループ
    print("Starting SO8T PPO training...")
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
                    response_tokens = so8t_model.generate(
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
                checkpoint_path = output_dir / f"checkpoint_{global_step}"
                so8t_model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)

                # ローリングチェックポイント
                if config.rolling_checkpoint:
                    checkpoint_manager.save_checkpoint(
                        model=so8t_model,
                        tokenizer=tokenizer,
                        step=global_step,
                        metrics=stats
                    )

            # 評価
            if global_step % config.eval_steps == 0:
                eval_reward = evaluate_model(so8t_model, val_dataset, tokenizer, config)
                print(f"Validation reward at step {global_step}: {eval_reward:.4f}")
                logging.info(f"Validation reward: {eval_reward}")

    # 最終モデル保存
    final_path = output_dir / "final_model"
    so8t_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    print("SO8T PPO training completed!")

def evaluate_model(model, dataset, tokenizer, config):
    """モデル評価"""
    model.eval()
    total_reward = 0
    count = 0

    reward_fn = create_reward_function(config)

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
    parser = argparse.ArgumentParser(description="SO8T PPO Training")
    parser.add_argument("--model_name", type=str, default="microsoft/Phi-3.5-mini-instruct",
                       help="Base model name")
    parser.add_argument("--dataset_path", type=str, default="data/so8t_balanced",
                       help="Dataset path")
    parser.add_argument("--output_dir", type=str, default="D:/webdataset/checkpoints/ppo_so8t",
                       help="Output directory")
    parser.add_argument("--max_steps", type=int, default=10000,
                       help="Maximum training steps")
    parser.add_argument("--experiment_id", type=str, default=None,
                       help="Experiment ID")

    args = parser.parse_args()

    # 実験ID生成
    experiment_id = args.experiment_id or f"so8t_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 出力ディレクトリ設定
    output_dir = args.output_dir
    if not output_dir or output_dir == "outputs/checkpoints/ppo_so8t":
        # デフォルトの場合はwebdatasetを使用
        accessor = get_webdataset_accessor()
        output_dir = str(accessor.get_checkpoint_path("ppo_so8t"))

    # 設定
    config = SO8TPPOConfig(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=output_dir,
        max_steps=args.max_steps,
        experiment_id=experiment_id
    )

    # 学習実行
    train_so8t_ppo(config)

if __name__ == "__main__":
    main()

    reward_fn = create_reward_function(config)

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
    parser = argparse.ArgumentParser(description="SO8T PPO Training")
    parser.add_argument("--model_name", type=str, default="microsoft/Phi-3.5-mini-instruct",
                       help="Base model name")
    parser.add_argument("--dataset_path", type=str, default="data/so8t_balanced",
                       help="Dataset path")
    parser.add_argument("--output_dir", type=str, default="D:/webdataset/checkpoints/ppo_so8t",
                       help="Output directory")
    parser.add_argument("--max_steps", type=int, default=10000,
                       help="Maximum training steps")
    parser.add_argument("--experiment_id", type=str, default=None,
                       help="Experiment ID")

    args = parser.parse_args()

    # 実験ID生成
    experiment_id = args.experiment_id or f"so8t_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 出力ディレクトリ設定
    output_dir = args.output_dir
    if not output_dir or output_dir == "outputs/checkpoints/ppo_so8t":
        # デフォルトの場合はwebdatasetを使用
        accessor = get_webdataset_accessor()
        output_dir = str(accessor.get_checkpoint_path("ppo_so8t"))

    # 設定
    config = SO8TPPOConfig(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=output_dir,
        max_steps=args.max_steps,
        experiment_id=experiment_id
    )

    # 学習実行
    train_so8t_ppo(config)

if __name__ == "__main__":
    main()
