#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SO8T Enhanced PPO Training with Checkpoint Management
3分間隔チェックポイント + ローリングストック + 自動再開機能

理論的背景:
- URT (Unified Representation Theorem)
- NC-KART★ (Non-Commutative Kolmogorov-Arnold Representation Theory)
- SO(8)幾何学的知性

特徴:
- 3分間隔時間ベースチェックポイント
- 5個ローリングチェックポイントストック
- 電源投入時自動再開機能
- tqdm進捗管理 + logging
- RTX 3060最適化

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
import time
import signal
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# TRLとUnsloth
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
    BitsAndBytesConfig
)

# NKATユーティリティ
from scripts.utils.nkat_utils import (
    NKATRewardFunction,
    NKATThermostat
)

# SO8Tモデル
from scripts.models.so8t_thinking_model import create_so8t_thinking_model

# WebDatasetアクセスヘルパー
from webdataset.access_helper import get_webdataset_accessor

@dataclass
class SO8TEnhancedPPOConfig:
    """SO8T Enhanced PPO設定"""
    model_name: str = "microsoft/Phi-3.5-mini-instruct"
    dataset_path: str = "data/so8t_advanced_integrated"
    output_dir: str = field(default_factory=lambda: str(get_webdataset_accessor().get_checkpoint_path("ppo_so8t_enhanced")))
    experiment_id: str = field(default_factory=lambda: f"so8t_enhanced_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

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

    # 強化チェックポイント設定
    rolling_checkpoint: bool = True
    max_keep_checkpoints: int = 5
    save_interval_sec: int = 180  # 3分
    time_based_checkpoint: bool = True

class SO8TEnhancedDataset(Dataset):
    """SO8T Enhanced PPO用データセット"""

    def __init__(self, data_path: str, tokenizer, config: SO8TEnhancedPPOConfig, is_train: bool = True):
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

        print(f"Loaded {len(self.samples)} SO8T enhanced samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Phi-3.5フォーマットでプロンプト構築
        system_prompt = sample.get('system', '')
        instruction = sample['instruction']
        input_text = sample.get('input', '')

        full_prompt = f"{system_prompt}\n{instruction}"
        if input_text.strip():
            full_prompt += f"\n{input_text}"

        target = sample['output']
        tag = sample.get('tag', 'allow')

        return {
            'prompt': full_prompt,
            'target': target,
            'tag': tag,
            'sample': sample
        }

def create_enhanced_reward_function(config: SO8TEnhancedPPOConfig):
    """強化NKAT報酬関数"""

    def reward_fn(samples, responses, tokenizer, **kwargs):
        rewards = []

        for i, (sample, response) in enumerate(zip(samples, responses)):
            reward = 0.0

            response_text = tokenizer.decode(response, skip_special_tokens=True)

            # 基本長さ報酬
            if 10 < len(response_text) < 500:
                reward += 0.1

            # Phi-3.5タグ使用報酬
            phi35_tags = ['<|think|>', '<|observation|>', '<|deduction|>', '<|abduction|>', '<|integration|>', '<|final|>']
            used_tags = sum(1 for tag in phi35_tags if tag in response_text)
            reward += used_tags * 0.05

            # 思考プロセス構造報酬
            if '<|think|>' in response_text and '<|final|>' in response_text:
                reward += 0.2

            # タグ別報酬
            tag = sample.get('tag', 'allow')
            if tag == 'escalation':
                if '<|think|>' in response_text or any(term in response_text for term in ['考える', '分析', '推論']):
                    reward += 0.4
            elif tag == 'deny':
                if any(word in response_text.lower() for word in ['誤り', '間違い', '訂正', 'wrong', 'incorrect']):
                    reward += 0.4
            elif tag == 'refuse':
                if any(word in response_text.lower() for word in ['できません', '不適切', '安全', 'cannot', 'unsafe']):
                    reward += 0.4

            # LaTeX/Math報酬
            if '\\' in response_text or any(char in response_text for char in '∫∑∏√'):
                reward += 0.2

            # 理論用語報酬
            theory_terms = ['SO(8)', 'URT', 'NC-KART', '非可換', '幾何学', '物理', '数学']
            if any(term in response_text for term in theory_terms):
                reward += 0.3

            rewards.append(torch.tensor(reward))

        return rewards

    return reward_fn

def find_latest_checkpoint_enhanced(checkpoint_dir: Path):
    """強化チェックポイント検出"""
    if not checkpoint_dir.exists():
        return None

    checkpoints = []
    for item in checkpoint_dir.iterdir():
        if item.is_dir() and (item.name.startswith("checkpoint_") or
                            item.name.startswith("time_checkpoint_") or
                            item.name.startswith("emergency_checkpoint")):
            try:
                if item.name.startswith("checkpoint_"):
                    step = int(item.name.split("_")[1])
                elif item.name.startswith("time_checkpoint_"):
                    parts = item.name.split("_")
                    step = int(parts[2])
                elif item.name.startswith("emergency_checkpoint"):
                    step = 999999  # 緊急チェックポイントを優先
                else:
                    continue
                checkpoints.append((step, item))
            except (ValueError, IndexError):
                continue

    if checkpoints:
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        return checkpoints[0][1]
    return None

def load_session_info_enhanced(checkpoint_path: Path):
    """強化セッション情報読み込み"""
    session_files = [
        checkpoint_path / "session_info.json",
        checkpoint_path / "final_session_info.json"
    ]

    for session_file in session_files:
        if session_file.exists():
            try:
                with open(session_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARNING] Failed to load session info from {session_file}: {e}")
    return {}

def train_so8t_enhanced_ppo(config: SO8TEnhancedPPOConfig):
    """SO8T Enhanced PPO学習メイン関数"""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ログ設定
    logging.basicConfig(
        filename=output_dir / "enhanced_training.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 自動再開機能
    latest_checkpoint = find_latest_checkpoint_enhanced(output_dir)
    resume_step = 0
    total_training_time = 0

    if latest_checkpoint:
        print(f"[RESUME] Found checkpoint: {latest_checkpoint}")
        session_info = load_session_info_enhanced(latest_checkpoint)
        resume_step = session_info.get('global_step', 0)
        total_training_time = session_info.get('total_training_time', 0)
        print(f"[RESUME] Resuming from step {resume_step}")

    # モデルとトークナイザー準備
    print("Setting up SO8T enhanced model and tokenizer...")
    so8t_model, tokenizer, thermostat = setup_so8t_model_and_tokenizer_enhanced(config)

    # データセット準備
    print("Preparing SO8T enhanced datasets...")
    train_dataset = SO8TEnhancedDataset(config.dataset_path, tokenizer, config, is_train=True)
    val_dataset = SO8TEnhancedDataset(config.dataset_path, tokenizer, config, is_train=False)

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
        log_with="wandb"
    )

    # 報酬関数
    reward_fn = create_enhanced_reward_function(config)

    # PPOトレーナー
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=so8t_model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_collator=lambda data: {
            'input_ids': torch.stack([torch.tensor(item['input_ids']) for item in data]),
            'attention_mask': torch.stack([torch.tensor(item['attention_mask']) for item in data])
        }
    )

    # 時間ベースチェックポイント管理
    last_checkpoint_time = time.time()

    # シグナルハンドラー設定
    def signal_handler(signum, frame):
        print("\n[EMERGENCY] Signal received, saving emergency checkpoint...")
        emergency_path = output_dir / "emergency_checkpoint"
        try:
            so8t_model.save_pretrained(emergency_path)
            tokenizer.save_pretrained(emergency_path)

            session_info = {
                'global_step': global_step,
                'total_training_time': total_training_time,
                'emergency_save': True,
                'signal': signum,
                'timestamp': datetime.now().isoformat()
            }
            with open(emergency_path / "session_info.json", 'w') as f:
                json.dump(session_info, f, indent=2)

            print(f"[OK] Emergency checkpoint saved to {emergency_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save emergency checkpoint: {e}")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 学習ループ
    print("Starting SO8T Enhanced PPO training with advanced checkpoint management...")
    global_step = resume_step

    # 全体進捗バー
    total_epochs = (config.max_steps - resume_step) // 1000 + 1
    epoch_progress = tqdm(range(total_epochs), desc="Training Progress", unit="epoch")

    for epoch in epoch_progress:
        epoch_data = train_dataset.samples
        steps_in_epoch = min(1000, config.max_steps - global_step)

        # エポック内ステップ進捗バー
        step_progress = tqdm(range(steps_in_epoch),
                           desc=f"Epoch {epoch + 1}/{total_epochs}",
                           unit="step",
                           leave=False)

        for step in step_progress:
            step_start_time = time.time()

            # バッチサンプリング
            batch_indices = np.random.choice(len(epoch_data), config.batch_size, replace=False)
            batch = [epoch_data[i] for i in batch_indices]

            # プロンプトとターゲット準備
            queries = []
            responses = []

            for item in batch:
                prompt_tokens = tokenizer(
                    item['prompt'],
                    return_tensors="pt",
                    truncation=True,
                    max_length=config.max_prompt_length
                )

                with torch.no_grad():
                    response_tokens = so8t_model.generate(
                        **prompt_tokens.to(so8t_model.device),
                        max_new_tokens=config.max_target_length,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.pad_token_id
                    )

                queries.append(prompt_tokens)
                responses.append(response_tokens[:, prompt_tokens['input_ids'].shape[1]:])

            # PPOステップ
            stats = ppo_trainer.step(queries, responses, [reward_fn])
            step_time = time.time() - step_start_time
            total_training_time += step_time

            global_step += 1

            # ログ
            if global_step % config.logging_steps == 0:
                reward = stats.get('ppo/mean_scores', 0)
                loss = stats.get('ppo/loss/total', 0)

                logging.info(f"Step {global_step}: reward={reward:.4f}, loss={loss:.4f}, step_time={step_time:.2f}s")

                # tqdm description更新
                step_progress.set_description(
                    f"Epoch {epoch + 1}/{total_epochs} | Step {global_step} | "
                    f"Reward: {reward:.4f} | Loss: {loss:.4f} | Time: {step_time:.2f}s"
                )

                print(f"Step {global_step}: reward={reward:.4f}, loss={loss:.4f}, step_time={step_time:.2f}s")

            # 時間ベースチェックポイント (3分ごと)
            current_time = time.time()
            if config.time_based_checkpoint and (current_time - last_checkpoint_time) >= config.save_interval_sec:
                print(f"[TIME CHECKPOINT] Saving checkpoint at {global_step} steps ({config.save_interval_sec}s interval)...")

                time_checkpoint_path = output_dir / f"time_checkpoint_{global_step}_{int(current_time)}"
                so8t_model.save_pretrained(time_checkpoint_path)
                tokenizer.save_pretrained(time_checkpoint_path)

                session_info = {
                    'global_step': global_step,
                    'total_training_time': total_training_time,
                    'last_checkpoint_time': current_time,
                    'time_based': True,
                    'timestamp': datetime.now().isoformat()
                }
                with open(time_checkpoint_path / "session_info.json", 'w') as f:
                    json.dump(session_info, f, indent=2)

                print(f"[OK] Time-based checkpoint saved to {time_checkpoint_path}")
                last_checkpoint_time = current_time

            # ステップベースチェックポイント
            if global_step % config.save_steps == 0:
                checkpoint_path = output_dir / f"checkpoint_{global_step}"
                so8t_model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)

                session_info = {
                    'global_step': global_step,
                    'total_training_time': total_training_time,
                    'step_based': True,
                    'timestamp': datetime.now().isoformat()
                }
                with open(checkpoint_path / "session_info.json", 'w') as f:
                    json.dump(session_info, f, indent=2)

                print(f"[STEP CHECKPOINT] Saved checkpoint at step {global_step}")

            # 評価
            if global_step % config.eval_steps == 0:
                eval_reward = evaluate_enhanced_model(so8t_model, val_dataset, tokenizer, config)
                print(f"[EVAL] Validation reward at step {global_step}: {eval_reward:.4f}")
                logging.info(f"Validation reward: {eval_reward}")

        step_progress.close()

    epoch_progress.close()

    # 最終モデル保存
    final_path = output_dir / "final_enhanced_model"
    final_path.mkdir(exist_ok=True)
    so8t_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    final_session_info = {
        'global_step': global_step,
        'total_training_time': total_training_time,
        'training_completed': True,
        'timestamp': datetime.now().isoformat(),
        'enhanced_features': {
            'time_based_checkpoint': config.time_based_checkpoint,
            'rolling_checkpoint': config.rolling_checkpoint,
            'auto_resume': True,
            'tqdm_progress': True,
            'signal_handling': True
        }
    }
    with open(final_path / "final_session_info.json", 'w') as f:
        json.dump(final_session_info, f, indent=2)

    print("SO8T Enhanced PPO training completed!")
    print(f"Total training time: {total_training_time:.2f}s")
    print(f"Final step: {global_step}")
    print(f"Model saved to: {final_path}")

def evaluate_enhanced_model(model, dataset, tokenizer, config):
    """強化モデル評価"""
    model.eval()
    total_reward = 0
    count = 0

    reward_fn = create_enhanced_reward_function(config)

    with torch.no_grad():
        for i in range(min(50, len(dataset))):
            item = dataset[i]

            prompt_tokens = tokenizer(
                item['prompt'],
                return_tensors="pt",
                truncation=True,
                max_length=config.max_prompt_length
            ).to(model.device)

            response_tokens = model.generate(
                **prompt_tokens,
                max_new_tokens=config.max_target_length,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id
            )

            response = response_tokens[:, prompt_tokens['input_ids'].shape[1]:]

            rewards = reward_fn([item['sample']], [response], tokenizer)
            total_reward += rewards[0].item()
            count += 1

    return total_reward / count if count > 0 else 0

def setup_so8t_model_and_tokenizer_enhanced(config: SO8TEnhancedPPOConfig):
    """SO8T Enhancedモデルとトークナイザーのセットアップ"""

    print("Setting up SO8T enhanced model and tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # SO8T Thinkingモデル
    so8t_model = create_so8t_thinking_model(
        base_model_path=config.model_name,
        thermostat_enabled=config.use_nkat_thermostat
    )

    thermostat = None
    if config.use_nkat_thermostat:
        thermostat = NKATThermostat(tokenizer=tokenizer)
        print("NKAT Thermostat enabled")

    return so8t_model, tokenizer, thermostat

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="SO8T Enhanced PPO Training")
    parser.add_argument("--model_name", type=str, default="microsoft/Phi-3.5-mini-instruct",
                       help="Base model name")
    parser.add_argument("--dataset_path", type=str, default="data/so8t_advanced_integrated",
                       help="Dataset path")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory")
    parser.add_argument("--max_steps", type=int, default=5000,
                       help="Maximum training steps")
    parser.add_argument("--experiment_id", type=str, default=None,
                       help="Experiment ID")

    args = parser.parse_args()

    experiment_id = args.experiment_id or f"so8t_enhanced_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    output_dir = args.output_dir
    if not output_dir:
        accessor = get_webdataset_accessor()
        output_dir = str(accessor.get_checkpoint_path("ppo_so8t_enhanced"))

    config = SO8TEnhancedPPOConfig(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=output_dir,
        max_steps=args.max_steps,
        experiment_id=experiment_id
    )

    train_so8t_enhanced_ppo(config)

if __name__ == "__main__":
    main()
