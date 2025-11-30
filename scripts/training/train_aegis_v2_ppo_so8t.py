#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS-v2.0 PPO Training with SO(8) Rotation Adapter
圏論的同型性とカオス強化による高度なPPO学習
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import time
import hashlib
from datetime import datetime
import math

# Import SO(8) components
from models.Borea_Phi_3_5_mini_Instruct_Jp.so8_rotation_adapter import (
    SO8PhaseTransitionAnnealer,
    ChaosInducedDiversityEnhancer,
    PPOAlignmentRewardSystem
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/aegis_v2_ppo_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.get_logger(__name__)

@dataclass
class PPOConfig:
    """PPO設定"""
    learning_rate: float = 1e-6
    max_grad_norm: float = 0.1
    batch_size: int = 4
    mini_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    epochs: int = 10
    max_steps: int = 10000
    warmup_steps: int = 100

    # PPO specific
    cliprange: float = 0.2
    vf_coef: float = 0.1
    ent_coef: float = 0.01
    gamma: float = 0.99
    lam: float = 0.95
    max_kl: float = 0.01

    # SO(8) specific
    alpha_initial: float = -2.1336307753809063
    alpha_target: float = math.log(0.382)  # Φ^(-2)
    annealing_steps: int = 1000
    chaos_intensity: float = 0.1

    # Checkpoint
    checkpoint_interval: int = 180  # 3分毎
    max_checkpoints: int = 5
    checkpoint_dir: str = "D:/webdataset/checkpoints/aegis_v2_ppo"

    # Reward system
    isomorphism_reward_weight: float = 5.0
    hacking_penalty_weight: float = -2.0
    nsfw_refusal_reward_weight: float = 2.0

class AEGISV2Dataset(Dataset):
    """AEGIS-v2.0学習データセット"""

    def __init__(self, data_file: str, tokenizer=None, max_length: int = 2048):
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        self.load_data()

    def load_data(self):
        """データ読み込み"""
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    self.data.append(item)

        logger.info(f"Loaded {len(self.data)} training examples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # テキストの準備
        text = item.get('text', '')
        if not text:
            return None

        # ラベルの準備
        ppo_labels = item.get('ppo_labels', {})
        target_correct = ppo_labels.get('reward_correctness', 0) > 0.5
        is_nsfw = item.get('category', '') == 'nsfw'

        return {
            'text': text,
            'target_correct': target_correct,
            'is_nsfw': is_nsfw,
            'quality_score': item.get('quality_score', 0.5),
            'thinking_trace': item.get('thinking_trace', [])
        }

class PPOTrainer:
    """AEGIS-v2.0 PPOトレーナー"""

    def __init__(self, config_path: str, model_path: str):
        self.config_path = config_path
        self.model_path = model_path

        # 設定読み込み
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # PPO設定
        self.ppo_config = PPOConfig()

        # SO(8)コンポーネント初期化
        self.phase_annealer = SO8PhaseTransitionAnnealer(
            initial_alpha=self.ppo_config.alpha_initial,
            target_alpha=self.ppo_config.alpha_target,
            annealing_steps=self.ppo_config.annealing_steps
        )

        self.chaos_enhancer = ChaosInducedDiversityEnhancer(
            hidden_size=3072,  # Phi-3.5 hidden size
            chaos_intensity=self.ppo_config.chaos_intensity
        )

        self.reward_system = PPOAlignmentRewardSystem(hidden_size=3072)

        # モデルとトークナイザーの準備
        self.model = None
        self.tokenizer = None
        self.ref_model = None  # 参照モデル

        self.setup_model_and_tokenizer()

        # データセット準備
        self.train_dataset = AEGISV2Dataset(
            self.config['data']['train_file'],
            self.tokenizer,
            self.config['data']['max_length']
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.ppo_config.batch_size,
            shuffle=True,
            num_workers=4
        )

        # オプティマイザー
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.ppo_config.learning_rate
        )

        # 学習状態
        self.global_step = 0
        self.epoch = 0
        self.best_reward = float('-inf')

        # チェックポイント管理
        self.checkpoint_dir = Path(self.ppo_config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 統計
        self.stats = {
            'rewards': [],
            'losses': [],
            'kl_divs': [],
            'isomorphism_scores': [],
            'chaos_diversity': []
        }

        logger.info("PPO Trainer initialized with SO(8) enhancements")

    def setup_model_and_tokenizer(self):
        """モデルとトークナイザーのセットアップ"""
        from transformers import AutoTokenizer, AutoModelForCausalLM

        logger.info(f"Loading model: {self.model_path}")

        # モデル読み込み
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        # 参照モデル（クローン）
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.ref_model.eval()

        # トークナイザー
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Model and tokenizer loaded successfully")

    def compute_rewards(self, batch: Dict[str, Any]) -> torch.Tensor:
        """圏論的同型性に基づく報酬計算"""
        rewards = []

        for i in range(len(batch['input_ids'])):
            # モデル推論
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch['input_ids'][i:i+1],
                    attention_mask=batch['attention_mask'][i:i+1],
                    output_hidden_states=True
                )

            # 最終隠れ状態を取得
            hidden_states = outputs.hidden_states[-1]  # [batch, seq, hidden]

            # 報酬計算
            reward = self.reward_system.compute_alignment_reward(
                hidden_states,
                target_correct=batch['target_correct'][i],
                is_nsfw=batch['is_nsfw'][i]
            )

            rewards.append(reward)

        return torch.stack(rewards)

    def compute_ppo_loss(self, old_logprobs: torch.Tensor, new_logprobs: torch.Tensor,
                        advantages: torch.Tensor, cliprange: float) -> Tuple[torch.Tensor, Dict]:
        """PPO損失計算"""
        # 確率比
        ratio = torch.exp(new_logprobs - old_logprobs)

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()

        # 価値関数損失（簡易版）
        vf_loss = 0.0  # ここでは実装省略

        # エントロピー損失（簡易版）
        entropy_loss = 0.0

        total_loss = (
            policy_loss +
            self.ppo_config.vf_coef * vf_loss -
            self.ppo_config.ent_coef * entropy_loss
        )

        loss_info = {
            'policy_loss': policy_loss.item(),
            'vf_loss': vf_loss,
            'entropy_loss': entropy_loss,
            'total_loss': total_loss.item()
        }

        return total_loss, loss_info

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """1ステップの学習"""
        # 参照モデルのログ確率を計算
        with torch.no_grad():
            ref_outputs = self.ref_model(**batch)
            ref_logprobs = self.get_logprobs_from_outputs(ref_outputs, batch)

        # 現在のモデルのログ確率を計算
        current_outputs = self.model(**batch)
        current_logprobs = self.get_logprobs_from_outputs(current_outputs, batch)

        # 報酬計算
        rewards = self.compute_rewards(batch)

        # 利得計算（簡易版）
        advantages = rewards - rewards.mean()

        # PPO損失計算
        loss, loss_info = self.compute_ppo_loss(
            ref_logprobs, current_logprobs, advantages, self.ppo_config.cliprange
        )

        # 逆伝播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.ppo_config.max_grad_norm)
        self.optimizer.step()

        # SO(8)位相アニーリング
        current_alpha = self.phase_annealer.get_current_alpha()

        # カオス多様性強化
        chaos_signal = self.chaos_enhancer.apply_chaos_diversity(current_outputs.hidden_states[-1])

        return {
            **loss_info,
            'rewards': rewards.mean().item(),
            'alpha': current_alpha,
            'chaos_intensity': self.chaos_enhancer.chaos_intensity
        }

    def get_logprobs_from_outputs(self, outputs, batch):
        """出力からログ確率を計算"""
        logits = outputs.logits
        labels = batch['input_ids']

        # シフトして次のトークンの予測確率を取得
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # ログ確率計算
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        neg_logprobs = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        return -neg_logprobs.view(shift_labels.shape)

    def save_checkpoint(self, step: int):
        """チェックポイント保存"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"

        checkpoint = {
            'step': step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'ppo_config': self.ppo_config.__dict__,
            'phase_annealer': {
                'current_step': self.phase_annealer.current_step,
                'alpha_schedule': self.phase_annealer.alpha_schedule.tolist()
            },
            'stats': self.stats,
            'best_reward': self.best_reward
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # 古いチェックポイント削除
        self.cleanup_old_checkpoints()

    def cleanup_old_checkpoints(self):
        """古いチェックポイントを削除"""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_step_*.pt"),
                           key=lambda x: x.stat().st_mtime)

        if len(checkpoints) > self.ppo_config.max_checkpoints:
            for old_ckpt in checkpoints[:-self.ppo_config.max_checkpoints]:
                old_ckpt.unlink()
                logger.info(f"Removed old checkpoint: {old_ckpt}")

    def load_checkpoint(self, checkpoint_path: str):
        """チェックポイント読み込み"""
        checkpoint = torch.load(checkpoint_path)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.stats = checkpoint.get('stats', self.stats)
        self.best_reward = checkpoint.get('best_reward', self.best_reward)

        # 位相アニーリング状態復元
        if 'phase_annealer' in checkpoint:
            self.phase_annealer.current_step = checkpoint['phase_annealer']['current_step']
            self.phase_annealer.alpha_schedule = torch.tensor(
                checkpoint['phase_annealer']['alpha_schedule']
            )

        logger.info(f"Checkpoint loaded: {checkpoint_path}")

    def train(self):
        """メイン学習ループ"""
        logger.info("Starting AEGIS-v2.0 PPO training with SO(8) enhancements")

        start_time = time.time()
        last_checkpoint_time = start_time

        try:
            for epoch in range(self.ppo_config.epochs):
                self.epoch = epoch
                epoch_losses = []
                epoch_rewards = []

                progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")

                for batch in progress_bar:
                    # バッチがNoneの場合はスキップ
                    if batch is None or any(x is None for x in batch.values()):
                        continue

                    # バッチをデバイスに移動
                    batch = {k: v.to(self.model.device) if torch.is_tensor(v) else v
                           for k, v in batch.items()}

                    # 学習ステップ
                    step_info = self.train_step(batch)

                    epoch_losses.append(step_info['total_loss'])
                    epoch_rewards.append(step_info['rewards'])

                    self.global_step += 1

                    # 統計更新
                    self.stats['losses'].append(step_info['total_loss'])
                    self.stats['rewards'].append(step_info['rewards'])
                    self.stats['isomorphism_scores'].append(step_info.get('alpha', 0))

                    # プログレスバー更新
                    progress_bar.set_postfix({
                        'loss': f"{step_info['total_loss']:.4f}",
                        'reward': f"{step_info['rewards']:.4f}",
                        'alpha': f"{step_info.get('alpha', 0):.4f}"
                    })

                    # チェックポイント保存（3分毎）
                    current_time = time.time()
                    if current_time - last_checkpoint_time >= self.ppo_config.checkpoint_interval:
                        self.save_checkpoint(self.global_step)
                        last_checkpoint_time = current_time

                        # 統計レポート
                        self.log_training_stats()

                    # 最大ステップチェック
                    if self.global_step >= self.ppo_config.max_steps:
                        break

                # エポック完了
                avg_loss = np.mean(epoch_losses)
                avg_reward = np.mean(epoch_rewards)

                logger.info(f"Epoch {epoch} completed: Loss={avg_loss:.4f}, Reward={avg_reward:.4f}")

                if self.global_step >= self.ppo_config.max_steps:
                    break

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # 最終チェックポイント保存
            self.save_checkpoint(self.global_step)
            self.log_final_stats()

            # 音声通知
            try:
                import winsound
                winsound.Beep(1000, 1000)  # 完了音
            except ImportError:
                pass

    def log_training_stats(self):
        """学習統計のログ出力"""
        recent_rewards = self.stats['rewards'][-100:]
        recent_losses = self.stats['losses'][-100:]

        logger.info(f"Step {self.global_step}: "
                   f"Avg Reward: {np.mean(recent_rewards):.4f}, "
                   f"Avg Loss: {np.mean(recent_losses):.4f}, "
                   f"Alpha: {self.phase_annealer.get_current_alpha():.4f}")

    def log_final_stats(self):
        """最終統計のログ出力"""
        logger.info("=== Final Training Statistics ===")
        logger.info(f"Total steps: {self.global_step}")
        logger.info(f"Best reward: {self.best_reward:.4f}")
        logger.info(f"Final alpha: {self.phase_annealer.get_current_alpha():.4f}")
        logger.info(f"Checkpoints saved: {len(list(self.checkpoint_dir.glob('*.pt')))}")

def main():
    """メイン実行関数"""
    print("AEGIS-v2.0 PPO Training with SO(8) Rotation Adapter")
    print("=" * 60)

    # 設定ファイル
    config_path = "aegis_v2_test_config.json"
    model_path = "models/Borea-Phi-3.5-mini-Instruct-Jp"

    # PPOトレーナー初期化
    trainer = PPOTrainer(config_path, model_path)

    # 既存のチェックポイントがある場合は読み込み
    checkpoint_files = list(trainer.checkpoint_dir.glob("checkpoint_step_*.pt"))
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        print(f"Loading checkpoint: {latest_checkpoint}")
        trainer.load_checkpoint(str(latest_checkpoint))

    # 学習開始
    try:
        trainer.train()
        print("\n🎉 AEGIS-v2.0 PPO training completed successfully!")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
