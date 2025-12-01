#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO(8)T PPO Training Pipeline
SO(8)統合PPO学習パイプライン
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
import os
import time
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import math
from datetime import datetime
import logging
from tqdm import tqdm
import random

# 自作モジュールのインポート
from so8_residual_adapter import SO8ThinkingModel, create_so8_adapter_config

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PPODataset(Dataset):
    """PPO用データセット"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(data_path)

    def load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """データセット読み込み"""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    data.append(item)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # PPO形式に変換
        instruction = item.get('instruction', '')
        output = item.get('output', '')

        # プロンプト構築
        prompt = f"Instruction: {instruction}\n\nResponse:"

        return {
            'prompt': prompt,
            'reference_output': output,
            'metadata': item.get('metadata', {})
        }

class PPOActorCritic(nn.Module):
    """PPO Actor-Criticモデル"""

    def __init__(self, base_model, vocab_size: int):
        super().__init__()
        self.base_model = base_model
        self.vocab_size = vocab_size

        # Actor: ポリシー出力
        self.actor_head = nn.Linear(base_model.config.hidden_size, vocab_size)

        # Critic: 価値関数
        self.critic_head = nn.Linear(base_model.config.hidden_size, 1)

        # SO(8)温度制御
        self.entropy_temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # ベースモデル出力
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )

        # 最後の隠れ状態を取得
        last_hidden = outputs.hidden_states[-1]  # [batch, seq, hidden]

        # Actor: ポリシー分布
        logits = self.actor_head(last_hidden)  # [batch, seq, vocab]

        # Critic: 状態価値
        values = self.critic_head(last_hidden).squeeze(-1)  # [batch, seq]

        return {
            'logits': logits,
            'values': values,
            'hidden_states': outputs.hidden_states,
            'entropy_temp': self.entropy_temperature
        }

class PPORewardModel(nn.Module):
    """PPO報酬モデル"""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

        # 報酬予測ヘッド
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1)

        # バイナリ分類ヘッド（好ましい/好ましくない）
        self.preference_head = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )

        last_hidden = outputs.hidden_states[-1]

        # 報酬スコア
        rewards = self.reward_head(last_hidden).squeeze(-1)

        # 好ましさスコア
        preferences = torch.sigmoid(self.preference_head(last_hidden).squeeze(-1))

        return {
            'rewards': rewards,
            'preferences': preferences
        }

class PPOExperienceBuffer:
    """PPO経験バッファ"""

    def __init__(self, buffer_size: int = 10000):
        self.buffer_size = buffer_size
        self.buffer = []

    def add_experience(self, experience: Dict[str, Any]):
        """経験を追加"""
        self.buffer.append(experience)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def sample_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """バッチサンプリング"""
        if len(self.buffer) < batch_size:
            return self.buffer.copy()

        return random.sample(self.buffer, batch_size)

    def clear(self):
        """バッファクリア"""
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

class SO8TPPOTrainer:
    """SO(8)T PPOトレーナー"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # SFT済みモデルをロード
        self.load_sft_model()

        # PPO固有のモデル作成
        self.setup_ppo_models()

        # データセット準備
        self.setup_datasets()

        # 経験バッファ
        self.experience_buffer = PPOExperienceBuffer(
            buffer_size=config.get('buffer_size', 10000)
        )

        # チェックポイント管理
        self.checkpoint_manager = SO8TPPOCheckpointManager(
            save_dir=config.get('output_dir', './checkpoints/ppo'),
            save_interval=config.get('checkpoint_interval', 180)
        )

        # PPOハイパーパラメータ
        self.ppo_config = {
            'clip_ratio': config.get('clip_ratio', 0.2),
            'value_coeff': config.get('value_coeff', 0.5),
            'entropy_coeff': config.get('entropy_coeff', 0.01),
            'max_grad_norm': config.get('max_grad_norm', 0.5),
            'ppo_epochs': config.get('ppo_epochs', 4),
            'batch_size': config.get('ppo_batch_size', 64),
            'learning_rate': config.get('ppo_lr', 1e-6)
        }

    def load_sft_model(self):
        """SFT済みモデルをロード"""
        sft_model_path = self.config.get('sft_model_path', './checkpoints/sft_so8t/final_model')

        logger.info(f"SFTモデル読み込み: {sft_model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # SO(8)Tモデルとしてロード
        base_model = AutoModelForCausalLM.from_pretrained(
            sft_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        so8_config = create_so8_adapter_config(base_model.config.hidden_size)
        self.sft_model = SO8ThinkingModel(base_model, so8_config)
        self.sft_model.eval()

    def setup_ppo_models(self):
        """PPO固有のモデルセットアップ"""
        logger.info("PPOモデルセットアップ開始")

        # Actor-Criticモデル
        self.actor_critic = PPOActorCritic(self.sft_model, self.tokenizer.vocab_size)

        # 報酬モデル（別途学習）
        reward_base = AutoModelForCausalLM.from_pretrained(
            self.config.get('sft_model_path', './checkpoints/sft_so8t/final_model'),
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.reward_model = PPORewardModel(reward_base)

        # オプティマイザー
        self.actor_optimizer = torch.optim.AdamW(
            self.actor_critic.parameters(),
            lr=self.ppo_config['learning_rate']
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.actor_critic.parameters(),
            lr=self.ppo_config['learning_rate']
        )
        self.reward_optimizer = torch.optim.AdamW(
            self.reward_model.parameters(),
            lr=self.ppo_config['learning_rate']
        )

        # デバイス移動
        self.actor_critic = self.actor_critic.to(self.device)
        self.reward_model = self.reward_model.to(self.device)

        logger.info("PPOモデルセットアップ完了")

    def setup_datasets(self):
        """データセット準備"""
        ppo_dataset_path = self.config.get('ppo_dataset', 'data/train_ppo.jsonl')

        self.ppo_dataset = PPODataset(ppo_dataset_path, self.tokenizer,
                                    max_length=self.config.get('max_length', 2048))

        logger.info(f"PPOデータセット数: {len(self.ppo_dataset)}")

    def generate_rollout(self, prompt: str, max_length: int = 512) -> Dict[str, Any]:
        """ロールアウト生成（軌跡生成）"""
        # プロンプトをトークナイズ
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        # 初期状態
        generated_tokens = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # ステップごとのデータを保存
        log_probs = []
        values = []
        rewards = []
        entropies = []

        for step in range(max_length):
            # Actor-Criticでアクションと価値を計算
            with torch.no_grad():
                ac_outputs = self.actor_critic(
                    input_ids=generated_tokens,
                    attention_mask=attention_mask
                )

            # 最後のトークンのみ使用
            last_logits = ac_outputs['logits'][:, -1, :]  # [batch, vocab]
            last_values = ac_outputs['values'][:, -1]     # [batch]

            # 確率分布
            probs = F.softmax(last_logits / ac_outputs['entropy_temp'], dim=-1)

            # アクションサンプリング
            action = torch.multinomial(probs, 1)  # [batch, 1]

            # ログ確率
            log_prob = torch.log(probs.gather(-1, action).squeeze())  # [batch]

            # エントロピー
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

            # 報酬計算（報酬モデル使用）
            reward_inputs = torch.cat([generated_tokens, action], dim=-1)
            reward_mask = torch.cat([attention_mask,
                                   torch.ones_like(action)], dim=-1)

            with torch.no_grad():
                reward_outputs = self.reward_model(
                    input_ids=reward_inputs,
                    attention_mask=reward_mask
                )
                reward = reward_outputs['rewards'][:, -1].mean()

            # データ保存
            log_probs.append(log_prob.cpu())
            values.append(last_values.cpu())
            rewards.append(reward.cpu())
            entropies.append(entropy.cpu())

            # 次のトークンを追加
            generated_tokens = torch.cat([generated_tokens, action], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(action)], dim=-1)

            # EOSトークンで終了
            if action.item() == self.tokenizer.eos_token_id:
                break

        return {
            'tokens': generated_tokens.cpu(),
            'log_probs': torch.stack(log_probs),
            'values': torch.stack(values),
            'rewards': torch.stack(rewards),
            'entropies': torch.stack(entropies)
        }

    def compute_ppo_loss(self, rollout_data: Dict[str, Any],
                        old_log_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """PPO損失計算"""
        tokens = rollout_data['tokens'].to(self.device)
        attention_mask = torch.ones_like(tokens).to(self.device)

        # 新しいポリシーと価値
        ac_outputs = self.actor_critic(
            input_ids=tokens[:, :-1],  # 最後のトークンを除く
            attention_mask=attention_mask[:, :-1]
        )

        new_logits = ac_outputs['logits']
        new_values = ac_outputs['values']

        # 確率分布
        new_probs = F.softmax(new_logits, dim=-1)

        # 新しいログ確率
        actions = tokens[:, 1:]  # ターゲットアクション
        new_log_probs = torch.log(new_probs.gather(-1, actions.unsqueeze(-1)).squeeze() + 1e-10)

        # 重要性サンプリング比率
        ratios = torch.exp(new_log_probs - old_log_probs.to(self.device))

        # クリッピング
        clipped_ratios = torch.clamp(ratios, 1 - self.ppo_config['clip_ratio'],
                                   1 + self.ppo_config['clip_ratio'])

        # 報酬の割引
        rewards = rollout_data['rewards'].to(self.device)
        gamma = self.config.get('gamma', 0.99)
        gae_lambda = self.config.get('gae_lambda', 0.95)

        # GAE (Generalized Advantage Estimation)
        advantages = self.compute_gae(rewards, new_values, gamma, gae_lambda)

        # ポリシー損失
        policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()

        # 価値損失
        value_loss = F.mse_loss(new_values, rewards)

        # エントロピー損失
        entropies = rollout_data['entropies'].to(self.device)
        entropy_loss = -entropies.mean()

        # 総損失
        total_loss = (
            policy_loss +
            self.ppo_config['value_coeff'] * value_loss +
            self.ppo_config['entropy_coeff'] * entropy_loss
        )

        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'advantages': advantages.mean(),
            'ratios': ratios.mean()
        }

    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor,
                   gamma: float, gae_lambda: float) -> torch.Tensor:
        """GAE計算"""
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * last_gae

        return advantages

    def train_ppo_step(self, batch_rollouts: List[Dict[str, Any]]) -> Dict[str, float]:
        """1ステップのPPOトレーニング"""
        total_losses = []

        for rollout in batch_rollouts:
            # 古いログ確率を保存
            old_log_probs = rollout['log_probs']

            # PPO損失計算
            losses = self.compute_ppo_loss(rollout, old_log_probs)

            # アクター更新
            self.actor_optimizer.zero_grad()
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.ppo_config['max_grad_norm'])
            self.actor_optimizer.step()

            total_losses.append(losses['total_loss'].item())

        return {
            'ppo_loss': np.mean(total_losses)
        }

    def train_reward_model(self, batch_data: List[Dict[str, Any]]):
        """報酬モデルトレーニング"""
        # 簡易実装：実際には好ましい/好ましくないのペアデータが必要
        pass

    def train(self):
        """PPOトレーニング実行"""
        logger.info("SO(8)T PPOトレーニング開始")

        num_epochs = self.config.get('ppo_epochs', 10)
        batch_size = self.config.get('ppo_batch_size', 4)

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            epoch_losses = []

            # データセットからバッチ処理
            dataloader = DataLoader(self.ppo_dataset, batch_size=batch_size,
                                  shuffle=True, drop_last=True)

            for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
                batch_rollouts = []

                # 各バッチアイテムに対してロールアウト生成
                for item in batch:
                    prompt = item['prompt']
                    rollout = self.generate_rollout(prompt)
                    batch_rollouts.append(rollout)

                # PPO更新
                losses = self.train_ppo_step(batch_rollouts)
                epoch_losses.append(losses['ppo_loss'])

                # チェックポイント保存
                self.checkpoint_manager.save_checkpoint(
                    self.actor_critic, self.reward_model, epoch
                )

            avg_loss = np.mean(epoch_losses)
            logger.info(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

        # 最終モデル保存
        final_path = Path(self.config.get('output_dir', './checkpoints/ppo')) / "final_model"
        final_path.mkdir(exist_ok=True)

        # Actor-Critic保存
        torch.save(self.actor_critic.state_dict(), final_path / "actor_critic.pt")
        torch.save(self.reward_model.state_dict(), final_path / "reward_model.pt")
        self.tokenizer.save_pretrained(final_path)

        logger.info(f"PPOトレーニング完了。最終モデル保存: {final_path}")

        return final_path

class SO8TPPOCheckpointManager:
    """SO(8)T PPOチェックポイント管理"""

    def __init__(self, save_dir: str, save_interval: int = 180):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.save_interval = save_interval
        self.last_save_time = time.time()
        self.checkpoints = []
        self.max_checkpoints = 5

    def save_checkpoint(self, actor_critic, reward_model, step):
        """チェックポイント保存"""
        current_time = time.time()

        if current_time - self.last_save_time >= self.save_interval:
            checkpoint_path = self.save_dir / f"checkpoint_step_{step}"
            checkpoint_path.mkdir(exist_ok=True)

            # モデル保存
            torch.save(actor_critic.state_dict(), checkpoint_path / "actor_critic.pt")
            torch.save(reward_model.state_dict(), checkpoint_path / "reward_model.pt")

            # メタデータ保存
            metadata = {
                'step': step,
                'timestamp': datetime.now().isoformat(),
                'save_time': current_time
            }

            with open(checkpoint_path / "metadata.json", 'w') as f:
                json.dump(metadata, f)

            # ローリングストック管理
            self.checkpoints.append(checkpoint_path)
            if len(self.checkpoints) > self.max_checkpoints:
                old_checkpoint = self.checkpoints.pop(0)
                if old_checkpoint.exists():
                    shutil.rmtree(old_checkpoint)

            self.last_save_time = current_time
            logger.info(f"PPOチェックポイント保存: {checkpoint_path}")

def create_ppo_config() -> Dict[str, Any]:
    """PPO設定を作成"""
    return {
        'sft_model_path': './checkpoints/sft_so8t/final_model',
        'ppo_dataset': 'data/train_ppo.jsonl',
        'output_dir': './checkpoints/ppo_so8t',
        'ppo_epochs': 10,
        'ppo_batch_size': 4,
        'learning_rate': 1e-6,
        'clip_ratio': 0.2,
        'value_coeff': 0.5,
        'entropy_coeff': 0.01,
        'max_grad_norm': 0.5,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'buffer_size': 10000,
        'max_length': 2048,
        'checkpoint_interval': 180  # 3分
    }

def main():
    """メイン関数"""
    print("🚀 SO(8)T PPO Training Pipeline")
    print("=" * 50)

    # 設定
    config = create_ppo_config()

    # PPOトレーナー作成
    trainer = SO8TPPOTrainer(config)

    # トレーニング実行
    final_path = trainer.train()

    print("
✅ PPOトレーニング完了!"    print(f"📊 トレーニング済みモデル: {final_path}")

    # 音声通知
    try:
        import subprocess
        subprocess.run([
            "powershell", "-ExecutionPolicy", "Bypass",
            "-File", "scripts\\utils\\play_audio_notification.ps1"
        ], check=True)
    except Exception as e:
        print(f"[WARNING] 音声通知失敗: {e}")

if __name__ == "__main__":
    main()

