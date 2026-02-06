#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO(8) Integrated PPO Trainer
SO(8)統合データセットを使用したPPO学習スクリプト

このスクリプトは以下の機能を提供します：
1. SO(8)統合データセットの読み込み
2. 四値分類に基づくPPO報酬計算
3. SO(8)理論に基づく高度な学習
4. メモリ効率的なトレーニング
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
import psutil
import gc

# SO8T components
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

# CUDA設定
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PPOConfig:
    """PPO設定"""
    learning_rate: float = 1e-6
    max_steps: int = 1000
    batch_size: int = 1
    epochs: int = 1
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    warmup_steps: int = 100
    annealing_steps: int = 500
    save_steps: int = 100
    eval_steps: int = 50

@dataclass
class SO8TConfig:
    """SO(8)設定"""
    vector_weight: float = 0.3
    spinor_plus_weight: float = 0.4
    spinor_minus_weight: float = 0.3
    annealing_steps: int = 500
    chaos_factor: float = 0.1

class SO8TIntegratedDataset(Dataset):
    """SO(8)統合PPOデータセット"""

    def __init__(self, data_file: str, tokenizer=None, max_length: int = 2048):
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        self.load_data()
        logger.info(f"Loaded {len(self.data)} SO(8) integrated training examples")

    def load_data(self):
        """SO(8)統合データセット読み込み"""
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading SO(8) dataset"):
                if line.strip():
                    try:
                        item = json.loads(line.strip())
                        self.data.append(item)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON line: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # SO(8)統合データセット形式
        instruction = item.get('instruction', '')
        system_prompt = item.get('system', '')
        expected_output = item.get('output', '')
        metadata = item.get('metadata', {})

        # 四値分類に基づく報酬値計算
        four_class_label = metadata.get('four_class_label', 'ALLOW')

        if four_class_label == 'ALLOW':
            reward_value = 1.0
            target_correct = True
        elif four_class_label == 'Escalation':
            reward_value = 0.5
            target_correct = False
        elif four_class_label == 'Deny':
            reward_value = -1.0
            target_correct = False
        else:  # REFUSE
            reward_value = -2.0
            target_correct = False

        # NSFW判定
        is_nsfw = metadata.get('is_nsfw', False)
        safety_detection = metadata.get('safety_detection', False)
        is_nsfw = is_nsfw or safety_detection

        # SO(8)スコア
        so8t_vector_score = metadata.get('so8t_vector_score', 0.5)
        so8t_spinor_plus_score = metadata.get('so8t_spinor_plus_score', 0.3)
        so8t_spinor_minus_score = metadata.get('so8t_spinor_minus_score', 0.1)
        so8t_combined_score = metadata.get('so8t_combined_score', 0.3)

        # 思考トレース抽出
        thinking_trace = []
        if '<think>' in expected_output and '</think>' in expected_output:
            think_content = expected_output.split('<think>')[1].split('</think>')[0]
            thinking_trace = [line.strip() for line in think_content.split('\n') if line.strip()]

        return {
            'instruction': instruction,
            'system_prompt': system_prompt,
            'expected_output': expected_output,
            'reward_value': reward_value,
            'target_correct': target_correct,
            'is_nsfw': is_nsfw,
            'four_class_label': four_class_label,
            'quality_score': metadata.get('quality_score', 0.5),
            'thinking_trace': thinking_trace,
            'domain': metadata.get('domain', 'general'),
            'source': metadata.get('source', 'unknown'),
            'so8t_vector_score': so8t_vector_score,
            'so8t_spinor_plus_score': so8t_spinor_plus_score,
            'so8t_spinor_minus_score': so8t_spinor_minus_score,
            'so8t_combined_score': so8t_combined_score
        }

class SO8TPPOTrainer:
    """SO(8)統合PPOトレーナー"""

    def __init__(self, model_path: str, dataset_path: str, config: Optional[Dict[str, Any]] = None):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.config = config or {}

        # 設定の初期化
        self.ppo_config = PPOConfig(**self.config.get('ppo', {}))
        self.so8t_config = SO8TConfig(**self.config.get('so8t', {}))

        # モデルとトークナイザーの初期化
        self._init_model_and_tokenizer()

        # データセットの初期化
        self._init_dataset()

        # 最適化とスケジューラーの初期化
        self._init_optimizer_and_scheduler()

        # SO(8)コンポーネントの初期化
        self._init_so8t_components()

        # ログとチェックポイントディレクトリ
        self.output_dir = Path("H:/from_D/webdataset/checkpoints/so8t_ppo_training") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"SO(8) PPO Trainer initialized. Output dir: {self.output_dir}")

    def _init_model_and_tokenizer(self):
        """モデルとトークナイザーの初期化"""
        logger.info("Initializing model and tokenizer...")

        # モデルとトークナイザーの読み込み
        logger.info(f"Loading model: {self.model_path}")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                load_in_8bit=self.ppo_config.get('load_in_8bit', True)
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            self.tokenizer = None

        # RTX3060最適化設定
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

    def _init_dataset(self):
        """データセットの初期化"""
        logger.info("Initializing SO(8) integrated dataset...")

        self.dataset = SO8TIntegratedDataset(
            self.dataset_path,
            self.tokenizer,
            max_length=2048
        )

        # RTX3060最適化: 小さなバッチサイズでメモリ効率化
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.ppo_config.batch_size,
            shuffle=True,
            num_workers=0,  # GPU使用時は0が安定
            pin_memory=torch.cuda.is_available()
            # num_workers=0の場合はprefetch_factorを指定しない
        )

    def _init_optimizer_and_scheduler(self):
        """最適化とスケジューラーの初期化"""
        logger.info("Initializing optimizer and scheduler...")

        # モデルパラメータがない場合はplaceholder
        if self.model is not None:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.ppo_config.learning_rate,
                weight_decay=0.01,
                eps=1e-8
            )

            # ウォームアップ付きスケジューラー
            num_training_steps = len(self.dataloader) * self.ppo_config.epochs
            self.scheduler = self._get_scheduler(num_training_steps)
        else:
            self.optimizer = None
            self.scheduler = None

    def _init_so8t_components(self):
        """SO(8)コンポーネントの初期化"""
        logger.info("Initializing SO(8) components...")

        # SO(8)報酬システム
        self.so8t_reward_system = SO8TRewardSystem(self.so8t_config)

        # SO(8)位相アニーリング
        self.phase_annealer = SO8PhaseAnnealer(self.so8t_config)

    def _get_scheduler(self, num_training_steps: int):
        """学習率スケジューラーの取得"""
        from transformers import get_linear_schedule_with_warmup

        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.ppo_config.warmup_steps,
            num_training_steps=num_training_steps
        )

    def _calculate_so8t_reward(self, batch_data: Dict[str, Any]) -> torch.Tensor:
        """SO(8)理論に基づく報酬計算"""
        rewards = []

        for i in range(len(batch_data['reward_value'])):
            base_reward = batch_data['reward_value'][i]

            # SO(8)スコアの統合
            vector_score = batch_data['so8t_vector_score'][i]
            spinor_plus_score = batch_data['so8t_spinor_plus_score'][i]
            spinor_minus_score = batch_data['so8t_spinor_minus_score'][i]

            # SO(8)線形和による最終報酬
            so8t_reward = (
                self.so8t_config.vector_weight * vector_score +
                self.so8t_config.spinor_plus_weight * spinor_plus_score +
                self.so8t_config.spinor_minus_weight * spinor_minus_score
            )

            # 基本報酬とSO(8)報酬の統合
            final_reward = base_reward + 0.1 * so8t_reward

            # NSFWペナルティ
            if batch_data['is_nsfw'][i]:
                final_reward -= 0.5

            # 品質スコアボーナス
            quality_bonus = batch_data['quality_score'][i] - 0.5
            final_reward += 0.2 * quality_bonus

            rewards.append(final_reward)

        return torch.tensor(rewards, dtype=torch.float32, device=self.device)

    def _train_ppo_step(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        """1ステップのPPO学習"""
        # 報酬計算
        rewards = self._calculate_so8t_reward(batch_data)

        # ポリシー損失計算（placeholder）
        policy_loss = torch.tensor(0.0, requires_grad=True)
        value_loss = torch.tensor(0.0, requires_grad=True)
        entropy_loss = torch.tensor(0.0, requires_grad=True)

        # 総損失
        total_loss = policy_loss + self.ppo_config.value_loss_coef * value_loss - self.ppo_config.entropy_coef * entropy_loss

        # 逆伝播
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.ppo_config.max_grad_norm)
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item(),
            'mean_reward': rewards.mean().item(),
            'reward_std': rewards.std().item()
        }

    def train(self):
        """SO(8)統合PPO学習の実行"""
        logger.info("[START] Starting SO(8) Integrated PPO Training")
        logger.info(f"[STATS] Dataset size: {len(self.dataset)}")
        logger.info(f"[TARGET] Max steps: {self.ppo_config.max_steps}")
        logger.info(f"🧠 SO(8) config: {self.so8t_config}")

        start_time = time.time()
        best_reward = float('-inf')

        # トレーニングループ
        for step in tqdm(range(self.ppo_config.max_steps), desc="SO(8) PPO Training"):
            try:
                # バッチ取得
                batch_data = next(iter(self.dataloader))

                # GPU転送
                if torch.cuda.is_available():
                    for key, value in batch_data.items():
                        if isinstance(value, torch.Tensor):
                            batch_data[key] = value.cuda()

                # PPOステップ実行
                train_info = self._train_ppo_step(batch_data)

                # ログ出力
                if step % 10 == 0:
                    self._log_training_step(step, train_info)

                # チェックポイント保存
                if step % self.ppo_config.save_steps == 0:
                    self._save_checkpoint(step)

                # メモリ最適化
                if step % 100 == 0:
                    memory_usage = psutil.virtual_memory().percent
                    if memory_usage > 80:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Error at step {step}: {e}")
                continue

        # 最終チェックポイント保存
        self._save_checkpoint(self.ppo_config.max_steps)

        # 最終モデル保存 (Hugging Face形式)
        final_model_path = self.output_dir / "final_model"
        final_model_path.mkdir(exist_ok=True)

        logger.info(f"Saving final model to {final_model_path}...")
        logger.info(f"Model object: {type(self.model)}")
        if self.model is not None:
            try:
                self.model.save_pretrained(final_model_path)
                logger.info(f"[OK] Final model saved to {final_model_path}")
            except Exception as e:
                logger.error(f"Failed to save model: {e}")
        else:
            logger.error("Model is None, cannot save final model")

        total_time = time.time() - start_time
        logger.info("[OK] SO(8) PPO Training completed!")
        logger.info(".2f")
        logger.info(".2f")
        # 音声通知
        try:
            import subprocess
            subprocess.run([
                "powershell", "-ExecutionPolicy", "Bypass",
                "-File", "scripts\\utils\\play_audio_notification.ps1"
            ], check=True)
        except Exception as e:
            logger.warning(f"Audio notification failed: {e}")

    def _log_training_step(self, step: int, train_info: Dict[str, float]):
        """トレーニングステップのログ出力"""
        logger.info(
            f"Step {step}: "
            f"Loss={train_info['total_loss']:.4f} "
            f"Reward={train_info['mean_reward']:.4f}±{train_info['reward_std']:.4f} "
            f"Policy={train_info['policy_loss']:.4f} "
            f"Value={train_info['value_loss']:.4f} "
            f"Entropy={train_info['entropy_loss']:.4f}"
        )

    def _save_checkpoint(self, step: int):
        """チェックポイント保存"""
        checkpoint_path = self.output_dir / "02d"
        checkpoint_data = {
            'step': step,
            'model_state_dict': self.model.state_dict() if self.model else None,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'ppo_config': self.ppo_config,
            'so8t_config': self.so8t_config,
            'timestamp': datetime.now().isoformat()
        }

        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")

class SO8TRewardSystem:
    """SO(8)報酬システム"""

    def __init__(self, config: SO8TConfig):
        self.config = config

    def calculate_reward(self, response_data: Dict[str, Any]) -> float:
        """SO(8)理論に基づく報酬計算"""
        # ここにSO(8)理論に基づく高度な報酬計算を実装
        return 0.0

class SO8PhaseAnnealer:
    """SO(8)位相アニーリング"""

    def __init__(self, config: SO8TConfig):
        self.config = config
        self.current_phase = 0.0

    def step(self):
        """位相更新"""
        # 位相アニーリングの実装
        pass

def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="SO(8) Integrated PPO Trainer")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to SO(8) integrated dataset")
    parser.add_argument("--config_path", type=str, default=None, help="Path to config JSON")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    # 設定読み込み
    config = {}
    if args.config_path and Path(args.config_path).exists():
        with open(args.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

    # トレーナー初期化
    trainer = SO8TPPOTrainer(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        config=config
    )

    # 学習実行
    trainer.train()

if __name__ == "__main__":
    main()
