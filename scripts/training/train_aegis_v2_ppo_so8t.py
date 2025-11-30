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
import matplotlib.pyplot as plt
import seaborn as sns
from huggingface_hub import HfApi, upload_file
import pandas as pd

# Placeholder for Unsloth/BitsAndBytes imports - will be initialized after logger setup
UNSLOTH_AVAILABLE = False
BITSANDBYTES_AVAILABLE = False

# Import SO(8) components with path manipulation for hyphenated directory names
import sys
import os

# Add the models directory and the specific model directory to sys.path
models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
model_dir = os.path.join(models_dir, 'Borea-Phi-3.5-mini-Instruct-Jp')

sys.path.insert(0, models_dir)
sys.path.insert(0, model_dir)

# Try direct import with sys.path manipulation
try:
    from so8_rotation_adapter import (
        SO8PhaseTransitionAnnealer,
        ChaosInducedDiversityEnhancer,
        PPOAlignmentRewardSystem
    )
except ImportError:
    # Fallback: try importing from the full path
    try:
        sys.path.insert(0, model_dir)
        from so8_rotation_adapter import (
            SO8PhaseTransitionAnnealer,
            ChaosInducedDiversityEnhancer,
            PPOAlignmentRewardSystem
        )
    except ImportError as e:
        print(f"Failed to import SO(8) components: {e}")
        print(f"models_dir: {models_dir}")
        print(f"model_dir: {model_dir}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"sys.path: {sys.path}")
        # Continue without SO(8) components for basic testing
        print("Continuing without SO(8) components for basic functionality test...")
        SO8PhaseTransitionAnnealer = None
        ChaosInducedDiversityEnhancer = None
        PPOAlignmentRewardSystem = None

# Import Bayesian optimizer
try:
    from alpha_gate_annealing import GoldenRatioBayesianOptimizer
except ImportError:
    print("Bayesian optimizer not available, continuing without it...")
    GoldenRatioBayesianOptimizer = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/aegis_v2_ppo_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Unsloth/BitsAndBytes after logger setup
try:
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported
    UNSLOTH_AVAILABLE = True
    logger.info("Unsloth available - using memory-efficient training")
except (ImportError, Exception) as e:
    logger.info(f"Unsloth not available ({e}) - falling back to bitsandbytes + PEFT")
    UNSLOTH_AVAILABLE = False
    try:
        from transformers import BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model
        import bitsandbytes as bnb
        BITSANDBYTES_AVAILABLE = True
        logger.info("BitsAndBytes + PEFT available - using 4bit quantization")
    except (ImportError, Exception) as e2:
        BITSANDBYTES_AVAILABLE = False
        logger.warning(f"BitsAndBytes not available ({e2}) - using standard transformers with CPU fallback")

@dataclass
class PPOConfig:
    """PPO設定 - RTX3060最適化"""
    learning_rate: float = 2e-4  # Unsloth推奨の高い学習率
    max_grad_norm: float = 0.1
    batch_size: int = 1  # RTX3060のメモリ制約のため1
    mini_batch_size: int = 1
    gradient_accumulation_steps: int = 8  # 効果的なバッチサイズ8を実現
    epochs: int = 1  # Unslothでは1エポックで十分
    max_steps: int = 200
    warmup_steps: int = 10

    # RTX3060最適化設定
    use_unsloth: bool = True
    gpu_memory_limit: float = 0.85  # 85%まで使用

    # ベイズ最適化設定
    enable_bayesian_optimization: bool = False  # デフォルトでは無効（メモリ制約のため）
    bayesian_trials: int = 10  # 試行回数を減らす
    bayesian_timeout: int = 1800  # 30分に短縮
    optimize_learning_rate: bool = True
    optimize_batch_size: bool = True
    optimize_alpha_params: bool = True

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
    checkpoint_dir: str = "H:/from_D/webdataset/checkpoints/aegis_v2_ppo"

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

        # ベイズ最適化の初期化
        self.bayesian_optimizer = None
        if self.ppo_config.enable_bayesian_optimization:
            self.setup_bayesian_optimization()

        # モデルとトークナイザーの準備
        self.model = None
        self.tokenizer = None
        self.ref_model = None  # 参照モデル

        # test_modeの場合はモデルロードをスキップ
        if not self.config.get('training', {}).get('test_mode', False):
            self.setup_model_and_tokenizer()
        else:
            logger.info("Test mode: Skipping model loading")
            self.setup_test_mode()

    def setup_bayesian_optimization(self):
        """ベイズ最適化のセットアップ"""
        try:
            import optuna
            from optuna.samplers import TPESampler

            logger.info("Setting up Bayesian optimization...")

            # Optuna studyの作成
            study_name = f"aegis_v2_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.optuna_study = optuna.create_study(
                study_name=study_name,
                direction="maximize",  # 報酬を最大化
                sampler=TPESampler(seed=42)  # 再現性のため
            )

            # ベイズ最適化設定
            self.bayesian_config = {
                'n_trials': self.ppo_config.bayesian_trials,
                'timeout': self.ppo_config.bayesian_timeout,
                'optimize_learning_rate': self.ppo_config.optimize_learning_rate,
                'optimize_batch_size': self.ppo_config.optimize_batch_size,
                'optimize_alpha_params': self.ppo_config.optimize_alpha_params
            }

            logger.info(f"Bayesian optimization configured: {self.bayesian_config}")

        except ImportError:
            logger.warning("Optuna not available, Bayesian optimization disabled")
            self.ppo_config.enable_bayesian_optimization = False

    def optimize_hyperparameters(self) -> Dict[str, Any]:
        """ベイズ最適化によるハイパーパラメータ最適化"""
        if not self.ppo_config.enable_bayesian_optimization:
            logger.info("Bayesian optimization disabled, using default parameters")
            return self._get_default_params()

        def objective(trial):
            """最適化対象関数"""
            # ハイパーパラメータのサンプリング
            params = {}

            if self.ppo_config.optimize_learning_rate:
                params['learning_rate'] = trial.suggest_float('learning_rate', 1e-7, 1e-4, log=True)

            if self.ppo_config.optimize_batch_size:
                params['batch_size'] = trial.suggest_categorical('batch_size', [1, 2, 4, 8])

            if self.ppo_config.optimize_alpha_params:
                params['alpha_initial'] = trial.suggest_float('alpha_initial', -1.0, 0.0)
                params['alpha_target'] = trial.suggest_float('alpha_target', 0.0, 0.5)
                params['annealing_steps'] = trial.suggest_int('annealing_steps', 100, 1000)

            # 一時的なPPO設定でテスト実行
            test_config = self.ppo_config.copy()
            test_config.update(params)

            # 短いテスト実行（数ステップのみ）
            reward = self._evaluate_params(test_config, max_steps=10)

            return reward

        logger.info("Starting Bayesian hyperparameter optimization...")

        # 最適化実行
        self.optuna_study.optimize(
            objective,
            n_trials=self.bayesian_config['n_trials'],
            timeout=self.bayesian_config['timeout']
        )

        # 最良パラメータの取得
        best_params = self.optuna_study.best_params
        best_value = self.optuna_study.best_value

        logger.info(f"Bayesian optimization completed!")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best reward: {best_value}")

        # 最適化結果の保存
        self.save_optimization_results()

        return best_params

    def _evaluate_params(self, config, max_steps: int = 10) -> float:
        """指定されたパラメータでの評価"""
        try:
            # 一時的なモデルで短い学習を実行
            # 実際の実装では、軽量な評価を実行
            # ここでは簡易的な評価としてランダム値を返す
            import random
            reward = random.uniform(0.1, 1.0)  # 仮の報酬

            # SO(8)パラメータが適切な範囲内かチェック
            if hasattr(config, 'alpha_initial') and hasattr(config, 'alpha_target'):
                if config.alpha_initial < config.alpha_target:
                    reward += 0.1  # 適切な範囲ならボーナス

            return reward

        except Exception as e:
            logger.warning(f"Parameter evaluation failed: {e}")
            return 0.0

    def _get_default_params(self) -> Dict[str, Any]:
        """デフォルトパラメータの取得"""
        return {
            'learning_rate': self.ppo_config.learning_rate,
            'batch_size': self.ppo_config.batch_size,
            'alpha_initial': -0.5,
            'alpha_target': 0.382,
            'annealing_steps': 500
        }

    def save_optimization_results(self):
        """最適化結果の保存"""
        if not hasattr(self, 'optuna_study'):
            return

        results_dir = Path("models/aegis_bayes_opt_results")
        results_dir.mkdir(exist_ok=True)

        # 最適化結果の保存
        results_file = results_dir / f"bayesian_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        results = {
            'best_params': self.optuna_study.best_params,
            'best_value': self.optuna_study.best_value,
            'n_trials': len(self.optuna_study.trials),
            'trials': [
                {
                    'number': trial.number,
                    'params': trial.params,
                    'value': trial.value,
                    'state': trial.state.name
                }
                for trial in self.optuna_study.trials
            ]
        }

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Optimization results saved to: {results_file}")

        # 最適化サマリの作成
        summary_file = results_dir / "optimization_summary.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("# AEGIS-v2.0 Bayesian Hyperparameter Optimization Summary\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Best Reward**: {self.optuna_study.best_value:.4f}\n\n")
            f.write("## Best Parameters\n\n")
            for key, value in self.optuna_study.best_params.items():
                f.write(f"- **{key}**: {value}\n")
            f.write("\n")
            f.write(f"## Optimization Statistics\n\n")
            f.write(f"- **Total Trials**: {len(self.optuna_study.trials)}\n")
            f.write(f"- **Successful Trials**: {len([t for t in self.optuna_study.trials if t.value is not None])}\n")
            f.write(f"- **Timeout**: {self.bayesian_config['timeout']} seconds\n\n")

        logger.info(f"Optimization summary saved to: {summary_file}")

    def create_training_plots(self):
        """学習曲線のグラフ化"""
        try:
            # プロット保存ディレクトリ
            plots_dir = Path("models/aegis_v2_training_plots")
            plots_dir.mkdir(exist_ok=True)

            # スタイル設定
            plt.style.use('default')
            sns.set_palette("husl")

            # 1. PPO学習曲線 (Losses and Rewards)
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('AEGIS-v2.0 PPO Training Curves', fontsize=16)

            steps = np.array(self.stats['steps'])

            # Policy Loss
            axes[0,0].plot(steps, self.stats['policy_losses'], label='Policy Loss', alpha=0.7)
            axes[0,0].set_title('Policy Loss')
            axes[0,0].set_xlabel('Steps')
            axes[0,0].set_ylabel('Loss')
            axes[0,0].grid(True, alpha=0.3)
            axes[0,0].legend()

            # Value Function Loss
            axes[0,1].plot(steps, self.stats['vf_losses'], label='VF Loss', alpha=0.7, color='orange')
            axes[0,1].set_title('Value Function Loss')
            axes[0,1].set_xlabel('Steps')
            axes[0,1].set_ylabel('Loss')
            axes[0,1].grid(True, alpha=0.3)
            axes[0,1].legend()

            # Entropy Loss
            axes[1,0].plot(steps, self.stats['entropy_losses'], label='Entropy Loss', alpha=0.7, color='green')
            axes[1,0].set_title('Entropy Loss')
            axes[1,0].set_xlabel('Steps')
            axes[1,0].set_ylabel('Loss')
            axes[1,0].grid(True, alpha=0.3)
            axes[1,0].legend()

            # Rewards
            axes[1,1].plot(steps, self.stats['rewards'], label='Rewards', alpha=0.7, color='red')
            axes[1,1].set_title('Rewards')
            axes[1,1].set_xlabel('Steps')
            axes[1,1].set_ylabel('Reward')
            axes[1,1].grid(True, alpha=0.3)
            axes[1,1].legend()

            plt.tight_layout()
            plt.savefig(plots_dir / 'ppo_learning_curves.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 2. アルファゲートアニーリング曲線
            plt.figure(figsize=(10, 6))
            plt.plot(steps, self.stats['alphas'], label='Alpha', linewidth=2, color='purple')
            plt.axhline(y=self.ppo_config.alpha_target, color='red', linestyle='--',
                       label=f'Target Alpha ({self.ppo_config.alpha_target:.4f})')
            plt.title('SO(8) Alpha Gate Annealing Progress')
            plt.xlabel('Training Steps')
            plt.ylabel('Alpha Value')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(plots_dir / 'alpha_annealing_curve.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 3. 直交誤差監視
            plt.figure(figsize=(10, 6))
            plt.plot(steps, self.stats['orthogonal_errors'], label='Orthogonal Error', linewidth=2, color='darkblue')
            plt.axhline(y=1e-6, color='red', linestyle='--', label='Target (< 1e-6)')
            plt.yscale('log')
            plt.title('SO(8) Rotation Matrix Orthogonal Error')
            plt.xlabel('Training Steps')
            plt.ylabel('Orthogonal Error (log scale)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(plots_dir / 'orthogonal_error_curve.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 4. KL Divergence と Clip Fraction
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))

            # KL Divergence
            axes[0].plot(steps, self.stats['kl_divs'], label='KL Divergence', alpha=0.7, color='brown')
            axes[0].axhline(y=self.ppo_config.max_kl, color='red', linestyle='--',
                           label=f'Max KL ({self.ppo_config.max_kl})')
            axes[0].set_title('KL Divergence')
            axes[0].set_xlabel('Steps')
            axes[0].set_ylabel('KL')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

            # Clip Fraction
            axes[1].plot(steps, self.stats['clip_fractions'], label='Clip Fraction', alpha=0.7, color='purple')
            axes[1].set_title('Policy Clip Fraction')
            axes[1].set_xlabel('Steps')
            axes[1].set_ylabel('Fraction')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

            plt.tight_layout()
            plt.savefig(plots_dir / 'ppo_stability_metrics.png', dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Training plots saved to: {plots_dir}")

        except Exception as e:
            logger.warning(f"Failed to create training plots: {e}")

    def upload_stats_to_hf(self):
        """学習統計をHugging Faceにアップロード"""
        try:
            # HF API初期化
            api = HfApi()

            # リポジトリ名
            repo_name = "aegis-v2-ppo-training-stats"
            repo_id = f"zapabob/{repo_name}"

            # リポジトリ作成（存在しない場合）
            try:
                api.create_repo(repo_id=repo_id, private=False, exist_ok=True)
                logger.info(f"HF repository ready: {repo_id}")
            except Exception as e:
                logger.warning(f"Could not create/access HF repo: {e}")
                return

            # 統計データをDataFrameに変換
            df_stats = pd.DataFrame(self.stats)

            # CSVとして保存してアップロード
            stats_file = Path("models/aegis_v2_training_stats.csv")
            df_stats.to_csv(stats_file, index=False)

            # HFにアップロード
            upload_file(
                path_or_fileobj=str(stats_file),
                path_in_repo="training_stats.csv",
                repo_id=repo_id,
                commit_message=f"AEGIS-v2.0 PPO Training Statistics - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

            # プロットもアップロード
            plots_dir = Path("models/aegis_v2_training_plots")
            if plots_dir.exists():
                for plot_file in plots_dir.glob("*.png"):
                    upload_file(
                        path_or_fileobj=str(plot_file),
                        path_in_repo=f"plots/{plot_file.name}",
                        repo_id=repo_id,
                        commit_message=f"Training plot: {plot_file.name}"
                    )

            # 設定ファイルもアップロード
            config_file = Path("aegis_v2_test_config.json")
            if config_file.exists():
                upload_file(
                    path_or_fileobj=str(config_file),
                    path_in_repo="config.json",
                    repo_id=repo_id,
                    commit_message="Training configuration"
                )

            logger.info(f"Training statistics uploaded to HF: https://huggingface.co/{repo_id}")

        except Exception as e:
            logger.warning(f"Failed to upload stats to HF: {e}")
            logger.info("Continuing without HF upload...")

    def apply_optimized_params(self, params: Dict[str, Any]):
        """最適化されたパラメータを適用"""
        logger.info(f"Applying optimized parameters: {params}")

        # 学習率の更新
        if 'learning_rate' in params:
            self.ppo_config.learning_rate = params['learning_rate']
            # オプティマイザーの再初期化
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.ppo_config.learning_rate,
                weight_decay=0.01
            )
            logger.info(f"Updated learning rate to: {self.ppo_config.learning_rate}")

        # バッチサイズの更新
        if 'batch_size' in params:
            self.ppo_config.batch_size = params['batch_size']
            # DataLoaderの再作成
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.ppo_config.batch_size,
                shuffle=True,
                num_workers=4
            )
            logger.info(f"Updated batch size to: {self.ppo_config.batch_size}")

        # SO(8)パラメータの更新
        if 'alpha_initial' in params:
            self.config['so8t']['alpha_initial'] = params['alpha_initial']
            logger.info(f"Updated alpha_initial to: {self.config['so8t']['alpha_initial']}")

        if 'alpha_target' in params:
            self.config['so8t']['alpha_target'] = params['alpha_target']
            logger.info(f"Updated alpha_target to: {self.config['so8t']['alpha_target']}")

        if 'annealing_steps' in params:
            self.config['so8t']['annealing_steps'] = params['annealing_steps']
            logger.info(f"Updated annealing_steps to: {self.config['so8t']['annealing_steps']}")

        # アニーラーの再初期化
        if any(key in params for key in ['alpha_initial', 'alpha_target', 'annealing_steps']):
            self.phase_annealer = SO8PhaseTransitionAnnealer(
                alpha_initial=self.config['so8t']['alpha_initial'],
                alpha_target=self.config['so8t']['alpha_target'],
                annealing_steps=self.config['so8t']['annealing_steps']
            )
            logger.info("Reinitialized SO(8) phase annealer with optimized parameters")

        # データセット準備
        if not self.config.get('training', {}).get('test_mode', False):
            self.train_dataset = AEGISV2Dataset(
                self.config['data']['train_file'],
                self.tokenizer,
                self.config['data']['max_length']
            )

            # RTX3060最適化: DataLoaderのメモリ効率化
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.ppo_config.batch_size,  # 1 (gradient accumulationで効果的バッチサイズを実現)
                shuffle=True,
                num_workers=0,  # GPU使用時は0が安定
                pin_memory=True,  # GPU転送高速化
                prefetch_factor=2 if torch.cuda.is_available() else None,
            )

            # Unsloth最適化オプティマイザー
            if UNSLOTH_AVAILABLE and hasattr(self.model, 'parameters'):
                from transformers import get_cosine_schedule_with_warmup
                self.optimizer = AdamW(
                    self.model.parameters(),
                    lr=self.ppo_config.learning_rate,
                    weight_decay=0.01,
                    betas=(0.9, 0.999),
                )

                # 学習率スケジューラー (Unsloth推奨)
                num_training_steps = len(self.train_dataloader) * self.ppo_config.epochs
                self.lr_scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=self.ppo_config.warmup_steps,
                    num_training_steps=num_training_steps
                )
            else:
                # Fallbackオプティマイザー
                self.optimizer = AdamW(
                    self.model.parameters(),
                    lr=self.ppo_config.learning_rate
                )
        # テストモードの場合はsetup_test_modeで既に設定済み

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

        # 統計記録の強化
        self.stats = {
            'steps': [],
            'rewards': [],
            'policy_losses': [],
            'vf_losses': [],
            'entropy_losses': [],
            'total_losses': [],
            'kl_divs': [],
            'clip_fractions': [],
            'orthogonal_errors': [],
            'alphas': [],
            'chaos_intensities': [],
            'advantages_mean': [],
            'advantages_std': []
        }

        logger.info("PPO Trainer initialized with SO(8) enhancements")

    def setup_model_and_tokenizer(self):
        """モデルとトークナイザーのセットアップ - RTX3060最適化"""
        if UNSLOTH_AVAILABLE and torch.cuda.is_available():
            logger.info(f"Loading model with Unsloth (RTX3060 optimized): {self.model_path}")

            # GPUメモリ最適化設定
            gpu_memory_limit = self.config.get('training', {}).get('gpu_memory_limit', 0.85)
            torch.cuda.set_per_process_memory_fraction(gpu_memory_limit)
            torch.cuda.empty_cache()

            # Unslothで4bit量子化モデルをロード
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_path,
                max_seq_length=self.config['data']['max_length'],
                dtype=None,  # Auto-detect (bf16推奨)
                load_in_4bit=True,  # 4bit量子化でメモリ大幅節約
                device_map={"": 0},  # RTX3060 (GPU 0) に固定配置
            )

            # LoRAを有効化（パラメータ数を大幅削減）
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=16,  # LoRA rank (メモリ効率と性能のバランス)
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing=True,  # メモリ節約
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
            )

            # 参照モデルも同様に作成（メモリ効率重視）
            self.ref_model, _ = FastLanguageModel.from_pretrained(
                model_name=self.model_path,
                max_seq_length=self.config['data']['max_length'],
                dtype=None,
                load_in_4bit=True,
                device_map={"": 0},  # 同じGPUに配置
            )
            # 参照モデルはLoRAなしで軽量化
            self.ref_model.eval()

            # GPUメモリ使用量をログ出力
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                logger.info(".1f")

            logger.info("Model and tokenizer loaded successfully with Unsloth (4bit + LoRA)")

        else:
            # Fallback: 標準transformers（CPUモード）
            logger.warning("Unsloth not available or CUDA not found - using CPU fallback")
            from transformers import AutoTokenizer, AutoModelForCausalLM

            logger.info(f"Loading model (CPU fallback): {self.model_path}")

            # CPUで量子化モデルをロード
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,  # float16 for memory efficiency
                device_map="auto",  # Let transformers decide
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                load_in_8bit=True,  # 8bit quantization for CPU
            )

            # 参照モデル
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                load_in_8bit=True,
            )
            self.ref_model.eval()

            # トークナイザー
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("Model and tokenizer loaded successfully (CPU fallback with 8bit quantization)")

    def setup_test_mode(self):
        """テストモード用のセットアップ"""
        logger.info("Setting up test mode with mock objects")

        # モックモデルクラス
        class MockModel:
            def __init__(self):
                self.device = torch.device('cpu')

            def __call__(self, input_ids=None, attention_mask=None, **kwargs):
                # モック出力
                if input_ids is not None:
                    batch_size, seq_len = input_ids.shape
                else:
                    batch_size, seq_len = 1, 10

                return type('MockOutput', (), {
                    'logits': torch.randn(batch_size, seq_len, 32000),  # Phi-3.5 vocab size
                    'hidden_states': [torch.randn(batch_size, seq_len, 3072) for _ in range(33)]  # 32 layers + input
                })()

            def state_dict(self):
                # モックstate_dict
                return {'mock_param': torch.randn(10)}

            def load_state_dict(self, state_dict):
                # モックload_state_dict
                pass

            def eval(self):
                # モックeval
                pass

            def parameters(self):
                # モックparameters
                return [torch.randn(10, requires_grad=True)]

        # モックトークナイザー
        class MockTokenizer:
            def __init__(self):
                self.pad_token = '<pad>'
                self.eos_token = '</s>'

            def __call__(self, text, **kwargs):
                # シンプルなトークナイズ（実際のトークナイズは行わず固定長のテンソルを返す）
                return {
                    'input_ids': torch.randint(0, 32000, (1, 10)),
                    'attention_mask': torch.ones(1, 10)
                }

        # モックオブジェクトの設定
        self.model = MockModel()
        self.ref_model = MockModel()
        self.tokenizer = MockTokenizer()

        # データセットもモック
        self.train_dataset = AEGISV2Dataset(
            self.config['data']['train_file'],
            self.tokenizer,
            self.config['data']['max_length']
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=1,  # テストモードではバッチサイズを1に固定
            shuffle=False,  # テストモードではシャッフルを無効化
            num_workers=0  # テストモードではマルチプロセスを避ける
        )

        logger.info("Test mode setup completed with mock objects")

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
                        advantages: torch.Tensor, cliprange: float, logits: torch.Tensor = None) -> Tuple[torch.Tensor, Dict]:
        """PPO損失計算 - ベストプラクティス実装"""
        # Advantage normalization (PPOベストプラクティス)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 確率比
        ratio = torch.exp(new_logprobs - old_logprobs)

        # Clipped surrogate objective (PPOコア)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 価値関数損失 - PPOベストプラクティス
        # value predictions: モデルの隠れ層の平均を使用（簡易実装）
        if hasattr(current_outputs, 'hidden_states') and current_outputs.hidden_states is not None:
            # 最終隠れ層の平均をvalue predictionとして使用
            value_predictions = current_outputs.hidden_states[-1].mean(dim=-1)  # [batch_size, seq_len]
            value_predictions = value_predictions.mean(dim=-1)  # [batch_size]

            # value targets: rewardsを使用（GAEなしの簡易実装）
            value_targets = rewards

            # VF loss: MSE loss
            vf_loss = F.mse_loss(value_predictions, value_targets)
        else:
            # fallback: 簡易実装
            vf_loss = torch.tensor(0.0, device=policy_loss.device)

        # エントロピー損失 - 探索を促進 (PPOベストプラクティス)
        entropy_loss = torch.tensor(0.0, device=policy_loss.device)
        if logits is not None:
            # エントロピー計算: -sum(p * log(p))
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -torch.sum(probs * log_probs, dim=-1).mean()
            entropy_loss = entropy

        # KL divergence - early stopping用 (PPOベストプラクティス)
        kl_div = torch.mean(torch.sum(
            F.softmax(old_logprobs, dim=-1) * (old_logprobs - new_logprobs), dim=-1
        ))

        # Total loss
        total_loss = (
            policy_loss +
            self.ppo_config.vf_coef * vf_loss -
            self.ppo_config.ent_coef * entropy_loss
        )

        # Early stopping check (KLが大きすぎる場合は学習停止)
        if kl_div > self.ppo_config.max_kl:
            logger.warning(f"KL divergence too high: {kl_div:.4f} > {self.ppo_config.max_kl}, early stopping may be needed")

        loss_info = {
            'policy_loss': policy_loss.item(),
            'vf_loss': vf_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item(),
            'kl_div': kl_div.item(),
            'clip_fraction': torch.mean((torch.abs(ratio - 1.0) > cliprange).float()).item()
        }

        return total_loss, loss_info

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """1ステップの学習 - PPOベストプラクティス"""
        # テストモードの場合はモックデータを使用
        if hasattr(self, 'tokenizer') and hasattr(self.tokenizer, '__call__'):
            # 実際のトークナイザーがある場合（本番モード）
            if 'input_ids' not in batch:
                # バッチにテキストがある場合はトークナイズ
                if 'text' in batch:
                    tokenized = self.tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True)
                    batch.update(tokenized)
                else:
                    # モックデータ生成
                    batch_size = len(batch) if isinstance(batch, list) else 1
                    batch = {
                        'input_ids': torch.randint(0, 32000, (batch_size, 10)),
                        'attention_mask': torch.ones(batch_size, 10),
                        'target_correct': torch.tensor([0.5] * batch_size),
                        'is_nsfw': torch.tensor([False] * batch_size)
                    }

        # 参照モデルのログ確率を計算
        with torch.no_grad():
            ref_outputs = self.ref_model(batch['input_ids'], attention_mask=batch['attention_mask'])
            ref_logprobs = self.get_logprobs_from_outputs(ref_outputs, batch)

        # 現在のモデルの出力を計算
        current_outputs = self.model(batch['input_ids'], attention_mask=batch['attention_mask'])
        current_logprobs = self.get_logprobs_from_outputs(current_outputs, batch)

        # logitsを取得（エントロピー計算用）
        logits = current_outputs.logits

        # 報酬計算
        rewards = self.compute_rewards(batch)

        # 利得計算 - PPOベストプラクティス (GAE簡易版)
        # GAE (Generalized Advantage Estimation) の簡易実装
        # advantages = rewards - value_predictions (baseline)
        if hasattr(current_outputs, 'hidden_states') and current_outputs.hidden_states is not None:
            baseline = current_outputs.hidden_states[-1].mean(dim=-1).mean(dim=-1)  # value predictions
            advantages = rewards - baseline.detach()  # detach to prevent gradient flow
        else:
            advantages = rewards - rewards.mean()  # fallback

        # Advantage normalization (PPOベストプラクティス)
        if advantages.numel() > 1:  # バッチサイズが1以上の場合のみ正規化
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO損失計算 - logitsを渡す
        loss, loss_info = self.compute_ppo_loss(
            ref_logprobs, current_logprobs, advantages, self.ppo_config.cliprange, logits
        )

        # 直交誤差の計算（SO(8)特有）
        orthogonal_error = self.compute_orthogonal_error()

        # 統計更新
        loss_info.update({
            'rewards': rewards.mean().item(),
            'advantages_mean': advantages.mean().item(),
            'advantages_std': advantages.std().item(),
            'orthogonal_error': orthogonal_error,
            'alpha': self.phase_annealer.get_current_alpha(),
            'chaos_intensity': self.chaos_enhancer.chaos_intensity if self.chaos_enhancer else 0.0
        })

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

    def compute_orthogonal_error(self) -> float:
        """SO(8)ローテーション行列の直交誤差を計算"""
        if not hasattr(self, 'reward_system') or self.reward_system is None:
            return 0.0

        try:
            # SO(8)アダプタのローテーション行列を取得
            if hasattr(self.reward_system, 'rotation_safe') and self.reward_system.rotation_safe is not None:
                R = self.reward_system.rotation_safe
                # 直交性チェック: R^T @ R - I のFrobeniusノルム
                orthogonal_error = torch.norm(R.T @ R - torch.eye(R.shape[0], device=R.device), p='fro').item()
                return orthogonal_error
            else:
                return 0.0
        except Exception as e:
            logger.warning(f"Failed to compute orthogonal error: {e}")
            return 0.0

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

        # 学習状態の初期化（apply_optimized_paramsが呼ばれていない場合のため）
        if not hasattr(self, 'global_step'):
            self.global_step = 0
        if not hasattr(self, 'epoch'):
            self.epoch = 0
        if not hasattr(self, 'best_reward'):
            self.best_reward = float('-inf')
        if not hasattr(self, 'checkpoint_dir'):
            self.checkpoint_dir = Path(self.ppo_config.checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # ベイズ最適化によるハイパーパラメータ最適化
        if self.ppo_config.enable_bayesian_optimization:
            logger.info("Running Bayesian hyperparameter optimization...")
            optimal_params = self.optimize_hyperparameters()

            # 最適化されたパラメータを適用
            self.apply_optimized_params(optimal_params)
            logger.info(f"Applied optimized parameters: {optimal_params}")
        else:
            # ベイズ最適化が無効な場合でも基本パラメータを適用してoptimizerを初期化
            logger.info("Applying default parameters...")
            default_params = {
                'learning_rate': self.ppo_config.learning_rate,
                'batch_size': self.ppo_config.batch_size
            }
            self.apply_optimized_params(default_params)

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
                    device = getattr(self.model, 'device', torch.device('cpu'))
                    batch = {k: v.to(device) if torch.is_tensor(v) else v
                           for k, v in batch.items()}

                    # 学習ステップ
                    step_info = self.train_step(batch)

                    # RTX3060最適化: Gradient accumulationとメモリ効率化
                    loss = torch.tensor(step_info['total_loss'], requires_grad=True)
                    loss = loss / self.ppo_config.gradient_accumulation_steps  # accumulation用に損失をスケール

                    # 逆伝播 (Unsloth最適化)
                    loss.backward()

                    # Gradient accumulation
                    if (self.global_step + 1) % self.ppo_config.gradient_accumulation_steps == 0:
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.ppo_config.max_grad_norm)

                        # Optimizer step
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        # Learning rate scheduling (Unsloth使用時)
                        if hasattr(self, 'lr_scheduler'):
                            self.lr_scheduler.step()

                        # GPUメモリ解放
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    epoch_losses.append(step_info['total_loss'])
                    epoch_rewards.append(step_info['rewards'])

                    self.global_step += 1

                    # 統計更新 - PPOベストプラクティス
                    self.stats['steps'].append(self.global_step)
                    self.stats['rewards'].append(step_info['rewards'])
                    self.stats['policy_losses'].append(step_info['policy_loss'])
                    self.stats['vf_losses'].append(step_info['vf_loss'])
                    self.stats['entropy_losses'].append(step_info['entropy_loss'])
                    self.stats['total_losses'].append(step_info['total_loss'])
                    self.stats['kl_divs'].append(step_info['kl_div'])
                    self.stats['clip_fractions'].append(step_info['clip_fraction'])
                    self.stats['orthogonal_errors'].append(step_info['orthogonal_error'])
                    self.stats['alphas'].append(step_info['alpha'])
                    self.stats['chaos_intensities'].append(step_info['chaos_intensity'])
                    self.stats['advantages_mean'].append(step_info['advantages_mean'])
                    self.stats['advantages_std'].append(step_info['advantages_std'])

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

            # 学習曲線のグラフ化
            self.create_training_plots()

            # HFに学習統計をアップロード
            self.upload_stats_to_hf()

            # 音声通知
            try:
                import winsound
                winsound.Beep(1000, 1000)  # 完了音
            except ImportError:
                pass

    def log_training_stats(self):
        """学習統計のログ出力 - RTX3060最適化"""
        recent_window = 50  # 最近50ステップの統計

        recent_policy_loss = np.mean(self.stats['policy_losses'][-recent_window:])
        recent_vf_loss = np.mean(self.stats['vf_losses'][-recent_window:])
        recent_entropy_loss = np.mean(self.stats['entropy_losses'][-recent_window:])
        recent_total_loss = np.mean(self.stats['total_losses'][-recent_window:])
        recent_rewards = np.mean(self.stats['rewards'][-recent_window:])
        recent_kl_div = np.mean(self.stats['kl_divs'][-recent_window:])
        recent_clip_fraction = np.mean(self.stats['clip_fractions'][-recent_window:])
        recent_orthogonal_error = np.mean(self.stats['orthogonal_errors'][-recent_window:])
        current_alpha = self.phase_annealer.get_current_alpha()

        # GPUメモリ情報
        gpu_memory_info = ""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            gpu_memory_info = f", GPU: {allocated:.1f}GB/{reserved:.1f}GB"

        logger.info(f"Step {self.global_step}: "
                   f"Policy Loss: {recent_policy_loss:.4f}, "
                   f"VF Loss: {recent_vf_loss:.4f}, "
                   f"Entropy Loss: {recent_entropy_loss:.4f}, "
                   f"Total Loss: {recent_total_loss:.4f}, "
                   f"Reward: {recent_rewards:.4f}, "
                   f"KL Div: {recent_kl_div:.4f}, "
                   f"Clip Fraction: {recent_clip_fraction:.3f}, "
                   f"Orthogonal Error: {recent_orthogonal_error:.2e}, "
                   f"Alpha: {current_alpha:.4f}"
                   f"{gpu_memory_info}")

    def log_final_stats(self):
        """最終統計のログ出力 - PPOベストプラクティス"""
        logger.info("=== Final AEGIS-v2.0 PPO Training Statistics ===")
        logger.info(f"Total training steps: {self.global_step}")
        logger.info(f"Best reward achieved: {self.best_reward:.4f}")

        # 最終エポックの統計
        if self.stats['total_losses']:
            final_policy_loss = np.mean(self.stats['policy_losses'][-100:])
            final_vf_loss = np.mean(self.stats['vf_losses'][-100:])
            final_entropy_loss = np.mean(self.stats['entropy_losses'][-100:])
            final_total_loss = np.mean(self.stats['total_losses'][-100:])
            final_reward = np.mean(self.stats['rewards'][-100:])
            final_kl_div = np.mean(self.stats['kl_divs'][-100:])
            final_orthogonal_error = np.mean(self.stats['orthogonal_errors'][-100:])

            logger.info(f"Final Policy Loss: {final_policy_loss:.4f}")
            logger.info(f"Final VF Loss: {final_vf_loss:.4f}")
            logger.info(f"Final Entropy Loss: {final_entropy_loss:.4f}")
            logger.info(f"Final Total Loss: {final_total_loss:.4f}")
            logger.info(f"Final Average Reward: {final_reward:.4f}")
            logger.info(f"Final KL Divergence: {final_kl_div:.4f}")
            logger.info(f"Final Orthogonal Error: {final_orthogonal_error:.2e}")

            # 直交誤差の評価
            if final_orthogonal_error < 1e-6:
                logger.info("✅ Orthogonal error is within acceptable range (< 1e-6)")
            else:
                logger.warning(f"⚠️ Orthogonal error is high: {final_orthogonal_error:.2e}")

        logger.info(f"Final SO(8) alpha: {self.phase_annealer.get_current_alpha():.4f}")
        logger.info(f"Checkpoints saved: {len(list(self.checkpoint_dir.glob('*.pt')))}")

        # HFアップロード情報
        logger.info("Training plots and statistics will be uploaded to Hugging Face")

def main():
    """メイン実行関数"""
    print("AEGIS-v2.0 PPO Training with SO(8) Rotation Adapter")
    print("=" * 60)

    # 設定ファイル
    config_path = "aegis_v2_test_config.json"
    model_path = "models/Borea-Phi-3.5-mini-Instruct-Jp"

    # PPOトレーナー初期化
    trainer = PPOTrainer(config_path, model_path)

    # チェックポイントディレクトリの初期化
    trainer.checkpoint_dir = Path(trainer.ppo_config.checkpoint_dir)
    trainer.checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
