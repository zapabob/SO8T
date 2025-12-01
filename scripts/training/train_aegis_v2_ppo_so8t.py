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
UNSLOTH_AVAILABLE = False  # Force disable Unsloth for testing
BITSANDBYTES_AVAILABLE = None

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

# Initialize BitsAndBytes after logger setup
try:
    from transformers import BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
    logger.info("BitsAndBytes + PEFT available - using 4bit quantization")
except (ImportError, Exception) as e2:
    BITSANDBYTES_AVAILABLE = False
    logger.warning(f"BitsAndBytes not available ({e2}) - using standard transformers with CPU fallback")

# Initialize Unsloth
try:
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported
    UNSLOTH_AVAILABLE = True
    logger.info("Unsloth available - using Unsloth for optimized training")
except (ImportError, Exception) as e:
    logger.info(f"Unsloth not available ({e}) - falling back to bitsandbytes + PEFT")
    UNSLOTH_AVAILABLE = False

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

    # SO(8)設定
    alpha_initial: float = -0.5
    alpha_target: float = 0.382
    annealing_steps: int = 500
    chaos_intensity: float = 0.1

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
        logger.info("=== Starting PPOTrainer initialization ===")
        self.config_path = config_path
        self.model_path = model_path

        # 設定読み込み
        logger.info(f"Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        logger.info("Config loaded successfully")

        # PPO設定
        logger.info("Initializing PPO config...")
        self.ppo_config = PPOConfig()
        logger.info("PPO config initialized")

        # SO(8)コンポーネント初期化
        logger.info("Initializing SO(8) components...")
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
        test_mode = self.config.get('training', {}).get('test_mode', False)
        logger.info(f"Test mode check: test_mode={test_mode}")
        if not test_mode:
            logger.info("Calling setup_model_and_tokenizer()")
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
        from tqdm import tqdm
        import time

        logger.info("=== Starting model and tokenizer setup ===")
        use_unsloth = self.config.get('training', {}).get('use_unsloth', True)
        logger.info(f"Unsloth requested: {use_unsloth}, Available: {UNSLOTH_AVAILABLE}, CUDA: {torch.cuda.is_available()}")

        if use_unsloth and UNSLOTH_AVAILABLE and torch.cuda.is_available():
            with tqdm(total=100, desc="Model Setup", unit="%") as setup_bar:
                setup_bar.update(5)
                logger.info(f"Loading model with Unsloth (RTX3060 optimized): {self.model_path}")

                # GPUメモリ最適化設定
                logger.info("Setting up GPU memory optimization...")
                gpu_memory_limit = self.config.get('training', {}).get('gpu_memory_limit', 0.85)
                torch.cuda.set_per_process_memory_fraction(gpu_memory_limit)
                torch.cuda.empty_cache()
                setup_bar.update(10)
                setup_bar.set_postfix({"step": "GPU setup"})

            # Unslothで4bit量子化モデルをロード
            logger.info("Loading model with Unsloth...")
            start_time = time.time()
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
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f} seconds")
            setup_bar.update(40)
            setup_bar.set_postfix({"step": "Model loaded"})

            # 参照モデルも同様に作成（メモリ効率重視）
            logger.info("Loading reference model...")
            ref_start_time = time.time()
            self.ref_model, _ = FastLanguageModel.from_pretrained(
                model_name=self.model_path,
                max_seq_length=self.config['data']['max_length'],
                dtype=None,
                load_in_4bit=True,
                device_map={"": 0},  # 同じGPUに配置
            )
            # 参照モデルはLoRAなしで軽量化
            self.ref_model.eval()
            ref_load_time = time.time() - ref_start_time
            logger.info(f"Reference model loaded in {ref_load_time:.2f} seconds")
            setup_bar.update(20)
            setup_bar.set_postfix({"step": "Ref model loaded"})

            # GPUメモリ使用量をログ出力
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                logger.info(".1f")
                setup_bar.set_postfix({"step": f"GPU: {allocated:.1f}GB"})

            logger.info("Model and tokenizer loaded successfully with Unsloth (4bit + LoRA)")
            setup_bar.update(25)
            setup_bar.set_postfix({"step": "Unsloth setup complete"})

            # RTX3060最適化: SO8Tアダプターがモデルにアタッチされていることを確認
            self._ensure_so8t_adapter_attached()

            # Unsloth使用時も元モデルの重みを凍結
            self._freeze_base_model_weights()
        else:
            # Fallback: 標準transformers（CPUモード）
            logger.warning("Unsloth not available or CUDA not found - using CPU fallback")
            with tqdm(total=100, desc="Model Setup", unit="%") as setup_bar:
                setup_bar.update(5)
                logger.info(f"Entered CPU fallback path for model: {self.model_path}")
                from transformers import AutoTokenizer, AutoModelForCausalLM
                setup_bar.update(10)
                setup_bar.set_postfix({"step": "CPU mode setup"})

            logger.info(f"Loading model (CPU fallback): {self.model_path}")

            # CPUで標準モデルをロード（BitsAndBytesはCPUでサポートされない）
            logger.info("Loading model on CPU without quantization")
            start_time = time.time()
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,  # float16 for memory efficiency
                    device_map="cpu",  # Force CPU
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    # BitsAndBytes 8bit quantization removed for CPU compatibility
                )
            except AttributeError as e:
                if "so8_adapter" in str(e) or "SCB" in str(e):
                    logger.warning(f"Model structure mismatch detected: {e}")
                    logger.warning("Attempting to load model by filtering incompatible parameters...")

                    # モデルを一旦空で初期化
                    from transformers import Phi3Config
                    config = Phi3Config.from_pretrained(self.model_path)
                    self.model = AutoModelForCausalLM.from_config(config)

                    # state_dictを読み込んで互換性のないパラメータをフィルタリング
                    try:
                        state_dict = torch.load(
                            Path(self.model_path) / "pytorch_model.bin",
                            map_location="cpu",
                            weights_only=True
            )
                    except FileNotFoundError:
                        # safetensors形式の場合
                        from safetensors.torch import load_file
                        state_dict = {}
                        for safetensor_file in Path(self.model_path).glob("*.safetensors"):
                            state_dict.update(load_file(safetensor_file, device="cpu"))

                    # SO8Tアダプター関連の互換性のないパラメータを除去
                    filtered_state_dict = {}
                    incompatible_keys = []
                    for key, value in state_dict.items():
                        # SCBパラメータを除去（RTX3060最適化での構造不一致）
                        if "so8_adapter.so8_gate.noncommutative_proj.SCB" in key:
                            incompatible_keys.append(key)
                            continue
                        # その他のSO8Tアダプター関連の互換性のないパラメータ
                        if "so8_adapter" in key and ("SCB" in key or "legacy" in key):
                            incompatible_keys.append(key)
                            continue
                        filtered_state_dict[key] = value

                    if incompatible_keys:
                        logger.warning(f"Skipped {len(incompatible_keys)} incompatible SO8T parameters:")
                        for key in incompatible_keys[:5]:  # 最初の5つだけ表示
                            logger.warning(f"  - {key}")
                        if len(incompatible_keys) > 5:
                            logger.warning(f"  ... and {len(incompatible_keys) - 5} more")

                    # フィルタリングしたstate_dictをロード
                    missing_keys, unexpected_keys = self.model.load_state_dict(filtered_state_dict, strict=False)
                    if missing_keys:
                        logger.warning(f"Missing keys: {missing_keys}")
                    if unexpected_keys:
                        logger.warning(f"Unexpected keys: {unexpected_keys}")

                    logger.info("Model loaded successfully with filtered parameters")
                    load_time = time.time() - start_time
                    logger.info(f"CPU model loaded in {load_time:.2f} seconds")
                    setup_bar.update(40)
                    setup_bar.set_postfix({"step": "CPU model loaded"})
                else:
                    raise

            # RTX3060最適化: SO8Tアダプターがモデルにアタッチされていない場合は初期化
            logger.info("Checking SO8T adapter attachment...")
            self._ensure_so8t_adapter_attached()
            setup_bar.update(30)
            setup_bar.set_postfix({"step": "SO8T adapter attached"})

            # 元モデルの重みを凍結（SO8Tアダプター部分のみ学習）
            self._freeze_base_model_weights()

    def _freeze_base_model_weights(self):
        """元モデルの重みを凍結（SO8Tアダプター部分のみ学習対象に）"""
        try:
            logger.info("Freezing base model weights, keeping only SO8T adapter trainable...")

            # base modelのパラメータを凍結
            for param in self.model.parameters():
                param.requires_grad = False

            # SO8Tアダプター部分のみ学習対象に
            trainable_params = 0
            so8t_params = 0

            # デバッグ: named_modulesでSO8Tアダプターを探す
            so8t_modules_found = []
            for name, module in self.model.named_modules():
                if 'so8_adapter' in name:
                    so8t_modules_found.append(name)
                    logger.info(f"Found SO8T module: {name}")
                    for param in module.parameters():
                        param.requires_grad = True
                        so8t_params += param.numel()
                        trainable_params += param.numel()

            logger.info(f"SO8T modules found: {so8t_modules_found}")

            # 統計情報表示
            # SO8Tアダプター以外のパラメータ数を計算
            base_model_params = 0
            for name, param in self.model.named_parameters():
                if 'so8_adapter' not in name:
                    base_model_params += param.numel()

            total_params = base_model_params + so8t_params
            frozen_params = base_model_params

            logger.info(f"Model freezing completed:")
            logger.info(f"  Total parameters: {total_params:,}")
            logger.info(f"  Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
            logger.info(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
            logger.info(f"  SO8T adapter parameters: {so8t_params:,}")

            if trainable_params == 0:
                logger.warning("No trainable parameters found! Check SO8T adapter structure.")

        except Exception as e:
            logger.error(f"Failed to freeze base model weights: {e}")
            raise

    def _ensure_so8t_adapter_attached(self):
        """RTX3060最適化: SO8Tアダプターがモデルにアタッチされていることを確認"""
        try:
            logger.info("Checking SO8T adapter attachment...")

            # モデルにSO8Tアダプターが存在するか確認
            has_so8t_adapter = False
            for layer_idx, layer in enumerate(self.model.model.layers):
                if hasattr(layer, 'so8_adapter'):
                    has_so8t_adapter = True
                    break

            if not has_so8t_adapter:
                logger.warning("SO8T adapter not found in model, initializing...")
                self._initialize_so8t_adapter()
            else:
                logger.info("SO8T adapter already attached to model")

        except Exception as e:
            logger.error(f"Failed to ensure SO8T adapter attachment: {e}")
            raise

    def _initialize_so8t_adapter(self):
        """RTX3060向けSO8Tアダプター初期化"""
        try:
            logger.info("Initializing SO8T adapter for RTX3060...")

            # SO8Tアダプターをインポート
            import sys
            import os
            import importlib.util

            # モデルディレクトリへのパスを取得
            models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
            adapter_file = os.path.join(models_dir, 'Borea-Phi-3.5-mini-Instruct-Jp', 'so8_rotation_adapter.py')

            # ファイルを直接インポート
            spec = importlib.util.spec_from_file_location("so8_rotation_adapter", adapter_file)
            so8_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(so8_module)
            SO8RotationGate = so8_module.SO8RotationGate

            num_layers = len(self.model.model.layers)
            hidden_size = self.model.config.hidden_size

            # 中間層(4-11)にSO8Tアダプターをアタッチ
            so8t_from = 4
            so8t_to = min(12, num_layers)  # RTX3060のメモリ制約を考慮

            for layer_idx in range(so8t_from, so8t_to):
                layer = self.model.model.layers[layer_idx]

                # SO8Tアダプターをアタッチ
                so8t_adapter = SO8RotationGate(hidden_size=hidden_size)
                layer.so8_adapter = so8t_adapter

                # アダプターをモデルに登録（トレーニングパラメータとして認識されるように）
                self.model.add_module(f"so8_adapter_{layer_idx}", so8t_adapter)

                # GPUに移動
                if torch.cuda.is_available():
                    so8t_adapter.cuda()

                logger.info(f"Attached SO8T adapter to layer {layer_idx}")

            logger.info(f"SO8T adapter initialized for layers {so8t_from}-{so8t_to-1}")

        except Exception as e:
            logger.error(f"Failed to initialize SO8T adapter: {e}")
            raise

    def _reinitialize_so8t_adapter(self):
        """
        SO8Tアダプターの構造を再初期化（パラメータ不一致時の対応）
        # 例: 40層Transformerなら初期層はそのままバニラ、中間4-11層にSO8T残差アダプターを付与
        """
        try:
            logger.info("Reinitializing SO8T adapter structure (attach SO8T adapter to mid layers 4-11)...")

            num_layers = len(self.model.model.layers)
            so8t_from = 4    # inclusive, 0-based
            so8t_to = 12     # exclusive (so covers layers 4-11: 4,5,...,11)

            for layer_idx, layer in enumerate(self.model.model.layers):
                if so8t_from <= layer_idx < so8t_to:
                    # 中間層(4-11)にはSO8Tアダプタを付与・修正
                    if hasattr(layer, 'so8_adapter'):
                        so8_adapter = layer.so8_adapter
                        if hasattr(so8_adapter, 'so8_gate'):
                            so8_gate = so8_adapter.so8_gate
                            if hasattr(so8_gate, 'noncommutative_proj'):
                                if not isinstance(so8_gate.noncommutative_proj, nn.Linear):
                                    logger.warning(f"Fixing noncommutative_proj structure in layer {layer_idx}")
                                    hidden_size = so8_gate.hidden_size
                                    so8_gate.noncommutative_proj = nn.Linear(hidden_size, hidden_size, bias=False)
                                    if torch.cuda.is_available():
                                        so8_gate.noncommutative_proj = so8_gate.noncommutative_proj.cuda()
                    else:
                        logger.warning(f"Layer {layer_idx}: SO8T adapter not found, consider instantiating adapter here if required.")
                else:
                    # 初期層/後段層はSO8Tを除去（またはスキップ）
                    if hasattr(layer, 'so8_adapter'):
                        logger.info(f"Layer {layer_idx}: Removing SO8T adapter for vanilla mode.")
                        delattr(layer, 'so8_adapter')
            logger.info("SO8T adapter structure reinitialized for target mid layers (4-11).")

        except Exception as e:
            logger.error(f"Failed to reinitialize SO8T adapter: {e}")
            # 再初期化に失敗しても続行（警告のみ）
def setup_reference_model(self):
    """
    参照モデルのセットアップ（メインモデルと同じロジック）
    """
    try:
        from transformers import AutoModelForCausalLM
        logger.info("Setting up reference model (same logic as main model)...")

        # 参照モデルのロード
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            load_in_8bit=True,
        )
    except AttributeError as e:
        if "so8_adapter" in str(e) or "SCB" in str(e):
            logger.warning(f"Reference model structure mismatch detected: {e}")
            logger.warning("Loading reference model by filtering incompatible parameters...")

            # 参照モデルも同じ方法でロード
            # Fix: mimic the rest of the model loading, fallback if Phi3Config doesn't exist
            try:
                from transformers import Phi3Config, AutoModelForCausalLM
                config = Phi3Config.from_pretrained(self.model_path)
                self.ref_model = AutoModelForCausalLM.from_config(config)
            except (ImportError, ModuleNotFoundError, AttributeError):
                # Fallback if Phi3Config not available: use generic config
                config = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    load_in_8bit=True,
                ).config
                self.ref_model = AutoModelForCausalLM.from_config(config)

            # state_dictを読み込んで互換性のないパラメータをフィルタリング
            try:
                state_dict = torch.load(
                    Path(self.model_path) / "pytorch_model.bin",
                    map_location="cpu",
                    weights_only=True
                )
            except FileNotFoundError:
                # safetensors形式の場合
                from safetensors.torch import load_file
                state_dict = {}
            for safetensor_file in Path(self.model_path).glob("*.safetensors"):
                state_dict.update(load_file(safetensor_file, device="cpu"))

            # SO8Tアダプター関連の互換性のないパラメータを除去
            filtered_state_dict = {}
            incompatible_keys = []
            for key, value in state_dict.items():
                if "so8_adapter.so8_gate.noncommutative_proj.SCB" in key:
                    incompatible_keys.append(key)
                    continue
                if "so8_adapter" in key and ("SCB" in key or "legacy" in key):
                    incompatible_keys.append(key)
                    continue
                filtered_state_dict[key] = value

            if incompatible_keys:
                logger.warning(f"Skipped {len(incompatible_keys)} incompatible SO8T parameters in ref model:")
                for key in incompatible_keys[:5]:  # 最初の5つだけ表示
                    logger.warning(f"  - {key}")
                if len(incompatible_keys) > 5:
                    logger.warning(f"  ... and {len(incompatible_keys) - 5} more")

            # フィルタリングしたstate_dictをロード
            missing_keys, unexpected_keys = self.ref_model.load_state_dict(filtered_state_dict, strict=False)
            if missing_keys:
                logger.warning(f"Ref model missing keys: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Ref model unexpected keys: {unexpected_keys}")

            logger.info("Reference model loaded successfully with filtered parameters")

            # 参照モデルは完全に凍結（学習対象外）
            for param in self.ref_model.parameters():
                param.requires_grad = False
            self.ref_model.eval()

        logger.info("Reference model setup completed successfully")                                                                                                                                                                                                                                                                                                                                                                                                                                                      

    def _train_ppo(self):                                                                                                           
        """PPOトレーニングメインループ"""
        try:
            logger.info("Starting PPO training...")

            # tqdmで詳細な進捗表示
            progress_bar = tqdm(
                range(self.ppo_config.max_steps),
                desc="SO8T PPO Training",
                unit="step",
                ncols=120,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )

            for step in range(self.ppo_config.max_steps):
                # 経験収集
                experiences = self._collect_experiences()

                # PPO更新
                train_info = self._update_ppo(experiences)

                # 統計記録
                self._log_training_stats(step, train_info)

                # 詳細な進捗バー更新
                progress_bar.set_postfix({
                    'loss': f'{train_info.get("total_loss", 0):.4f}',
                    'reward': f'{train_info.get("reward", 0):.4f}',
                    'alpha': f'{train_info.get("alpha", 0):.4f}',
                    'lr': f'{self.ppo_config.learning_rate:.2e}',
                    'kl': f'{train_info.get("kl_div", 0):.4f}'
                })
                progress_bar.update(1)

                # 定期チェックポイント保存
                if step % 50 == 0:
                    self._save_checkpoint(step)

            progress_bar.close()
            logger.info("=== PPO training completed successfully! ===")
            logger.info(f"Total steps trained: {self.ppo_config.max_steps}")
            logger.info(f"Final learning rate: {self.ppo_config.learning_rate:.2e}")
            logger.info(f"Final alpha value: {self.config.get('so8t', {}).get('alpha_target', 'N/A')}")
            logger.info("SO8T thinking model training completed!")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def _collect_experiences(self):
        """経験収集"""
        # 簡易実装 - 実際のPPOではより複雑
        return {
            'observations': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': []
        }

    def _update_ppo(self, experiences):
        """PPO更新"""
        # 簡易実装 - 実際のPPO損失計算
        return {
            'total_loss': 0.1,
            'policy_loss': 0.05,
            'value_loss': 0.03,
            'entropy_loss': 0.02,
            'reward': 0.8,
            'kl_div': 0.01
        }

    def _log_training_stats(self, step, train_info):
        """トレーニング統計記録"""
        self.stats['steps'].append(step)
        self.stats['rewards'].append(train_info.get('reward', 0))
        self.stats['policy_losses'].append(train_info.get('policy_loss', 0))
        self.stats['vf_losses'].append(train_info.get('value_loss', 0))
        self.stats['entropy_losses'].append(train_info.get('entropy_loss', 0))
        self.stats['total_losses'].append(train_info.get('total_loss', 0))
        self.stats['kl_divs'].append(train_info.get('kl_div', 0))

    def _save_checkpoint(self, step):
        """チェックポイント保存"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        try:
            torch.save({
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'stats': self.stats,
                'config': self.config
            }, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    # ---- 修正版（コード未到達箇所の実行不能部分を削除・整理） ----
    # except Exception: のブロック直下でインデントがおかしくなり、通常到達しない"mock"モデルの生成が
    # 例外処理に隠れていました。これを削除・整理します。
    # （本来この箇所は、init/モデルロード例外時に適切なハンドリング等で配置する必要がある）

    # raise以降のコード（MockModelなどインデントがおかしかった部分全部）は到達不可能なので削除し、
    # 本来例外ハンドリングでMockModelを使いたい場合は、except:内で代入処理に分離して書き直すべきです。
    # ここでは、未到達・実行不能な部分をカットします。

    def train(self):
        """PPOトレーニングを開始"""
        logger.info("Starting SO8T PPO training...")
        logger.info(f"Max steps: {self.ppo_config.max_steps}")
        self._train_ppo()

    def _train_ppo(self):
        """PPOトレーニングメインループ"""
        try:
            logger.info("Starting PPO training...")

            # tqdmで詳細な進捗表示
            progress_bar = tqdm(
                range(self.ppo_config.max_steps),
                desc="SO8T PPO Training",
                unit="step",
                ncols=120,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )

            for step in range(self.ppo_config.max_steps):
                # 経験収集
                experiences = self._collect_experiences()

                # PPO更新
                train_info = self._update_ppo(experiences)

                # 統計記録
                self._log_training_stats(step, train_info)

                # 詳細な進捗バー更新
                progress_bar.set_postfix({
                    'loss': f'{train_info.get("total_loss", 0):.4f}',
                    'reward': f'{train_info.get("reward", 0):.4f}',
                    'alpha': f'{train_info.get("alpha", 0):.4f}',
                    'lr': f'{self.ppo_config.learning_rate:.2e}',
                    'kl': f'{train_info.get("kl_div", 0):.4f}'
                })
                progress_bar.update(1)

                # 定期チェックポイント保存
                if step % 50 == 0:
                    self._save_checkpoint(step)

            progress_bar.close()
            logger.info("=== PPO training completed successfully! ===")
            logger.info(f"Total steps trained: {self.ppo_config.max_steps}")
            logger.info(f"Final learning rate: {self.ppo_config.learning_rate:.2e}")
            logger.info(f"Final alpha value: {self.config.get('so8t', {}).get('alpha_target', 'N/A')}")
            logger.info("SO8T thinking model training completed!")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    import torch
    import numpy as np

    def _collect_experiences(self):
        """経験収集 - PPO標準に基づくバッチ分割の経験収集"""
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_dones = []
        batch_log_probs = []
        batch_values = []

        obs = self.env.reset()
        for _ in range(self.ppo_config.rollout_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits, value = self.model(obs_tensor)
                action_dist = torch.distributions.Categorical(logits=logits)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
            
            next_obs, reward, done, info = self.env.step(action.item())

            batch_obs.append(obs)
            batch_actions.append(action.item())
            batch_rewards.append(reward)
            batch_dones.append(done)
            batch_log_probs.append(log_prob.item())
            batch_values.append(value.item())

            obs = next_obs
            if done:
                obs = self.env.reset()

        # GAEによるAdvantage計算
        returns, advantages = self._compute_gae(
            rewards=batch_rewards,
            values=batch_values,
            dones=batch_dones
        )

        return {
            'observations': np.array(batch_obs, dtype=np.float32),
            'actions': np.array(batch_actions, dtype=np.int64),
            'rewards': np.array(batch_rewards, dtype=np.float32),
            'values': np.array(batch_values, dtype=np.float32),
            'log_probs': np.array(batch_log_probs, dtype=np.float32),
            'advantages': advantages,
            'returns': returns
        }

    def _compute_gae(self, rewards, values, dones, gamma=0.99, lam=0.95):
        """Generalized Advantage Estimator (GAE)"""
        advantages = []
        gae = 0
        values = values + [0]  # terminal value
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t+1] * mask - values[t]
            gae = delta + gamma * lam * mask * gae
            advantages.insert(0, gae)
        returns = [adv + v for adv, v in zip(advantages, values[:-1])]
        return np.array(returns, dtype=np.float32), np.array(advantages, dtype=np.float32)

    def _update_ppo(self, experiences):
        """PPO更新 - ミニバッチ反復・クリッピング適用"""
        batch_size = len(experiences['observations'])
        batch_inds = np.arange(batch_size)
        minibatch_size = self.ppo_config.minibatch_size
        num_epochs = self.ppo_config.epochs

        observations = torch.tensor(experiences['observations'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(experiences['actions'], dtype=torch.int64).to(self.device)
        old_log_probs = torch.tensor(experiences['log_probs'], dtype=torch.float32).to(self.device)
        returns = torch.tensor(experiences['returns'], dtype=torch.float32).to(self.device)
        advantages = torch.tensor(experiences['advantages'], dtype=torch.float32).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # normalize

        total_loss = 0.0
        policy_loss = 0.0
        value_loss = 0.0
        entropy_loss = 0.0
        kl_div = 0.0

        for _ in range(num_epochs):
            np.random.shuffle(batch_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = batch_inds[start:end]

                logits, values = self.model(observations[mb_inds])
                action_dist = torch.distributions.Categorical(logits=logits)
                entropy = action_dist.entropy().mean()
                new_log_probs = action_dist.log_prob(actions[mb_inds])

                ratio = (new_log_probs - old_log_probs[mb_inds]).exp()
                surr1 = ratio * advantages[mb_inds]
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_config.clip_range,
                                          1.0 + self.ppo_config.clip_range) * advantages[mb_inds]
                policy_loss_mb = -torch.min(surr1, surr2).mean()

                value_pred = values.squeeze(-1)
                value_loss_mb = torch.nn.functional.mse_loss(value_pred, returns[mb_inds])

                loss = (policy_loss_mb
                        + self.ppo_config.vf_coef * value_loss_mb
                        - self.ppo_config.entropy_coef * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.ppo_config.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    kl_mb = (old_log_probs[mb_inds] - new_log_probs).mean().item()

                total_loss += loss.item()
                policy_loss += policy_loss_mb.item()
                value_loss += value_loss_mb.item()
                entropy_loss += entropy.item()
                kl_div += kl_mb

        avg_steps = num_epochs * ((batch_size + minibatch_size - 1) // minibatch_size)
        results = {
            'total_loss': total_loss / avg_steps,
            'policy_loss': policy_loss / avg_steps,
            'value_loss': value_loss / avg_steps,
            'entropy_loss': entropy_loss / avg_steps,
            'reward': np.mean(experiences['rewards']),
            'kl_div': kl_div / avg_steps
        }
        return results

    def _log_training_stats(self, step, train_info):
        """トレーニング統計記録 PPO標準"""
        self.stats['steps'].append(step)
        self.stats['rewards'].append(float(train_info.get('reward', 0)))
        self.stats['policy_losses'].append(float(train_info.get('policy_loss', 0)))
        self.stats['vf_losses'].append(float(train_info.get('value_loss', 0)))
        self.stats['entropy_losses'].append(float(train_info.get('entropy_loss', 0)))
        self.stats['total_losses'].append(float(train_info.get('total_loss', 0)))
        self.stats['kl_divs'].append(float(train_info.get('kl_div', 0)))

    def _save_checkpoint(self, step):
        """チェックポイント保存"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        try:
            torch.save({
                'step': step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'stats': self.stats,
                'config': self.config
            }, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")


def main():
    """メイン実行関数"""
    print("Starting SO8T PPO training script...")
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='SO8T PPO Training')
    parser.add_argument('--config', type=str, default='aegis_v2_test_config.json',
                        help='Path to config file')
    parser.add_argument('--model_path', type=str,
                        default='models/Borea-Phi-3.5-mini-Instruct-Jp',
                        help='Path to model')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Maximum training steps (overrides config)')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='Path to checkpoint to load')

    args = parser.parse_args()

    try:
        # PPOトレーナーの初期化
        trainer = PPOTrainer(args.config, args.model_path)

        # 最大ステップ数のオーバーライド
        if args.max_steps is not None:
            trainer.ppo_config.max_steps = args.max_steps
            print(f"Overriding max_steps to: {args.max_steps}")

        # チェックポイントのロード
        if args.load_checkpoint:
            trainer.load_checkpoint(args.load_checkpoint)

        # トレーニング実行
        trainer.train()

    except Exception as e:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()