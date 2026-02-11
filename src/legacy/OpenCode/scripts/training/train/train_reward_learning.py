#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
報酬学習統合スクリプト

DPO/PPOの選択機能、設定ファイルからの読み込み、
既存のSFTモデルからの継続学習、チェックポイント機能を提供
"""

import sys
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Optional
import yaml

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train_reward_learning.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RewardLearningTrainer:
    """報酬学習統合トレーナー"""
    
    def __init__(self, config_path: Path):
        """初期化"""
        self.config_path = config_path
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.reward_config = self.config.get("reward_learning", {})
        self.method = self.reward_config.get("method", "dpo")
        
        logger.info(f"[INIT] RewardLearningTrainer initialized with method: {self.method}")
    
    def train_dpo(self, model_path: str, dataset_path: Path, output_dir: Path, reference_model: Optional[str] = None):
        """DPO学習を実行"""
        logger.info("="*80)
        logger.info("DPO Training")
        logger.info("="*80)
        
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "training" / "train_dpo_reward_learning.py"),
            "--config", str(self.config_path),
            "--model-path", model_path,
            "--dataset", str(dataset_path),
            "--output-dir", str(output_dir)
        ]
        
        if reference_model:
            cmd.extend(["--reference-model", reference_model])
        
        logger.info(f"[CMD] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"[ERROR] DPO training failed: {result.stderr}")
            raise RuntimeError(f"DPO training failed: {result.stderr}")
        
        logger.info(f"[SUCCESS] DPO training completed: {output_dir}")
        return output_dir / "final_model"
    
    def train_ppo(self, model_path: str, dataset_path: Path, output_dir: Path, reward_model: Optional[str] = None):
        """PPO学習を実行"""
        logger.info("="*80)
        logger.info("PPO Training")
        logger.info("="*80)
        
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "training" / "train_ppo_reward_learning.py"),
            "--config", str(self.config_path),
            "--model-path", model_path,
            "--dataset", str(dataset_path),
            "--output-dir", str(output_dir)
        ]
        
        if reward_model:
            cmd.extend(["--reward-model", reward_model])
        
        logger.info(f"[CMD] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"[ERROR] PPO training failed: {result.stderr}")
            raise RuntimeError(f"PPO training failed: {result.stderr}")
        
        logger.info(f"[SUCCESS] PPO training completed: {output_dir}")
        return output_dir / "final_model"
    
    def train_so8t_quadruple_ppo(self, dataset_path: Path, output_dir: Path):
        """SO8T四重推論PPO学習を実行"""
        logger.info("="*80)
        logger.info("SO8T Quadruple PPO Training")
        logger.info("="*80)
        
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "training" / "train_so8t_quadruple_ppo.py"),
            "--config", str(self.config_path),
            "--dataset", str(dataset_path),
            "--output-dir", str(output_dir)
        ]
        
        logger.info(f"[CMD] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"[ERROR] SO8T Quadruple PPO training failed: {result.stderr}")
            raise RuntimeError(f"SO8T Quadruple PPO training failed: {result.stderr}")
        
        logger.info(f"[SUCCESS] SO8T Quadruple PPO training completed: {output_dir}")
        return output_dir / "final_model"
    
    def train(self, model_path: str, dataset_path: Path, output_dir: Path, **kwargs):
        """報酬学習を実行（メソッドに応じて選択）"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.method == "dpo":
            return self.train_dpo(
                model_path=model_path,
                dataset_path=dataset_path,
                output_dir=output_dir,
                reference_model=kwargs.get("reference_model")
            )
        elif self.method == "ppo":
            return self.train_ppo(
                model_path=model_path,
                dataset_path=dataset_path,
                output_dir=output_dir,
                reward_model=kwargs.get("reward_model")
            )
        elif self.method == "so8t_quadruple_ppo":
            return self.train_so8t_quadruple_ppo(
                dataset_path=dataset_path,
                output_dir=output_dir
            )
        else:
            raise ValueError(f"Unknown reward learning method: {self.method}")


def main():
    parser = argparse.ArgumentParser(
        description="Train model with reward learning (DPO/PPO)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Config file path (YAML)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Base model path (SFT済みモデル)"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Pairwise dataset path (JSONL format)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--reference-model",
        type=str,
        default=None,
        help="Reference model path for DPO (optional)"
    )
    parser.add_argument(
        "--reward-model",
        type=str,
        default=None,
        help="Reward model path for PPO (optional)"
    )
    
    args = parser.parse_args()
    
    # 報酬学習トレーナーを初期化
    trainer = RewardLearningTrainer(config_path=args.config)
    
    # 報酬学習を実行
    final_model_dir = trainer.train(
        model_path=args.model_path,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        reference_model=args.reference_model,
        reward_model=args.reward_model
    )
    
    logger.info(f"[COMPLETE] Reward learning completed: {final_model_dir}")


if __name__ == "__main__":
    main()
