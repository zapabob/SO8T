#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phi-3.5 SO8T /thinking Model Conversion Pipeline

Borea-Phi3.5-instinct-jpをPPO学習で/thinkingモデル化
HFデータセットと既存データセットをPhi-3.5 SO8T/thinkingモデルに変換

パイプラインステップ:
1. HFデータセット収集と初期処理
2. 既存データセット統合
3. Phi-3.5 Thinkingフォーマット変換
4. PPO学習（アルファゲートアニーリング）
5. モデル評価と最適化
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import subprocess
import time
import math
import warnings
warnings.filterwarnings("ignore")

# HuggingFaceキャッシュ設定
os.environ["HF_HOME"] = r"D:\webdataset\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"D:\webdataset\hf_cache\transformers"
os.environ["HF_DATASETS_CACHE"] = r"D:\webdataset\hf_cache\datasets"
os.environ["HF_HUB_CACHE"] = r"D:\webdataset\hf_cache\hub"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/phi35_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Phi35SO8TThinkingPipeline:
    """Phi-3.5 SO8T /thinking Model Conversion Pipeline"""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.project_root = Path(__file__).parent.parent.parent
        self.pipeline_state = self._load_pipeline_state()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # パイプライン固有設定の追加
        config.setdefault('pipeline', {})['name'] = 'phi35_so8t_thinking'
        config['pipeline']['steps'] = [
            'hf_dataset_collection',
            'dataset_integration',
            'phi35_conversion',
            'ppo_training',
            'evaluation'
        ]

        return config

    def _load_pipeline_state(self) -> Dict[str, Any]:
        """パイプライン状態の読み込み"""
        state_file = Path("D:/webdataset/pipeline_state/phi35_pipeline_state.json")
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    return json.load(f)
            except:
                pass

        # 初期状態
        return {
            'current_step': 0,
            'completed_steps': [],
            'last_run': None,
            'checkpoints': {},
            'errors': []
        }

    def _save_pipeline_state(self):
        """パイプライン状態の保存"""
        state_file = Path("D:/webdataset/pipeline_state/phi35_pipeline_state.json")
        state_file.parent.mkdir(parents=True, exist_ok=True)

        self.pipeline_state['last_run'] = datetime.now().isoformat()

        with open(state_file, 'w') as f:
            json.dump(self.pipeline_state, f, indent=2, default=str)

    def run_pipeline(self, resume: bool = False):
        """パイプライン実行"""
        logger.info("="*80)
        logger.info("Phi-3.5 SO8T /thinking Model Conversion Pipeline")
        logger.info("="*80)

        steps = self.config['pipeline']['steps']

        if resume:
            start_step = self.pipeline_state.get('current_step', 0)
            logger.info(f"Resuming from step {start_step}: {steps[start_step]}")
        else:
            start_step = 0
            self.pipeline_state = {'current_step': 0, 'completed_steps': [], 'checkpoints': {}, 'errors': []}

        try:
            for i in range(start_step, len(steps)):
                step_name = steps[i]
                self.pipeline_state['current_step'] = i

                logger.info(f"Step {i+1}/{len(steps)}: {step_name.upper()}")
                logger.info("-" * 50)

                success = getattr(self, f'_run_{step_name}')()

                if success:
                    self.pipeline_state['completed_steps'].append(step_name)
                    self.pipeline_state['checkpoints'][step_name] = datetime.now().isoformat()
                    logger.info(f"✓ Step {step_name} completed successfully")
                else:
                    logger.error(f"✗ Step {step_name} failed")
                    self.pipeline_state['errors'].append({
                        'step': step_name,
                        'timestamp': datetime.now().isoformat(),
                        'error': 'Step execution failed'
                    })
                    break

                self._save_pipeline_state()

            # パイプライン完了
            if len(self.pipeline_state['completed_steps']) == len(steps):
                logger.info("="*80)
                logger.info("🎉 Phi-3.5 SO8T /thinking Model Conversion Pipeline COMPLETED!")
                logger.info("="*80)

                # 最終オーディオ通知
                self._play_audio_notification()

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            import traceback
            traceback.print_exc()

    def _run_hf_dataset_collection(self) -> bool:
        """HFデータセット収集"""
        logger.info("Collecting HF datasets for Phi-3.5 conversion...")

        try:
            # HFデータセット収集スクリプト実行
            cmd = [
                sys.executable,
                "scripts/data/expand_datasets.py",
                "--output", "D:/webdataset/datasets",
                "--max-size", "2000000"  # 2GB
            ]

            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("HF dataset collection completed")
                return True
            else:
                logger.error(f"HF dataset collection failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"HF dataset collection error: {e}")
            return False

    def _run_dataset_integration(self) -> bool:
        """既存データセット統合"""
        logger.info("Integrating existing datasets...")

        try:
            # データセット統合スクリプト実行
            cmd = [
                sys.executable,
                "scripts/data/integrate_hf_datasets.py",
                "--input", "D:/webdataset/datasets",
                "--output", "D:/webdataset/integrated_dataset_full.jsonl"
            ]

            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("Dataset integration completed")
                return True
            else:
                logger.error(f"Dataset integration failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Dataset integration error: {e}")
            return False

    def _run_phi35_conversion(self) -> bool:
        """Phi-3.5 Thinkingフォーマット変換"""
        logger.info("Converting datasets to Phi-3.5 Thinking format...")

        try:
            # Phi-3.5変換スクリプト実行
            cmd = [
                sys.executable,
                "scripts/data/convert_integrated_to_phi35.py",
                "--input", "D:/webdataset/integrated_dataset_full.jsonl",
                "--output", "D:/webdataset/phi35_integrated",
                "--cot-weight", "3.0"
            ]

            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("Phi-3.5 conversion completed")
                return True
            else:
                logger.error(f"Phi-3.5 conversion failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Phi-3.5 conversion error: {e}")
            return False

    def _run_ppo_training(self) -> bool:
        """PPO学習実行（アルファゲートアニーリング）"""
        logger.info("Starting PPO training with alpha gate annealing...")

        try:
            # PPO学習スクリプト実行
            cmd = [
                sys.executable,
                "scripts/training/train_phi35_so8t_ppo_annealing.py",
                "--config", "configs/train_phi35_so8t_annealing.yaml",
                "--output", f"D:/webdataset/checkpoints/training/phi35_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ]

            result = subprocess.run(cmd, cwd=self.project_root)

            if result.returncode == 0:
                logger.info("PPO training completed")
                return True
            else:
                logger.error(f"PPO training failed with code {result.returncode}")
                return False

        except Exception as e:
            logger.error(f"PPO training error: {e}")
            return False

    def _run_evaluation(self) -> bool:
        """モデル評価"""
        logger.info("Evaluating Phi-3.5 SO8T /thinking model...")

        try:
            # 評価スクリプト実行（GGUF変換とベンチマーク）
            cmd = [
                sys.executable,
                "scripts/post_training_workflow.py",
                "--model-path", "D:/webdataset/checkpoints/training/phi35_pipeline_*",
                "--output", "D:/webdataset/gguf_models/phi35_so8t_thinking"
            ]

            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("Model evaluation completed")
                return True
            else:
                logger.error(f"Model evaluation failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Model evaluation error: {e}")
            return False

    def _play_audio_notification(self):
        """オーディオ通知"""
        try:
            subprocess.run([
                "powershell", "-ExecutionPolicy", "Bypass", "-File",
                "scripts/utils/play_audio_notification.ps1"
            ], check=True)
        except:
            pass

    def check_power_resume(self) -> bool:
        """電源投入時の自動再開チェック"""
        logger.info("Checking for incomplete pipeline sessions...")

        # 未完了のパイプラインセッションを検索
        checkpoint_dirs = list(Path("D:/webdataset/checkpoints/training").glob("phi35_pipeline_*"))

        for checkpoint_dir in checkpoint_dirs:
            # 完了チェック（final_modelの存在）
            if not (checkpoint_dir / "final_model").exists():
                logger.info(f"Found incomplete session: {checkpoint_dir.name}")

                # パイプライン状態ファイルの確認
                state_file = checkpoint_dir / "pipeline_state.json"
                if state_file.exists():
                    try:
                        with open(state_file, 'r') as f:
                            state = json.load(f)

                        last_step = state.get('current_step', 0)
                        logger.info(f"Resuming from step {last_step}")

                        # 再開
                        self.pipeline_state = state
                        self.run_pipeline(resume=True)
                        return True

                    except Exception as e:
                        logger.error(f"Error loading pipeline state: {e}")

        logger.info("No incomplete pipeline sessions found")
        return False


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Phi-3.5 SO8T /thinking Model Conversion Pipeline")
    parser.add_argument("--config", type=str, default="configs/train_phi35_so8t_annealing.yaml",
                       help="Configuration file")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from last checkpoint")
    parser.add_argument("--power-resume", action="store_true",
                       help="Check and resume incomplete sessions (for power-on)")

    args = parser.parse_args()

    pipeline = Phi35SO8TThinkingPipeline(args.config)

    if args.power_resume:
        # 電源投入時の自動再開
        if not pipeline.check_power_resume():
            logger.info("Starting new Phi-3.5 pipeline...")
            pipeline.run_pipeline(resume=False)
    elif args.resume:
        # 手動再開
        pipeline.run_pipeline(resume=True)
    else:
        # 新規実行
        pipeline.run_pipeline(resume=False)


if __name__ == "__main__":
    main()
