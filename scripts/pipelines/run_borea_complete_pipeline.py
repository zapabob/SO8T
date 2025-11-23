#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Borea-Phi-3.5-mini 完全パイプライン実行スクリプト

段階A-Eを統合実行:
- A: ベースGGUF変換
- B: SO8T+PET焼きこみ
- C: 日本語ファインチューニング + 四値分類
- D: HFベンチマーク評価
- E: Q5量子化

Usage:
    python scripts/run_borea_complete_pipeline.py --phase all
    python scripts/run_borea_complete_pipeline.py --phase data_collection
    python scripts/run_borea_complete_pipeline.py --phase finetune
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent


class BoreaPipeline:
    """Borea-Phi-3.5-mini 完全パイプライン"""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.checkpoints_dir = self.project_root / "checkpoints"
        self.eval_results_dir = self.project_root / "eval_results"
        
        # ディレクトリ作成
        for dir_path in [self.data_dir, self.models_dir, self.checkpoints_dir, self.eval_results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Borea Pipeline initialized")
        logger.info(f"  Project root: {self.project_root}")
        logger.info(f"  Data dir: {self.data_dir}")
        logger.info(f"  Models dir: {self.models_dir}")
    
    def run_data_collection(self, target_samples: int = 100000):
        """データ収集フェーズ（合成 + 公開 + Webクロール）"""
        logger.info("="*80)
        logger.info("Phase: Data Collection")
        logger.info("="*80)
        
        # 1. 合成データセット生成
        logger.info("[STEP 1] Generating synthetic dataset...")
        synthetic_script = self.project_root / "scripts" / "generate_synthetic_japanese.py"
        if synthetic_script.exists():
            cmd = [
                sys.executable,
                str(synthetic_script),
                "--output", str(self.data_dir / "synthetic_data.jsonl"),
                "--total_samples", "50000"
            ]
            result = subprocess.run(cmd, cwd=str(self.project_root))
            if result.returncode != 0:
                logger.error(f"[FAILED] Synthetic data generation failed")
                return False
        else:
            logger.warning(f"[WARNING] Synthetic script not found: {synthetic_script}")
        
        # 2. 公開データセット収集 + Webクロール
        logger.info("[STEP 2] Collecting public datasets + web crawling...")
        collect_script = self.project_root / "so8t-mmllm" / "scripts" / "data" / "collect_japanese_data.py"
        if collect_script.exists():
            # 出力ディレクトリはスクリプト内で data/collected に固定されている
            cmd = [
                sys.executable,
                str(collect_script),
                "--target", str(target_samples),
                "--workers", "4",
                "--auto-resume"
            ]
            result = subprocess.run(cmd, cwd=str(self.project_root))
            if result.returncode != 0:
                logger.error(f"[FAILED] Data collection failed")
                return False
        else:
            logger.warning(f"[WARNING] Collection script not found: {collect_script}")
        
        logger.info("[OK] Data collection completed")
        return True
    
    def run_data_cleaning(self):
        """データクレンジングフェーズ"""
        logger.info("="*80)
        logger.info("Phase: Data Cleaning")
        logger.info("="*80)
        
        clean_script = self.project_root / "scripts" / "clean_japanese_dataset.py"
        if not clean_script.exists():
            logger.error(f"[ERROR] Cleaning script not found: {clean_script}")
            return False
        
        # collect_japanese_data.pyは data/collected にドメイン別JSONLファイルを保存
        # クレンジングスクリプトはそのディレクトリ内の全JSONLファイルを処理
        collected_dir = self.project_root / "data" / "collected"
        if not collected_dir.exists():
            logger.error(f"[ERROR] Collected data directory not found: {collected_dir}")
            logger.info("[INFO] Please run data collection first")
            return False
        
        cmd = [
            sys.executable,
            str(clean_script),
            "--input", str(collected_dir),
            "--output", str(self.data_dir / "cleaned")
        ]
        
        result = subprocess.run(cmd, cwd=str(self.project_root))
        if result.returncode != 0:
            logger.error(f"[FAILED] Data cleaning failed")
            return False
        
        logger.info("[OK] Data cleaning completed")
        return True
    
    def run_labeling(self):
        """四値分類ラベル付けフェーズ"""
        logger.info("="*80)
        logger.info("Phase: Four Class Labeling")
        logger.info("="*80)
        
        label_script = self.project_root / "scripts" / "label_four_class_dataset.py"
        if not label_script.exists():
            logger.error(f"[ERROR] Labeling script not found: {label_script}")
            return False
        
        cmd = [
            sys.executable,
            str(label_script),
            "--input", str(self.data_dir / "cleaned"),
            "--output", str(self.data_dir / "labeled")
        ]
        
        result = subprocess.run(cmd, cwd=str(self.project_root))
        if result.returncode != 0:
            logger.error(f"[FAILED] Labeling failed")
            return False
        
        logger.info("[OK] Labeling completed")
        return True
    
    def run_splitting(self):
        """データ分割フェーズ"""
        logger.info("="*80)
        logger.info("Phase: Dataset Splitting")
        logger.info("="*80)
        
        split_script = self.project_root / "scripts" / "split_dataset.py"
        if not split_script.exists():
            logger.error(f"[ERROR] Splitting script not found: {split_script}")
            return False
        
        cmd = [
            sys.executable,
            str(split_script),
            "--input", str(self.data_dir / "labeled"),
            "--output", str(self.data_dir / "splits")
        ]
        
        result = subprocess.run(cmd, cwd=str(self.project_root))
        if result.returncode != 0:
            logger.error(f"[FAILED] Splitting failed")
            return False
        
        logger.info("[OK] Splitting completed")
        return True
    
    def run_finetune_japanese(self):
        """日本語ファインチューニングフェーズ"""
        logger.info("="*80)
        logger.info("Phase: Japanese Finetuning")
        logger.info("="*80)
        
        finetune_script = self.project_root / "scripts" / "finetune_borea_japanese.py"
        if not finetune_script.exists():
            logger.error(f"[ERROR] Finetuning script not found: {finetune_script}")
            return False
        
        config_file = self.project_root / "configs" / "finetune_borea_japanese.yaml"
        if not config_file.exists():
            logger.error(f"[ERROR] Config file not found: {config_file}")
            return False
        
        cmd = [
            sys.executable,
            str(finetune_script),
            "--config", str(config_file),
            "--auto-resume"
        ]
        
        result = subprocess.run(cmd, cwd=str(self.project_root))
        if result.returncode != 0:
            logger.error(f"[FAILED] Finetuning failed")
            return False
        
        logger.info("[OK] Japanese finetuning completed")
        return True
    
    def run_four_class_training(self):
        """四値分類学習フェーズ"""
        logger.info("="*80)
        logger.info("Phase: Four Class Classification Training")
        logger.info("="*80)
        
        train_script = self.project_root / "scripts" / "train_four_class_classifier.py"
        if not train_script.exists():
            logger.error(f"[ERROR] Training script not found: {train_script}")
            return False
        
        config_file = self.project_root / "configs" / "train_four_class.yaml"
        if not config_file.exists():
            logger.error(f"[ERROR] Config file not found: {config_file}")
            return False
        
        cmd = [
            sys.executable,
            str(train_script),
            "--config", str(config_file)
        ]
        
        result = subprocess.run(cmd, cwd=str(self.project_root))
        if result.returncode != 0:
            logger.error(f"[FAILED] Four class training failed")
            return False
        
        logger.info("[OK] Four class training completed")
        return True
    
    def run_evaluation(self):
        """四値分類評価フェーズ"""
        logger.info("="*80)
        logger.info("Phase: Four Class Evaluation")
        logger.info("="*80)
        
        eval_script = self.project_root / "scripts" / "evaluate_four_class.py"
        if not eval_script.exists():
            logger.error(f"[ERROR] Evaluation script not found: {eval_script}")
            return False
        
        model_dir = self.checkpoints_dir / "borea_phi35_mini_four_class" / "final_model"
        test_data = self.data_dir / "splits" / "test.jsonl"
        
        if not model_dir.exists():
            logger.error(f"[ERROR] Model not found: {model_dir}")
            return False
        
        if not test_data.exists():
            logger.error(f"[ERROR] Test data not found: {test_data}")
            return False
        
        cmd = [
            sys.executable,
            str(eval_script),
            "--model", str(model_dir),
            "--test", str(test_data),
            "--output", str(self.eval_results_dir / "four_class_evaluation.json")
        ]
        
        result = subprocess.run(cmd, cwd=str(self.project_root))
        if result.returncode != 0:
            logger.error(f"[FAILED] Evaluation failed")
            return False
        
        logger.info("[OK] Evaluation completed")
        return True
    
    def run_phase(self, phase: str):
        """指定フェーズを実行"""
        phases = {
            "data_collection": self.run_data_collection,
            "data_cleaning": self.run_data_cleaning,
            "labeling": self.run_labeling,
            "splitting": self.run_splitting,
            "finetune": self.run_finetune_japanese,
            "four_class": self.run_four_class_training,
            "evaluation": self.run_evaluation,
        }
        
        if phase not in phases:
            logger.error(f"[ERROR] Unknown phase: {phase}")
            logger.info(f"Available phases: {', '.join(phases.keys())}")
            return False
        
        return phases[phase]()
    
    def run_all(self, target_samples: int = 100000):
        """全フェーズを実行"""
        logger.info("="*80)
        logger.info("Borea-Phi-3.5-mini Complete Pipeline")
        logger.info("="*80)
        
        phases = [
            ("Data Collection", lambda: self.run_data_collection(target_samples)),
            ("Data Cleaning", self.run_data_cleaning),
            ("Labeling", self.run_labeling),
            ("Splitting", self.run_splitting),
            ("Japanese Finetuning", self.run_finetune_japanese),
            ("Four Class Training", self.run_four_class_training),
            ("Evaluation", self.run_evaluation),
        ]
        
        start_time = time.time()
        
        for phase_name, phase_func in phases:
            logger.info(f"\n{'='*80}")
            logger.info(f"Starting: {phase_name}")
            logger.info(f"{'='*80}\n")
            
            if not phase_func():
                logger.error(f"[FAILED] Phase '{phase_name}' failed. Stopping pipeline.")
                return False
            
            logger.info(f"[OK] Phase '{phase_name}' completed")
        
        total_time = time.time() - start_time
        logger.info("="*80)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Total time: {total_time/3600:.2f} hours")
        logger.info("="*80)
        
        # 音声通知
        self._play_audio_notification()
        
        return True
    
    def _play_audio_notification(self):
        """音声通知を再生"""
        audio_file = self.project_root / ".cursor" / "marisa_owattaze.wav"
        if audio_file.exists():
            try:
                ps_cmd = f"""
                if (Test-Path '{audio_file}') {{
                    Add-Type -AssemblyName System.Windows.Forms
                    $player = New-Object System.Media.SoundPlayer '{audio_file}'
                    $player.PlaySync()
                    Write-Host '[OK] Audio notification played' -ForegroundColor Green
                }}
                """
                subprocess.run(
                    ["powershell", "-Command", ps_cmd],
                    cwd=str(self.project_root),
                    check=False
                )
            except Exception as e:
                logger.warning(f"Failed to play audio: {e}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Borea-Phi-3.5-mini Complete Pipeline"
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["all", "data_collection", "data_cleaning", "labeling", "splitting", 
                 "finetune", "four_class", "evaluation"],
        help="Phase to execute"
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        default=100000,
        help="Target number of samples for data collection"
    )
    
    args = parser.parse_args()
    
    pipeline = BoreaPipeline()
    
    try:
        if args.phase == "all":
            success = pipeline.run_all(target_samples=args.target_samples)
        else:
            success = pipeline.run_phase(args.phase)
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.warning("[WARNING] Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"[FAILED] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

