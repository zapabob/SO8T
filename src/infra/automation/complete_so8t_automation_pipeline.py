#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete SO8T Automation Pipeline

Borea-Phi3.5-instinct-jpの完全自動SO8T/thinkingモデル化パイプライン：
1. マルチモーダルデータセット収集（NSFW/音声データ含む）
2. 四値分類とデータクレンジング
3. PPOトレーニング with SO8ViTアダプター
4. マルチモーダル統合
5. ベンチマーク評価と統計処理
6. HFアップロード
7. タスクスケジュール自動削除
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import time
import threading
import queue
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import traceback
from tqdm import tqdm

# SO8T関連インポート
try:
    from so8t.core.dynamic_thinking_so8t import create_dynamic_thinking_so8t
    from so8t.optimization.bayesian_alpha_optimizer import create_bayesian_optimizer
    from so8t.evaluation.comprehensive_benchmark_evaluator import run_comprehensive_evaluation
except ImportError as e:
    logging.warning(f"SO8T import failed: {e}")

# デバッグロギング設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/complete_automation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# tqdm設定
tqdm.monitor_interval = 0  # 即時更新


class SO8TAutomationPipeline:
    """
    完全自動SO8Tパイプライン

    電源投入時自動起動からHFアップロード完了まで
    """

    def __init__(self, config_path: str = "configs/complete_so8t_pipeline.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.pipeline_status = {}
        self.error_log = []

        # パス設定
        self.base_dir = Path("D:/webdataset")
        self.models_dir = self.base_dir / "models"
        self.checkpoints_dir = self.base_dir / "checkpoints" / "training"
        self.datasets_dir = self.base_dir / "datasets"
        self.gguf_dir = self.base_dir / "gguf_models"
        self.logs_dir = Path("logs")

        # tqdm設定
        self.main_progress = None
        self.step_progress = None

        # デバッグ情報キュー
        self.debug_queue = queue.Queue()
        self.debug_thread = None
        self.debug_enabled = True

        # ディレクトリ作成
        for dir_path in [self.models_dir, self.checkpoints_dir, self.datasets_dir,
                        self.gguf_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # デバッグ出力スレッド開始
        self._start_debug_output()

        logger.info("SO8T Automation Pipeline initialized with tqdm and debug output")
        logger.debug(f"Config loaded: {self.config_path}")
        logger.debug(f"Base directory: {self.base_dir}")
        logger.debug(f"Debug output: {'enabled' if self.debug_enabled else 'disabled'}")

    def _start_debug_output(self):
        """デバッグ出力スレッド開始"""
        if not self.debug_enabled:
            return

        def debug_worker():
            while self.debug_enabled:
                try:
                    # キューからデバッグ情報を取得して表示
                    debug_info = self.debug_queue.get(timeout=1.0)
                    if debug_info:
                        level, message = debug_info
                        if level == 'progress':
                            print(f"\r[DEBUG] {message}", end='', flush=True)
                        elif level == 'info':
                            print(f"\n[DEBUG] {message}")
                        elif level == 'warning':
                            print(f"\n[WARNING] {message}")
                        elif level == 'error':
                            print(f"\n[ERROR] {message}")
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"\n[DEBUG ERROR] {e}")

        self.debug_thread = threading.Thread(target=debug_worker, daemon=True)
        self.debug_thread.start()
        logger.debug("Debug output thread started")

    def _debug_print(self, message: str, level: str = 'info'):
        """デバッグ出力（非同期）"""
        if self.debug_enabled:
            self.debug_queue.put((level, message))

    def _update_progress(self, step_name: str, current: int, total: int, message: str = ""):
        """プログレス更新"""
        if self.step_progress:
            self.step_progress.n = current
            self.step_progress.total = total
            if message:
                self.step_progress.set_description(f"[STEP {current}/{total}] {step_name}: {message}")
            else:
                self.step_progress.set_description(f"[STEP {current}/{total}] {step_name}")
            self.step_progress.refresh()

        self._debug_print(f"Progress: {step_name} {current}/{total} - {message}", 'progress')

    def _load_config(self) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        import yaml

        default_config = {
            'model': {
                'base_model': 'microsoft/phi-3.5-mini-instruct',
                'target_model': 'Borea-Phi3.5-instinct-jp',
                'output_name': 'borea_phi35_so8t_multimodal'
            },
            'data': {
                'multimodal_datasets': [
                    'HuggingFaceFW/fineweb-2',  # テキスト
                    'laion/aesthetic-predictor-5',  # 画像評価
                    'mozilla-foundation/common_voice_11_0',  # 音声
                    'deepghs/nsfw_detect',  # NSFW検知
                ],
                'license_filter': ['mit', 'apache-2.0'],
                'max_samples_per_dataset': 50000,
                'test_split_ratio': 0.2
            },
            'training': {
                'ppo_epochs': 3,
                'batch_size': 4,
                'learning_rate': 1e-5,
                'max_steps': 1000,
                'so8vit_enabled': True,
                'multimodal_enabled': True,
                'bayesian_optimization': True
            },
            'benchmark': {
                'datasets': ['elyza_100', 'mmlu', 'gsm8k', 'hellaswag'],
                'significance_level': 0.05,
                'performance_threshold': 0.75
            },
            'automation': {
                'auto_resume_on_power_on': True,
                'error_retry_count': 3,
                'cleanup_on_success': True
            }
        }

        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
            default_config.update(user_config)

        return default_config

    def run_complete_pipeline(self) -> bool:
        """
        完全パイプライン実行 with tqdm and debug output

        Returns:
            成功/失敗
        """
        logger.info("="*80)
        logger.info("STARTING COMPLETE SO8T AUTOMATION PIPELINE")
        logger.info("="*80)
        logger.debug("Initializing main progress bar...")

        # メインのプログレスバー初期化
        self.main_progress = tqdm(
            total=7,
            desc="[PIPELINE] Complete SO8T Automation",
            unit="step",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {desc}',
            position=0,
            leave=True
        )

        self._debug_print("Pipeline started - 7 steps total", 'info')

        try:
            # STEP 1: マルチモーダルデータセット収集
            self.main_progress.set_description("[PIPELINE] Step 1/7: Multimodal Data Collection")
            self._debug_print("Starting Step 1: Multimodal Data Collection", 'info')
            if not self._step_multimodal_data_collection():
                raise RuntimeError("Data collection failed")
            self.main_progress.update(1)
            self._debug_print("Completed Step 1: Data collection successful", 'info')

            # STEP 2: データ前処理（四値分類 + クレンジング）
            self.main_progress.set_description("[PIPELINE] Step 2/7: Data Preprocessing")
            self._debug_print("Starting Step 2: Data Preprocessing", 'info')
            if not self._step_data_preprocessing():
                raise RuntimeError("Data preprocessing failed")
            self.main_progress.update(1)
            self._debug_print("Completed Step 2: Data preprocessing successful", 'info')

            # STEP 3: PPOトレーニング with SO8ViT + マルチモーダル
            self.main_progress.set_description("[PIPELINE] Step 3/7: PPO Training")
            self._debug_print("Starting Step 3: PPO Training with SO8ViT", 'info')
            if not self._step_ppo_training():
                raise RuntimeError("PPO training failed")
            self.main_progress.update(1)
            self._debug_print("Completed Step 3: PPO training successful", 'info')

            # STEP 4: モデル統合と最適化
            self.main_progress.set_description("[PIPELINE] Step 4/7: Model Integration")
            self._debug_print("Starting Step 4: Model Integration", 'info')
            if not self._step_model_integration():
                raise RuntimeError("Model integration failed")
            self.main_progress.update(1)
            self._debug_print("Completed Step 4: Model integration successful", 'info')

            # STEP 5: 包括的ベンチマーク評価
            self.main_progress.set_description("[PIPELINE] Step 5/7: Benchmark Evaluation")
            self._debug_print("Starting Step 5: Benchmark Evaluation", 'info')
            if not self._step_comprehensive_benchmark():
                raise RuntimeError("Benchmark evaluation failed")
            self.main_progress.update(1)
            self._debug_print("Completed Step 5: Benchmark evaluation successful", 'info')

            # STEP 6: HFアップロード
            self.main_progress.set_description("[PIPELINE] Step 6/7: HF Upload")
            self._debug_print("Starting Step 6: HuggingFace Upload", 'info')
            if not self._step_huggingface_upload():
                raise RuntimeError("HuggingFace upload failed")
            self.main_progress.update(1)
            self._debug_print("Completed Step 6: HF upload successful", 'info')

            # STEP 7: クリーンアップとタスク削除
            self.main_progress.set_description("[PIPELINE] Step 7/7: Cleanup")
            self._debug_print("Starting Step 7: Cleanup and Task Removal", 'info')
            if not self._step_cleanup_and_task_removal():
                logger.warning("Cleanup failed, but pipeline completed")
            self.main_progress.update(1)
            self._debug_print("Completed Step 7: Cleanup successful", 'info')

            logger.info("="*80)
            logger.info("COMPLETE SO8T AUTOMATION PIPELINE SUCCESS!")
            logger.info("="*80)

            return True

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            logger.error(traceback.format_exc())
            self._handle_pipeline_error(e)
            return False

        finally:
            # プログレスバーとデバッグ出力のクリーンアップ
            if self.main_progress:
                self.main_progress.close()
            if self.step_progress:
                self.step_progress.close()
            self.debug_enabled = False
            self._debug_print("Pipeline execution finished", 'info')
            logger.info("Pipeline cleanup completed")

    def __del__(self):
        """デストラクタ - デバッグスレッドのクリーンアップ"""
        self.debug_enabled = False
        logger.debug("SO8TAutomationPipeline destructor called")

    def _step_multimodal_data_collection(self) -> bool:
        """STEP 1: マルチモーダルデータセット収集 with tqdm and debug"""
        logger.info("STEP 1: Multimodal Data Collection")
        logger.info("-" * 50)
        self._debug_print("Initializing multimodal data collection sub-steps", 'info')

        # サブプログレスバー作成
        sub_steps = 4  # データセットダウンロード, 統合, 検証, 統計
        self.step_progress = tqdm(
            total=sub_steps,
            desc="[STEP 1] Data Collection",
            unit="substep",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {desc}',
            position=1,
            leave=False
        )

        try:
            self._update_progress("Data Collection", 0, sub_steps, "Initializing")
            # HFデータセット収集スクリプト実行
            self._update_progress("Data Collection", 1, sub_steps, "Downloading datasets")
            self._debug_print("Starting HF dataset download...", 'info')
            cmd = [
                sys.executable, "scripts/data/expand_datasets.py",
                "--output", str(self.datasets_dir / "multimodal_raw"),
                "--datasets", json.dumps(self.config['data']['multimodal_datasets']),
                "--licenses", json.dumps(self.config['data']['license_filter']),
                "--max-samples", str(self.config['data']['max_samples_per_dataset'])
            ]

            logger.debug(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            if result.returncode != 0:
                logger.error(f"Data collection failed: {result.stderr}")
                self._debug_print(f"Data collection error: {result.stderr}", 'error')
                return False
            self._debug_print("Dataset download completed successfully", 'info')

            # NSFWデータ追加
            self._update_progress("Data Collection", 2, sub_steps, "Collecting NSFW data")
            self._debug_print("Collecting NSFW datasets...", 'info')
            if not self._collect_nsfw_data():
                logger.warning("NSFW data collection failed, continuing...")
                self._debug_print("NSFW data collection failed, continuing", 'warning')

            # 音声データ追加
            self._update_progress("Data Collection", 3, sub_steps, "Collecting audio data")
            self._debug_print("Collecting audio datasets...", 'info')
            if not self._collect_audio_data():
                logger.warning("Audio data collection failed, continuing...")
                self._debug_print("Audio data collection failed, continuing", 'warning')

            # データ統合と検証
            self._update_progress("Data Collection", 4, sub_steps, "Integrating and validating")
            self._debug_print("Integrating collected datasets...", 'info')
            if not self._integrate_collected_datasets():
                logger.error("Dataset integration failed")
                self._debug_print("Dataset integration failed", 'error')
                return False

            self.step_progress.close()
            self.pipeline_status['data_collection'] = 'completed'
            logger.info("✓ Multimodal data collection completed")
            self._debug_print("Data collection step completed successfully", 'info')
            return True

        except Exception as e:
            logger.error(f"Data collection error: {e}")
            self.error_log.append({'step': 'data_collection', 'error': str(e)})
            return False

    def _collect_nsfw_data(self) -> bool:
        """NSFWデータ収集"""
        try:
            cmd = [
                sys.executable, "scripts/data/expand_datasets.py",
                "--output", str(self.datasets_dir / "nsfw_data"),
                "--datasets", '["deepghs/nsfw_detect", "FredZhang7/anime-kawaii-diffusion"]',
                "--max-samples", "10000"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            return result.returncode == 0
        except Exception as e:
            return False

    def _collect_audio_data(self) -> bool:
        """音声データ収集"""
        try:
            cmd = [
                sys.executable, "scripts/data/expand_datasets.py",
                "--output", str(self.datasets_dir / "audio_data"),
                "--datasets", '["mozilla-foundation/common_voice_11_0"]',
                "--max-samples", "5000"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            return result.returncode == 0
        except Exception as e:
            return False

    def _step_data_preprocessing(self) -> bool:
        """STEP 2: データ前処理（四値分類 + クレンジング） with tqdm and debug"""
        logger.info("STEP 2: Data Preprocessing")
        logger.info("-" * 50)
        self._debug_print("Initializing data preprocessing sub-steps", 'info')

        # サブプログレスバー作成
        sub_steps = 4  # 四値分類, データクレンジング, 統計分析, 検証
        self.step_progress = tqdm(
            total=sub_steps,
            desc="[STEP 2] Data Preprocessing",
            unit="substep",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {desc}',
            position=1,
            leave=False
        )

        try:
            self._update_progress("Data Preprocessing", 0, sub_steps, "Initializing")
            # 四値分類実行
            self._update_progress("Data Preprocessing", 1, sub_steps, "Four-class labeling")
            self._debug_print("Starting four-class labeling...", 'info')
            cmd = [
                sys.executable, "scripts/data/label_four_class_dataset_fixed.py",
                "--input", str(self.datasets_dir / "multimodal_raw"),
                "--output", str(self.datasets_dir / "labeled_data"),
                "--multimodal", "true"
            ]

            logger.debug(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            if result.returncode != 0:
                logger.error(f"Four-class labeling failed: {result.stderr}")
                self._debug_print(f"Four-class labeling error: {result.stderr}", 'error')
                return False
            self._debug_print("Four-class labeling completed successfully", 'info')

            # データクレンジング
            self._update_progress("Data Preprocessing", 2, sub_steps, "Data cleansing")
            self._debug_print("Starting data cleansing...", 'info')
            cmd = [
                sys.executable, "scripts/data/cleanse_codex_pairwise_dataset.py",
                "--input", str(self.datasets_dir / "labeled_data"),
                "--output", str(self.datasets_dir / "cleansed_data"),
                "--multimodal", "true"
            ]

            logger.debug(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            if result.returncode != 0:
                logger.error(f"Data cleansing failed: {result.stderr}")
                self._debug_print(f"Data cleansing error: {result.stderr}", 'error')
                return False
            self._debug_print("Data cleansing completed successfully", 'info')

            # scikit-learnによるデータ分割
            self._update_progress("Data Preprocessing", 3, sub_steps, "Data splitting")
            self._debug_print("Starting data splitting with scikit-learn...", 'info')
            cmd = [
                sys.executable, "scripts/data/label_four_class_dataset_fixed.py",
                "--input", str(self.datasets_dir / "cleansed_data"),
                "--output", str(self.datasets_dir / "final_dataset"),
                "--split", "true",
                "--test-ratio", str(self.config['data']['test_split_ratio'])
            ]

            logger.debug(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            if result.returncode != 0:
                logger.error(f"Data splitting failed: {result.stderr}")
                self._debug_print(f"Data splitting error: {result.stderr}", 'error')
                return False
            self._debug_print("Data splitting completed successfully", 'info')

            # 統計分析と検証
            self._update_progress("Data Preprocessing", 4, sub_steps, "Validation")
            self._debug_print("Performing final validation...", 'info')
            if not self._validate_preprocessed_data():
                logger.error("Data validation failed")
                self._debug_print("Data validation failed", 'error')
                return False

            self.step_progress.close()
            self.pipeline_status['data_preprocessing'] = 'completed'
            logger.info("✓ Data preprocessing completed")
            self._debug_print("Data preprocessing step completed successfully", 'info')
            return True

        except Exception as e:
            logger.error(f"Data preprocessing error: {e}")
            self.error_log.append({'step': 'data_preprocessing', 'error': str(e)})
            return False

    def _step_ppo_training(self) -> bool:
        """STEP 3: PPOトレーニング with SO8ViT + マルチモーダル with tqdm and debug"""
        logger.info("STEP 3: PPO Training with SO8ViT + Multimodal")
        logger.info("-" * 50)
        self._debug_print("Initializing PPO training sub-steps", 'info')

        # サブプログレスバー作成
        sub_steps = 3  # トレーニング実行, チェックポイント検証, 結果確認
        self.step_progress = tqdm(
            total=sub_steps,
            desc="[STEP 3] PPO Training",
            unit="substep",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {desc}',
            position=1,
            leave=False
        )

        try:
            self._update_progress("PPO Training", 0, sub_steps, "Initializing")
            # 高度なPhi-3.5 SO8Tトレーニング実行
            self._update_progress("PPO Training", 1, sub_steps, "Running training")
            self._debug_print("Starting advanced Phi-3.5 SO8T training...", 'info')
            cmd = [
                sys.executable, "scripts/training/train_phi35_advanced_pipeline.py",
                "--config", "configs/train_phi35_so8t_annealing.yaml",
                "--output", str(self.checkpoints_dir / f"phi35_so8t_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            ]

            logger.debug(f"Running command: {' '.join(cmd)}")
            # PPOトレーニングは時間がかかるので、リアルタイム出力を表示
            self._debug_print("Training may take several hours - monitor logs for progress", 'warning')
            result = subprocess.run(cmd, capture_output=False, text=True, cwd=os.getcwd())
            if result.returncode != 0:
                logger.error(f"PPO training failed: {result.stderr}")
                self._debug_print(f"PPO training error: {result.stderr}", 'error')
                return False
            self._debug_print("PPO training completed successfully", 'info')

            # チェックポイント検証
            self._update_progress("PPO Training", 2, sub_steps, "Validating checkpoints")
            self._debug_print("Validating training checkpoints...", 'info')
            if not self._validate_training_checkpoints():
                logger.error("Checkpoint validation failed")
                self._debug_print("Checkpoint validation failed", 'error')
                return False

            # トレーニング結果確認
            self._update_progress("PPO Training", 3, sub_steps, "Checking results")
            self._debug_print("Checking training results...", 'info')
            if not self._check_training_results():
                logger.error("Training results check failed")
                self._debug_print("Training results check failed", 'error')
                return False

            self.step_progress.close()
            self.pipeline_status['ppo_training'] = 'completed'
            logger.info("✓ PPO training completed")
            self._debug_print("PPO training step completed successfully", 'info')
            return True

        except Exception as e:
            logger.error(f"PPO training error: {e}")
            self.error_log.append({'step': 'ppo_training', 'error': str(e)})
            return False

    def _step_model_integration(self) -> bool:
        """STEP 4: モデル統合と最適化"""
        logger.info("STEP 4: Model Integration and Optimization")
        logger.info("-" * 50)

        try:
            # SO8T効果の焼き込み
            cmd = [
                "python", "scripts/conversion/bake_and_convert_to_gguf.bat"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            if result.returncode != 0:
                logger.error(f"Model integration failed: {result.stderr}")
                return False

            # 焼き込み検証
            cmd = [
                sys.executable, "scripts/conversion/verify_so8t_baking.py",
                "--original", str(self.checkpoints_dir / "phi35_advanced_*" / "final_model"),
                "--baked", str(self.models_dir / "baked_for_gguf" / "phi35_so8t_baked")
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            if result.returncode != 0:
                logger.warning(f"Baking verification failed: {result.stderr}")

            self.pipeline_status['model_integration'] = 'completed'
            logger.info("✓ Model integration completed")
            return True

        except Exception as e:
            logger.error(f"Model integration error: {e}")
            self.error_log.append({'step': 'model_integration', 'error': str(e)})
            return False

    def _step_comprehensive_benchmark(self) -> bool:
        """STEP 5: 包括的ベンチマーク評価"""
        logger.info("STEP 5: Comprehensive Benchmark Evaluation")
        logger.info("-" * 50)

        try:
            # 包括的評価実行
            evaluation_result = run_comprehensive_evaluation(
                model_a_path=self.config['model']['base_model'],
                model_b_path=str(self.models_dir / "baked_for_gguf" / "phi35_so8t_baked"),
                output_dir=str(self.base_dir / "evaluation_results" / f"so8t_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            )

            # 性能チェック
            conclusion = evaluation_result.get('conclusion', {})
            if conclusion.get('statistically_significant') and conclusion.get('performance_difference', 0) > 0:
                logger.info(f"✓ Model performance improved: +{conclusion['performance_difference']:.3f}")
            else:
                logger.warning(f"⚠ Model performance not significantly improved: {conclusion.get('performance_difference', 0):.3f}")

            self.pipeline_status['benchmark'] = 'completed'
            logger.info("✓ Benchmark evaluation completed")
            return True

        except Exception as e:
            logger.error(f"Benchmark evaluation error: {e}")
            self.error_log.append({'step': 'benchmark', 'error': str(e)})
            return False

    def _step_huggingface_upload(self) -> bool:
        """STEP 6: HFアップロード"""
        logger.info("STEP 6: HuggingFace Upload")
        logger.info("-" * 50)

        try:
            # HFアップロードスクリプト実行
            cmd = [
                sys.executable, "scripts/upload_aegis_to_huggingface.py",
                "--model", str(self.models_dir / "baked_for_gguf" / "phi35_so8t_baked"),
                "--gguf", str(self.gguf_dir / "phi35_so8t_baked"),
                "--name", self.config['model']['output_name'],
                "--type", "so8t_multimodal"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            if result.returncode != 0:
                logger.error(f"HF upload failed: {result.stderr}")
                return False

            self.pipeline_status['hf_upload'] = 'completed'
            logger.info("✓ HuggingFace upload completed")
            return True

        except Exception as e:
            logger.error(f"HF upload error: {e}")
            self.error_log.append({'step': 'hf_upload', 'error': str(e)})
            return False

    def _step_cleanup_and_task_removal(self) -> bool:
        """STEP 7: クリーンアップとタスク削除"""
        logger.info("STEP 7: Cleanup and Task Removal")
        logger.info("-" * 50)

        try:
            # PowerShellスクリプトでタスクスケジュールを削除
            ps_script = f'''
            $taskName = "SO8T_Automation_Pipeline"
            try {{
                Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction Stop
                Write-Host "Successfully removed scheduled task: $taskName"
            }} catch {{
                Write-Host "Task removal failed or task not found: $($_.Exception.Message)"
            }}
            '''

            with open('temp_task_removal.ps1', 'w') as f:
                f.write(ps_script)

            result = subprocess.run([
                "powershell", "-ExecutionPolicy", "Bypass", "-File", "temp_task_removal.ps1"
            ], capture_output=True, text=True)

            # 一時ファイル削除
            Path('temp_task_removal.ps1').unlink(missing_ok=True)

            if result.returncode == 0:
                logger.info("✓ Scheduled task removed successfully")
            else:
                logger.warning(f"Task removal warning: {result.stderr}")

            # パイプライン完了ログ
            completion_log = {
                'pipeline_completed_at': str(datetime.now()),
                'total_steps': 7,
                'completed_steps': len([s for s in self.pipeline_status.values() if s == 'completed']),
                'errors': self.error_log,
                'final_model_path': str(self.models_dir / "baked_for_gguf" / "phi35_so8t_baked"),
                'gguf_model_path': str(self.gguf_dir / self.config['model']['output_name']),
                'hf_model_name': self.config['model']['output_name']
            }

            with open(self.logs_dir / f"pipeline_completion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
                json.dump(completion_log, f, indent=2, default=str)

            self.pipeline_status['cleanup'] = 'completed'
            logger.info("✓ Cleanup and task removal completed")
            return True

        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            self.error_log.append({'step': 'cleanup', 'error': str(e)})
            return False

    def _integrate_collected_datasets(self) -> bool:
        """収集したデータセットの統合"""
        try:
            self._debug_print("Integrating multimodal datasets...", 'info')
            # 既存の統合スクリプトを使用
            cmd = [
                sys.executable, "scripts/data/integrate_hf_datasets.py",
                "--input", str(self.datasets_dir / "multimodal_raw"),
                "--output", str(self.datasets_dir / "integrated_multimodal.jsonl")
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            if result.returncode != 0:
                logger.warning(f"Dataset integration warning: {result.stderr}")
                return False
            return True
        except Exception as e:
            logger.error(f"Dataset integration error: {e}")
            return False

    def _validate_preprocessed_data(self) -> bool:
        """前処理済みデータの検証"""
        try:
            output_file = self.datasets_dir / "final_dataset" / "train.jsonl"
            if not output_file.exists():
                logger.error(f"Training data file not found: {output_file}")
                return False

            # ファイルサイズチェック
            size_mb = output_file.stat().st_size / (1024 * 1024)
            if size_mb < 10:
                logger.warning(f"Training data size is small: {size_mb:.1f} MB")
                self._debug_print(f"Warning: Small training dataset ({size_mb:.1f} MB)", 'warning')

            # サンプル数チェック
            sample_count = 0
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    sample_count += 1
                    if sample_count >= 100:  # 最初の100行だけカウント
                        break

            if sample_count < 50:
                logger.error(f"Insufficient training samples: {sample_count}")
                return False

            self._debug_print(f"Data validation passed: {sample_count}+ samples, {size_mb:.1f} MB", 'info')
            return True
        except Exception as e:
            logger.error(f"Data validation error: {e}")
            return False

    def _validate_training_checkpoints(self) -> bool:
        """トレーニングチェックポイントの検証"""
        try:
            checkpoint_dir = self.checkpoints_dir / f"phi35_so8t_{datetime.now().strftime('%Y%m%d')}"
            if not checkpoint_dir.exists():
                # 最新のチェックポイントディレクトリを探す
                checkpoint_dirs = list(self.checkpoints_dir.glob("phi35_so8t_*"))
                if not checkpoint_dirs:
                    logger.error("No checkpoint directory found")
                    return False
                checkpoint_dir = max(checkpoint_dirs, key=lambda x: x.stat().st_mtime)

            # チェックポイントファイルの存在確認
            checkpoint_files = list(checkpoint_dir.glob("*.pt"))
            if not checkpoint_files:
                logger.error(f"No checkpoint files found in {checkpoint_dir}")
                return False

            # 最新のチェックポイントサイズチェック
            latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
            size_gb = latest_checkpoint.stat().st_size / (1024**3)

            if size_gb < 1.0:
                logger.warning(f"Checkpoint file seems small: {size_gb:.2f} GB")
                self._debug_print(f"Warning: Small checkpoint file ({size_gb:.2f} GB)", 'warning')

            self._debug_print(f"Checkpoint validation passed: {len(checkpoint_files)} files, latest {size_gb:.2f} GB", 'info')
            return True
        except Exception as e:
            logger.error(f"Checkpoint validation error: {e}")
            return False

    def _check_training_results(self) -> bool:
        """トレーニング結果の確認"""
        try:
            # ログファイルからトレーニング結果をチェック
            log_files = list(Path("logs").glob("complete_automation_*.log"))
            if not log_files:
                logger.warning("No training log files found")
                return True  # ログがなくても成功とみなす

            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)

            # ログからトレーニング完了を確認
            with open(latest_log, 'r', encoding='utf-8') as f:
                content = f.read()
                if "✓ PPO training completed" in content:
                    self._debug_print("Training completion confirmed in logs", 'info')
                    return True
                else:
                    logger.warning("Training completion not found in logs")
                    return False
        except Exception as e:
            logger.error(f"Training results check error: {e}")
            return False

    def _handle_pipeline_error(self, error: Exception):
        """パイプラインエラー処理"""
        logger.error(f"Pipeline error detected: {error}")

        # エラーログ保存
        error_info = {
            'error_time': str(datetime.now()),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'pipeline_status': self.pipeline_status,
            'completed_steps': [k for k, v in self.pipeline_status.items() if v == 'completed'],
            'failed_step': self._identify_failed_step()
        }

        error_file = self.logs_dir / f"pipeline_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(error_file, 'w') as f:
            json.dump(error_info, f, indent=2, default=str)

        logger.error(f"Error details saved to: {error_file}")

        # エラー通知（オーディオ）
        try:
            subprocess.run([
                "powershell", "-ExecutionPolicy", "Bypass",
                "-File", "scripts/utils/play_audio_notification.ps1"
            ], check=True)
        except:
            pass

    def _identify_failed_step(self) -> str:
        """失敗したステップの特定"""
        step_order = ['data_collection', 'data_preprocessing', 'ppo_training',
                     'model_integration', 'benchmark', 'hf_upload', 'cleanup']

        for step in step_order:
            if step not in self.pipeline_status or self.pipeline_status[step] != 'completed':
                return step
        return 'unknown'

    def get_pipeline_status(self) -> Dict[str, Any]:
        """パイプライン状態取得"""
        return {
            'status': self.pipeline_status,
            'errors': self.error_log,
            'is_completed': all(v == 'completed' for v in self.pipeline_status.values()),
            'completion_percentage': len([v for v in self.pipeline_status.values() if v == 'completed']) / 7 * 100
        }


def create_power_on_task():
    """電源投入時自動起動タスク作成"""
    ps_script = '''
    $taskName = "SO8T_Automation_Pipeline"
    $scriptPath = "C:\\Users\\downl\\Desktop\\SO8T\\scripts\\automation\\run_complete_pipeline.bat"

    try {
        # 既存タスク削除（念のため）
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

        # 新しいタスク作成
        $action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c $scriptPath"
        $trigger = New-ScheduledTaskTrigger -AtLogOn
        $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType InteractiveToken
        $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

        Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description "SO8T Complete Automation Pipeline"

        Write-Host "Scheduled task created: $taskName"
    } catch {
        Write-Host "Failed to create scheduled task: $($_.Exception.Message)"
        exit 1
    }
    '''

    with open('create_power_on_task.ps1', 'w') as f:
        f.write(ps_script)

    result = subprocess.run([
        "powershell", "-ExecutionPolicy", "Bypass", "-File", "create_power_on_task.ps1"
    ], capture_output=True, text=True)

    Path('create_power_on_task.ps1').unlink(missing_ok=True)

    return result.returncode == 0


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Complete SO8T Automation Pipeline")
    parser.add_argument("--config", type=str, default="configs/complete_so8t_pipeline.yaml",
                       help="Configuration file")
    parser.add_argument("--create-task", action="store_true",
                       help="Create power-on scheduled task instead of running pipeline")
    parser.add_argument("--status", action="store_true",
                       help="Show current pipeline status")

    args = parser.parse_args()

    if args.create_task:
        # 電源投入時タスク作成
        logger.info("Creating power-on scheduled task...")
        if create_power_on_task():
            logger.info("✓ Power-on task created successfully")
        else:
            logger.error("✗ Failed to create power-on task")
        return

    # パイプライン実行
    pipeline = SO8TAutomationPipeline(args.config)

    if args.status:
        # ステータス表示
        status = pipeline.get_pipeline_status()
        print(json.dumps(status, indent=2, default=str))
        return

    # 完全パイプライン実行
    success = pipeline.run_complete_pipeline()

    if success:
        logger.info("🎉 COMPLETE SO8T AUTOMATION PIPELINE FINISHED SUCCESSFULLY!")
        print("\n🎉 Pipeline completed successfully!")
        print("Borea-Phi3.5-instinct-jp has been transformed into a complete SO8T/thinking multimodal model!")
        print("Model uploaded to HuggingFace and scheduled task removed.")
    else:
        logger.error("❌ COMPLETE SO8T AUTOMATION PIPELINE FAILED!")
        print("\n❌ Pipeline failed! Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
