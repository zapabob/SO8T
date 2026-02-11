#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete SO8T Automation Pipeline for RTX3060
Frozen base weights, QLoRA fine-tuning enabled

Borea-Phi3.5-instinct-jpの完全自動SO8T/thinkingモデル化パイプライン（RTX3060対応）：
1. マルチモーダルデータセット収集（NSFW/音声データ含む）
2. 四値分類とデータクレンジング
3. QLoRAファインチューニング with SO8ViTアダプター（重み凍結）
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
        logging.FileHandler(f'logs/complete_automation_rtx3060_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# tqdm設定
tqdm.monitor_interval = 0  # 即時更新


class RTX3060SO8TAutomationPipeline:
    """
    RTX3060対応SO8T完全自動パイプライン
    元の重みを凍結しつつQLoRAでファインチューニング
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

        # RTX3060固有設定
        self.rtx3060_config = {
            'max_memory_gb': 8.0,
            'batch_size': 1,
            'gradient_accumulation': 4,
            'use_qlora': True,
            'freeze_base_weights': True
        }

        # ディレクトリ作成
        for dir_path in [self.models_dir, self.checkpoints_dir, self.datasets_dir,
                        self.gguf_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # デバッグ出力スレッド開始
        self._start_debug_output()

        logger.info("RTX3060 SO8T Automation Pipeline initialized with QLoRA and frozen weights")
        logger.debug(f"Config loaded: {self.config_path}")
        logger.debug(f"RTX3060 memory limit: {self.rtx3060_config['max_memory_gb']}GB")

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
        """設定ファイル読み込み（RTX3060最適化済み）"""
        import yaml

        default_config = {
            'model': {
                'base_model': 'microsoft/phi-3.5-mini-instruct',
                'target_model': 'Borea-Phi3.5-instinct-jp',
                'output_name': 'borea_phi35_AEGIS2.0',
                'use_qlora': True,
                'freeze_base_weights': True
            },
            'data': {
                'multimodal_datasets': [
                    "HuggingFaceFW/fineweb-2",
                    "laion/aesthetic-predictor-5",
                    "mozilla-foundation/common_voice_11_0",
                    "deepghs/nsfw_detect",
                    "FredZhang7/anime-kawaii-diffusion",
                    "openai/gsm8k"
                ],
                'max_samples_per_dataset': 50000,
                'test_split_ratio': 0.2
            },
            'training': {
                'ppo_epochs': 3,
                'max_steps': 1000,
                'learning_rate': 1e-5,
                'batch_size': 1,  # RTX3060 optimized
                'gradient_accumulation_steps': 4,  # RTX3060 optimized
                'max_length': 2048,
                'use_gradient_checkpointing': True,
                'use_mixed_precision': True,
                'so8vit_enabled': True,
                'dynamic_thinking_enabled': True,
                'bayesian_optimization': True
            },
            'resources': {
                'required_gpu_memory_gb': 8,  # RTX3060
                'required_disk_space_gb': 200
            }
        }

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            # RTX3060設定で上書き
            config['resources']['required_gpu_memory_gb'] = 8
            config['training']['batch_size'] = 1
            config['training']['gradient_accumulation_steps'] = 4
            config['training']['use_gradient_checkpointing'] = True
            config['training']['use_mixed_precision'] = True
            config['model']['use_qlora'] = True
            config['model']['freeze_base_weights'] = True
            return config
        except Exception as e:
            logger.warning(f"Failed to load config, using defaults: {e}")
            return default_config

    def _check_rtx3060_resources(self) -> bool:
        """RTX3060のリソースチェック"""
        try:
            import torch
            if not torch.cuda.is_available():
                logger.error("[RTX3060] CUDA not available")
                return False

            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory < 8:
                logger.warning(f"[RTX3060] GPU memory {gpu_memory:.1f}GB < 8GB required")
                return False

            logger.info(f"[RTX3060] GPU memory: {gpu_memory:.1f}GB ✓")
            return True
        except Exception as e:
            logger.error(f"[RTX3060] Resource check failed: {e}")
            return False

    def run_complete_pipeline(self) -> bool:
        """
        RTX3060対応完全自動パイプライン実行
        元の重みを凍結しつつQLoRAでファインチューニング
        """
        logger.info("[START] RTX3060 SO8T Complete Automation Pipeline")
        logger.info("Frozen base weights + QLoRA fine-tuning enabled")

        # RTX3060リソースチェック
        if not self._check_rtx3060_resources():
            logger.error("RTX3060 resource check failed")
            return False

        steps = [
            "Multimodal Dataset Collection",
            "Data Preprocessing (4-class + Cleansing)",
            "Soul Weights Dataset Generation",  # 魂の重み学習データ生成
            "QLoRA Fine-tuning (Frozen Weights + Soul)",
            "Multimodal Integration",
            "Benchmark Evaluation",
            "HF Upload",
            "Cleanup & Task Removal"
        ]

        with tqdm(total=len(steps), desc="RTX3060 SO8T Pipeline", unit="step") as self.main_progress:
            try:
                # STEP 1: マルチモーダルデータセット収集
                self._update_progress("Dataset Collection", 1, len(steps))
                if not self._collect_multimodal_datasets():
                    return False
                self.main_progress.update(1)

                # STEP 2: データ前処理（四値分類 + クレンジング）
                self._update_progress("Data Preprocessing", 2, len(steps))
                if not self._preprocess_data():
                    return False
                self.main_progress.update(1)

                # STEP 3: 魂の重みデータセット生成（実装ログに基づく）
                self._update_progress("Soul Weights Dataset Generation", 3, len(steps))
                if not self._generate_soul_weights_dataset():
                    return False
                self.main_progress.update(1)

                # STEP 4: QLoRAファインチューニング（重み凍結 + 魂の重み）
                self._update_progress("QLoRA Training (Frozen Weights + Soul)", 4, len(steps))
                if not self._train_qlora_frozen_with_soul():
                    return False
                self.main_progress.update(1)

                # STEP 4: マルチモーダル統合と最適化
                self._update_progress("Multimodal Integration", 4, len(steps))
                if not self._integrate_multimodal():
                    return False
                self.main_progress.update(1)

                # STEP 5: 包括的ベンチマーク評価
                self._update_progress("Benchmark Evaluation", 5, len(steps))
                if not self._run_benchmarks():
                    return False
                self.main_progress.update(1)

                # STEP 6: HFアップロード
                self._update_progress("HF Upload", 6, len(steps))
                if not self._upload_to_hf():
                    return False
                self.main_progress.update(1)

                # STEP 7: クリーンアップとタスク削除
                self._update_progress("Cleanup", 7, len(steps))
                self._cleanup_and_remove_tasks()
                self.main_progress.update(1)

                logger.info("[SUCCESS] RTX3060 SO8T Pipeline completed successfully!")
                return True

            except Exception as e:
                logger.error(f"[ERROR] Pipeline failed: {e}")
                logger.error(traceback.format_exc())
                self.error_log.append(str(e))
                return False

    def _collect_multimodal_datasets(self) -> bool:
        """STEP 1: マルチモーダルデータセット収集"""
        logger.info("[STEP 1] Collecting multimodal datasets...")

        try:
            # データセット収集スクリプト実行
            cmd = [
                sys.executable,
                "scripts/data/collect_multimodal_datasets.py",
                "--config", str(self.config_path),
                "--output", str(self.datasets_dir)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Dataset collection failed: {result.stderr}")
                return False

            logger.info("[OK] Multimodal datasets collected")
            return True

        except Exception as e:
            logger.error(f"Dataset collection error: {e}")
            return False

    def _preprocess_data(self) -> bool:
        """STEP 2: データ前処理（四値分類 + クレンジング）"""
        logger.info("[STEP 2] Preprocessing data with 4-class classification...")

        try:
            # データ前処理スクリプト実行
            cmd = [
                sys.executable,
                "scripts/data/preprocess_multimodal_data.py",
                "--input", str(self.datasets_dir),
                "--output", str(self.datasets_dir / "processed"),
                "--four_class_classification"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Data preprocessing failed: {result.stderr}")
                return False

            logger.info("[OK] Data preprocessing completed")
            return True

        except Exception as e:
            logger.error(f"Data preprocessing error: {e}")
            return False

    def _generate_soul_weights_dataset(self) -> bool:
        """STEP 3: 魂の重みデータセット生成（実装ログに基づく）"""
        logger.info("[STEP 3] Generating Soul Weights Dataset...")

        try:
            # 魂の重みデータセット生成スクリプト実行
            cmd = [
                sys.executable,
                "scripts/data/generate_soul_weights_dataset.py",
                "--config", "configs/generate_soul_weights.yaml",
                "--output_dir", str(self.datasets_dir / "soul_weights")
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Soul weights dataset generation failed: {result.stderr}")
                return False

            logger.info("[OK] Soul Weights Dataset generated successfully")
            return True

        except Exception as e:
            logger.error(f"Soul weights dataset generation error: {e}")
            return False

    def _train_qlora_frozen_with_soul(self) -> bool:
        """STEP 4: QLoRAファインチューニング（重み凍結 + 魂の重み）"""
        logger.info("[STEP 4] Training with QLoRA (frozen base weights + soul weights)...")

        try:
            # 魂の重みデータセットを統合したRTX3060向けQLoRAトレーニング実行
            qlora_config = "configs/train_so8t_phi3_qlora_rtx3060_soul.yaml"
            cmd = [
                sys.executable,
                "scripts/training/train_so8t_phi3_qlora_with_soul.py",
                "--config", qlora_config,
                "--soul_dataset", str(self.datasets_dir / "soul_weights")
            ]

            # 環境変数でRTX3060設定を適用
            env = os.environ.copy()
            env.update({
                'CUDA_VISIBLE_DEVICES': '0',
                'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
                'TORCH_USE_CUDA_DSA': '1'
            })

            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if result.returncode != 0:
                logger.error(f"QLoRA training with soul weights failed: {result.stderr}")
                return False

            logger.info("[OK] QLoRA training completed (frozen weights + soul weights)")
            return True

        except Exception as e:
            logger.error(f"QLoRA training with soul error: {e}")
            return False

    def _integrate_multimodal(self) -> bool:
        """STEP 4: マルチモーダル統合"""
        logger.info("[STEP 4] Integrating multimodal capabilities...")

        try:
            # SO8T統合スクリプト実行
            cmd = [
                sys.executable,
                "scripts/conversion/bake_so8t_into_transformer.py",
                "--model_path", str(self.checkpoints_dir / "finetuning" / "so8t_phi3_rtx3060" / "final"),
                "--output_path", str(self.models_dir / "borea_phi35_so8t_rtx3060")
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Multimodal integration failed: {result.stderr}")
                return False

            logger.info("[OK] Multimodal integration completed")
            return True

        except Exception as e:
            logger.error(f"Multimodal integration error: {e}")
            return False

    def _run_benchmarks(self) -> bool:
        """STEP 5: ベンチマーク評価"""
        logger.info("[STEP 5] Running comprehensive benchmarks...")

        try:
            # ベンチマーク評価スクリプト実行
            cmd = [
                sys.executable,
                "scripts/evaluation/run_comprehensive_benchmarks.py",
                "--model_path", str(self.models_dir / "borea_phi35_so8t_rtx3060"),
                "--output_dir", str(self.logs_dir / "benchmarks_rtx3060")
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Benchmark evaluation failed: {result.stderr}")
                return False

            logger.info("[OK] Benchmark evaluation completed")
            return True

        except Exception as e:
            logger.error(f"Benchmark evaluation error: {e}")
            return False

    def _upload_to_hf(self) -> bool:
        """STEP 6: HFアップロード"""
        logger.info("[STEP 6] Uploading to HuggingFace...")

        try:
            # HFアップロードスクリプト実行
            cmd = [
                sys.executable,
                "scripts/upload/upload_to_huggingface.py",
                "--model_path", str(self.models_dir / "borea_phi35_so8t_rtx3060"),
                "--repo_name", "borea-phi35-so8t-rtx3060",
                "--private", "false"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"HF upload failed: {result.stderr}")
                return False

            logger.info("[OK] Model uploaded to HuggingFace")
            return True

        except Exception as e:
            logger.error(f"HF upload error: {e}")
            return False

    def _cleanup_and_remove_tasks(self):
        """STEP 7: クリーンアップとタスク削除"""
        logger.info("[STEP 7] Cleaning up and removing scheduled tasks...")

        try:
            # タスクスケジューラからタスク削除
            cmd = [
                "schtasks", "/delete", "/tn", "SO8T_RTX3060_Automation", "/f"
            ]
            subprocess.run(cmd, capture_output=True)

            # 一時ファイルクリーンアップ
            temp_files = list(self.base_dir.glob("temp_*"))
            for temp_file in temp_files:
                if temp_file.exists():
                    temp_file.unlink()

            logger.info("[OK] Cleanup completed, tasks removed")

        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="RTX3060 SO8T Complete Automation Pipeline (Frozen weights + QLoRA)"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/complete_so8t_pipeline.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show current pipeline status'
    )

    args = parser.parse_args()

    # パイプライン初期化
    pipeline = RTX3060SO8TAutomationPipeline(args.config)

    if args.status:
        # ステータス表示
        print("RTX3060 SO8T Automation Pipeline Status:")
        print(f"- RTX3060 Memory: {pipeline.rtx3060_config['max_memory_gb']}GB")
        print(f"- QLoRA Enabled: {pipeline.rtx3060_config['use_qlora']}")
        print(f"- Freeze Base Weights: {pipeline.rtx3060_config['freeze_base_weights']}")
        print(f"- Batch Size: {pipeline.rtx3060_config['batch_size']}")
        print(f"- Gradient Accumulation: {pipeline.rtx3060_config['gradient_accumulation']}")
        return

    # パイプライン実行
    success = pipeline.run_complete_pipeline()

    if success:
        logger.info("[SUCCESS] RTX3060 SO8T pipeline completed successfully!")
        # オーディオ通知
        try:
            import winsound
            winsound.PlaySound(r"C:\Users\downl\Desktop\SO8T\.cursor\marisa_owattaze.wav", winsound.SND_FILENAME)
        except:
            print('\a')  # ビープ音
    else:
        logger.error("[FAILED] RTX3060 SO8T pipeline failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
