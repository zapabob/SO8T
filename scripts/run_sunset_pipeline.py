#!/usr/bin/env python3
"""
RTX 3060 Optimized Sunset Pipeline Main Script with PowerShell-style Progress
Sunset Pipeline Main Execution Script with tqdm-like progress and logging
"""

import os
import sys
import json
import argparse
import time
import subprocess
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

# tqdm風進捗表示用
class PowerShellProgressBar:
    def __init__(self, total: int, desc: str = "", unit: str = "it"):
        self.total = total
        self.current = 0
        self.desc = desc
        self.unit = unit
        self.start_time = time.time()
        self.last_update = self.start_time

    def update(self, n: int = 1):
        self.current += n
        self._display()

    def set_description(self, desc: str):
        self.desc = desc
        self._display()

    def _display(self):
        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
        else:
            eta = 0

        percent = min(100.0, (self.current / self.total) * 100)

        eta_str = f"{int(eta//3600):02d}:{int((eta%3600)//60):02d}:{int(eta%60):02d}"
        elapsed_str = f"{int(elapsed//3600):02d}:{int((elapsed%3600)//60):02d}:{int(elapsed%60):02d}"

        # PowerShell風のプログレスバー (ASCII対応)
        bar_width = 40
        filled = int(bar_width * percent / 100)
        bar = "=" * filled + "-" * (bar_width - filled)

        print(f"\r[{bar}] {percent:5.1f}% | {self.current}/{self.total} [{elapsed_str}<{eta_str}, {self.current/elapsed:.2f}{self.unit}/s] {self.desc}", end="", flush=True)

        if self.current >= self.total:
            print()  # 改行

# logging風フォーマット
class PowerShellLogger:
    def __init__(self):
        self.start_time = datetime.now()

    def info(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = datetime.now() - self.start_time
        elapsed_str = f"{elapsed.seconds//3600:02d}:{(elapsed.seconds%3600)//60:02d}:{elapsed.seconds%60:02d}"
        print(f"[{timestamp}] [INFO] [{elapsed_str}] {message}")

    def warning(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = datetime.now() - self.start_time
        elapsed_str = f"{elapsed.seconds//3600:02d}:{(elapsed.seconds%3600)//60:02d}:{elapsed.seconds%60:02d}"
        print(f"[{timestamp}] [WARN] [{elapsed_str}] {message}")

    def error(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = datetime.now() - self.start_time
        elapsed_str = f"{elapsed.seconds//3600:02d}:{(elapsed.seconds%3600)//60:02d}:{elapsed.seconds%60:02d}"
        print(f"[{timestamp}] [ERROR] [{elapsed_str}] {message}")

    def success(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = datetime.now() - self.start_time
        elapsed_str = f"{elapsed.seconds//3600:02d}:{(elapsed.seconds%3600)//60:02d}:{elapsed.seconds%60:02d}"
        print(f"[{timestamp}] [SUCCESS] [{elapsed_str}] {message}")

class SunsetPipelineRTX3060:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.config_dir = self.project_root / "config"
        self.scripts_dir = self.project_root / "scripts"

        # PowerShell風ロガー初期化
        self.logger = PowerShellLogger()

        # パイプライン設定
        self.pipeline_phases = {
            'data': {'duration': 1800, 'description': 'Data Pipeline Processing'},
            'training': {'duration': 7200, 'description': 'Unsloth SO8T Training'},
            'evaluation': {'duration': 3600, 'description': 'Benchmark Evaluation'},
            'abc': {'duration': 10800, 'description': 'ABC Comparative Testing'}
        }

        # Load configuration files
        self.load_configs()

    def load_configs(self):
        """Load configuration files"""
        config_files = ['hardware.json', 'training.json', 'dataset.json', 'benchmark.json']

        self.configs = {}
        for config_file in config_files:
            config_path = self.config_dir / config_file
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.configs[config_file.replace('.json', '')] = json.load(f)
            else:
                self.logger.warning(f"Config file not found: {config_file}")

    def run_data_pipeline(self):
        """Run data pipeline with PowerShell progress"""
        phase_config = self.pipeline_phases['data']
        self.logger.info(f"Starting {phase_config['description']}")

        data_script = self.scripts_dir / "data_processing" / "dataset_pipeline.py"
        if data_script.exists():
            success = self._run_script_with_progress(
                str(data_script), [], phase_config['duration'],
                f"Processing datasets..."
            )
            if success:
                self.logger.success(f"{phase_config['description']} completed")
            else:
                self.logger.error(f"{phase_config['description']} failed")
        else:
            self.logger.error("Data pipeline script not found")

    def run_model_training(self):
        """Run advanced SO8T Quadrality training with Unsloth and PowerShell progress"""
        phase_config = self.pipeline_phases['training']
        self.logger.info(f"Starting {phase_config['description']}")
        self.logger.info("Techniques: SO8T + DeepSeek GRPO + MHC + imatrix + Lightning Fast Training")

        training_script = self.scripts_dir / "training" / "train_unsloth_so8t.py"
        if training_script.exists():
            success = self._run_script_with_progress(
                str(training_script), ["--phase", "full"], phase_config['duration'],
                "Unsloth SO8T training in progress..."
            )
            if success:
                self.logger.success(f"{phase_config['description']} completed")
            else:
                self.logger.warning("Unsloth training failed, trying fallback")
                self._run_fallback_training()
        else:
            self.logger.error("Unsloth training script not found, using fallback")
            self._run_fallback_training()

    def _run_fallback_training(self):
        """Fallback training method"""
        fallback_script = self.scripts_dir / "training" / "train_quadrality_model.py"
        if fallback_script.exists():
            phase_config = self.pipeline_phases['training']
            success = self._run_script_with_progress(
                str(fallback_script), ["--phase", "full"], phase_config['duration'],
                "Standard SO8T training (fallback)..."
            )
            if success:
                self.logger.success("Fallback training completed")
            else:
                self.logger.error("Fallback training failed")

    def run_evaluation(self):
        """Run evaluation with PowerShell progress"""
        phase_config = self.pipeline_phases['evaluation']
        self.logger.info(f"Starting {phase_config['description']}")

        eval_script = self.scripts_dir / "evaluation" / "run_benchmarks.py"
        if eval_script.exists():
            success = self._run_script_with_progress(
                str(eval_script), [], phase_config['duration'],
                "Running benchmark evaluations..."
            )
            if success:
                self.logger.success(f"{phase_config['description']} completed")
            else:
                self.logger.error(f"{phase_config['description']} failed")
        else:
            self.logger.error("Evaluation script not found")

    def run_abc_testing(self):
        """Run ABC testing with PowerShell progress"""
        phase_config = self.pipeline_phases['abc']
        self.logger.info(f"Starting {phase_config['description']}")
        self.logger.info("Comparing: A(Qwen-base) vs B(SO8T-trained) vs C(AEGIS-Phi3.5)")

        abc_script = self.scripts_dir / "evaluation" / "abc_testing.py"
        if abc_script.exists():
            success = self._run_script_with_progress(
                str(abc_script), [], phase_config['duration'],
                "ABC comparative testing in progress..."
            )
            if success:
                self.logger.success(f"{phase_config['description']} completed")
            else:
                self.logger.error(f"{phase_config['description']} failed")
        else:
            self.logger.error("ABC testing script not found")

    def _run_script_with_progress(self, script_path: str, args: list, estimated_duration: int, description: str) -> bool:
        """PowerShell風にスクリプトを実行し、進捗を表示"""
        try:
            # 進捗バー初期化
            progress_bar = PowerShellProgressBar(total=100, desc=description)

            # サブプロセス開始
            cmd = [sys.executable, script_path] + args
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            start_time = time.time()
            last_progress = 0

            # リアルタイム監視
            while process.poll() is None:
                time.sleep(1)  # 1秒ごとにチェック
                elapsed = time.time() - start_time

                # 推定進捗計算（実際の進捗は取得できないので時間ベース）
                if elapsed < estimated_duration:
                    progress = min(95, int((elapsed / estimated_duration) * 100))
                else:
                    progress = 95  # 推定時間を超えても95%まで

                if progress > last_progress:
                    progress_bar.update(progress - last_progress)
                    last_progress = progress

            # 最終進捗
            if process.returncode == 0:
                progress_bar.update(100 - last_progress)
                return True
            else:
                self.logger.error(f"Script execution failed with code {process.returncode}")
                return False

        except Exception as e:
            self.logger.error(f"Error running script: {e}")
            return False

    def run_phase_with_timing(self, phase_name: str):
        """指定フェーズを実行し、時間を計測"""
        phase_config = self.pipeline_phases[phase_name]

        self.logger.info(f"=== PHASE {phase_name.upper()} START ===")
        self.logger.info(f"Duration: ~{phase_config['duration']//3600}h {(phase_config['duration']%3600)//60}m")
        self.logger.info(f"Description: {phase_config['description']}")

        start_time = time.time()

        if phase_name == 'data':
            self.run_data_pipeline()
        elif phase_name == 'training':
            self.run_model_training()
        elif phase_name == 'evaluation':
            self.run_evaluation()
        elif phase_name == 'abc':
            self.run_abc_testing()

        elapsed = time.time() - start_time
        elapsed_str = f"{int(elapsed//3600):02d}:{int((elapsed%3600)//60):02d}:{int(elapsed%60):02d}"
        self.logger.info(f"=== PHASE {phase_name.upper()} END ===")
        self.logger.info(f"Actual duration: {elapsed_str}")

    def run_full_pipeline(self):
        """Run full pipeline with PowerShell-style progress and logging"""
        self.logger.info("=" * 80)
        self.logger.info("SUNSET PIPELINE RTX 3060 FULL EXECUTION")
        self.logger.info("Advanced SO8T Quadrality Training with Unsloth Acceleration")
        self.logger.info("=" * 80)
        self.logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("Environment: RTX 3060 + 32GB RAM")
        self.logger.info("Techniques: SO8T + DeepSeek GRPO + MHC + imatrix + Unsloth 4-bit")
        self.logger.info("=" * 80)

        pipeline_start = time.time()
        total_phases = len(self.pipeline_phases)

        try:
            # Phase 1: Data preparation
            self.logger.info("\n[PHASE 1/4] Data Pipeline")
            self.run_phase_with_timing('data')

            # Phase 2: Model training
            self.logger.info("\n[PHASE 2/4] Model Training")
            self.run_phase_with_timing('training')

            # Phase 3: Evaluation
            self.logger.info("\n[PHASE 3/4] Benchmark Evaluation")
            self.run_phase_with_timing('evaluation')

            # Phase 4: ABC testing
            self.logger.info("\n[PHASE 4/4] ABC Comparative Testing")
            self.run_phase_with_timing('abc')

            # 完了サマリー
            total_elapsed = time.time() - pipeline_start
            total_elapsed_str = f"{int(total_elapsed//3600):02d}:{int((total_elapsed%3600)//60):02d}:{int(total_elapsed%60):02d}"

            self.logger.info("=" * 80)
            self.logger.success("SUNSET PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
            self.logger.info(f"Total execution time: {total_elapsed_str}")
            self.logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info("=" * 80)

            # 成果物確認
            self._show_pipeline_results()

            return True

        except KeyboardInterrupt:
            self.logger.warning("Pipeline execution interrupted by user")
            return False
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            return False

    def _show_pipeline_results(self):
        """パイプライン実行結果を表示"""
        self.logger.info("\n[PIPELINE RESULTS]")
        self.logger.info("-" * 40)

        # モデル確認
        models_dir = self.project_root / "models"
        if (models_dir / "unsloth_so8t_qwen_7b_final").exists():
            self.logger.success("✓ Unsloth SO8T trained model: Available")
        else:
            self.logger.warning("⚠ Unsloth SO8T trained model: Not found")

        # 評価結果確認
        results_dir = self.project_root / "results"
        if (results_dir / "benchmarks").exists():
            benchmark_files = list((results_dir / "benchmarks").glob("*.json"))
            self.logger.success(f"✓ Benchmark results: {len(benchmark_files)} files")

        if (results_dir / "abc_testing").exists():
            abc_files = list((results_dir / "abc_testing").glob("*.json"))
            self.logger.success(f"✓ ABC testing results: {len(abc_files)} files")

        self.logger.info("\n[USAGE EXAMPLES]")
        self.logger.info("python scripts/training/train_unsloth_so8t.py --phase sft    # SFT Training")
        self.logger.info("python scripts/evaluation/abc_testing.py                  # ABC Testing")
        self.logger.info("python scripts/sunset_pipeline_demo.py                    # Status Check")

def check_system_requirements():
    """システム要件チェック"""
    logger = PowerShellLogger()

    # GPUチェック
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            logger.warning("GPU: Not available - Unsloth requires NVIDIA GPU")
    except:
        logger.warning("GPU: Cannot detect - PyTorch may not be available")

    # Unslothチェック
    try:
        import unsloth
        logger.info(f"Unsloth: Available (v{unsloth.__version__})")
    except ImportError:
        logger.warning("Unsloth: Not installed - Will use fallback training")
    except NotImplementedError:
        logger.warning("Unsloth: GPU not available for training")

    return True

def main():
    parser = argparse.ArgumentParser(description='RTX 3060 Sunset Pipeline with PowerShell Progress')
    parser.add_argument('--phase', choices=['data', 'training', 'evaluation', 'abc', 'full'],
                       default='full', help='Phase to execute')
    parser.add_argument('--config', help='Configuration directory')
    parser.add_argument('--no-progress', action='store_true', help='Disable PowerShell-style progress')

    args = parser.parse_args()

    # システム要件チェック
    check_system_requirements()

    # パイプライン初期化
    pipeline = SunsetPipelineRTX3060()

    if args.config:
        pipeline.config_dir = Path(args.config)

    # 実行
    if args.phase == 'data':
        pipeline.run_phase_with_timing('data')
    elif args.phase == 'training':
        pipeline.run_phase_with_timing('training')
    elif args.phase == 'evaluation':
        pipeline.run_phase_with_timing('evaluation')
    elif args.phase == 'abc':
        pipeline.run_phase_with_timing('abc')
    elif args.phase == 'full':
        pipeline.run_full_pipeline()

if __name__ == "__main__":
    main()