#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automatic AEGIS-Phi3.5-Thinking-v2.0 Pipeline
完全自動化されたAEGIS-Phi3.5-Thinking-v2.0作成パイプライン

このスクリプトは以下の処理を自動実行します：
1. SFTデータセット統合（多変量解析クレンジング）
2. PPO学習実行
3. GGUF変換（BF16）
4. 業界標準ベンチマーク + ELYZA-100全問
5. ABテスト（エラーバー付きグラフ + 統計分析）
6. HFアップロード用フォルダー作成
"""

import os
import json
import time
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
import psutil
import winreg
import schedule
import atexit
import signal
import gc

# SO8T imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automatic_aegis_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutomaticAEGISPipeline:
    """完全自動化AEGISパイプライン"""

    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent
        self.checkpoint_dir = self.base_path / "checkpoints" / "automatic_aegis"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # セッション管理
        self.session_file = self.checkpoint_dir / 'session_state.json'
        self.rolling_checkpoints = []
        self.max_checkpoints = 5
        self.checkpoint_interval = 180  # 3分

        # パイプライン状態
        self.is_running = False
        self.current_stage = "idle"
        self.session_id = self.generate_session_id()

        # モデルパス
        self.model_a_path = None  # Boreas-phi3.5-instinct-jp BF16 GGUF
        self.model_b_path = None  # 新規作成モデル BF16 GGUF

        # ベンチマーク結果
        self.benchmark_results = {}

        # シグナルハンドラ設定
        self.setup_signal_handlers()
        atexit.register(self.emergency_save)

        logger.info(f"Automatic AEGIS Pipeline initialized - Session: {self.session_id}")

    def generate_session_id(self) -> str:
        """セッションID生成"""
        return f"aegis_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"

    def setup_signal_handlers(self):
        """シグナルハンドラ設定"""
        def signal_handler(signum, frame):
            logger.warning(f"Signal {signum} received - emergency save")
            self.emergency_save()
            os._exit(1)

        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except (OSError, ValueError) as e:
            logger.warning(f"Signal handler setup failed: {e}")

    def create_rolling_checkpoint(self):
        """ローリングチェックポイント作成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        checkpoint_data = {
            'timestamp': timestamp,
            'stage': self.current_stage,
            'session_id': self.session_id,
            'model_a_path': str(self.model_a_path) if self.model_a_path else None,
            'model_b_path': str(self.model_b_path) if self.model_b_path else None,
            'benchmark_results': self.benchmark_results,
            'system_info': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            }
        }

        checkpoint_file = self.checkpoint_dir / f"checkpoint_{timestamp}.json"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

        self.rolling_checkpoints.append(checkpoint_file)
        if len(self.rolling_checkpoints) > self.max_checkpoints:
            old_checkpoint = self.rolling_checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()

        self.save_session_state()
        logger.info(f"Rolling checkpoint created: {checkpoint_file}")

    def save_session_state(self):
        """セッション状態保存"""
        session_data = {
            'session_id': self.session_id,
            'current_stage': self.current_stage,
            'timestamp': datetime.now().isoformat(),
            'last_checkpoint': str(self.rolling_checkpoints[-1]) if self.rolling_checkpoints else None,
        }

        with open(self.session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

    def emergency_save(self):
        """緊急保存"""
        try:
            emergency_data = {
                'session_id': self.session_id,
                'current_stage': self.current_stage,
                'timestamp': datetime.now().isoformat(),
                'emergency_save': True,
                'benchmark_results': self.benchmark_results
            }

            emergency_file = self.checkpoint_dir / f'emergency_save_{self.session_id}.json'
            with open(emergency_file, 'w', encoding='utf-8') as f:
                json.dump(emergency_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Emergency save completed: {emergency_file}")
        except Exception as e:
            logger.error(f"Emergency save failed: {e}")

    def load_checkpoint(self, checkpoint_path: str):
        """チェックポイントから復旧"""
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)

            self.current_stage = checkpoint_data.get('stage', 'idle')
            self.model_a_path = Path(checkpoint_data['model_a_path']) if checkpoint_data.get('model_a_path') else None
            self.model_b_path = Path(checkpoint_data['model_b_path']) if checkpoint_data.get('model_b_path') else None
            self.benchmark_results = checkpoint_data.get('benchmark_results', {})

            logger.info(f"Loaded checkpoint: {checkpoint_path}")
            logger.info(f"Resuming from stage: {self.current_stage}")

            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    def run_sft_integration(self):
        """SFTデータセット統合実行"""
        logger.info("Starting SFT dataset integration...")
        self.current_stage = "sft_integration"

        try:
            cmd = [sys.executable, "scripts/data/sft_dataset_integration_phi35_thinking.py"]
            result = subprocess.run(cmd, cwd=self.base_path, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("SFT integration completed successfully")
                return True
            else:
                logger.error(f"SFT integration failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"SFT integration error: {e}")
            return False

    def run_ppo_training(self):
        """PPO学習実行"""
        logger.info("Starting PPO training...")
        self.current_stage = "ppo_training"

        try:
            # PPO学習スクリプト実行（既存のものを使用）
            # PPOトレーニング実行
            model_path = "models/Borea-Phi-3.5-mini-Instruct-Jp"
            dataset_path = "data/integrated/so8t_integrated_ppo_dataset_main_20251201_205340.jsonl"
            config_path = "scripts/training/so8t_ppo_config.json"
            output_dir = "H:/from_D/webdataset/checkpoints/automatic_aegis/ppo_output"

            cmd = [
                sys.executable, "scripts/training/so8t_integrated_ppo_trainer.py",
                "--model_path", model_path,
                "--dataset_path", dataset_path,
                "--config_path", config_path,
                "--output_dir", output_dir
            ]
            result = subprocess.run(cmd, cwd=self.base_path, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("PPO training completed successfully")
                return True
            else:
                logger.error(f"PPO training failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"PPO training error: {e}")
            return False

    def run_gguf_conversion(self):
        """GGUF変換実行（BF16）"""
        logger.info("Starting GGUF conversion (BF16)...")
        self.current_stage = "gguf_conversion"

        try:
            # Boreas-phi3.5-instinct-jp のGGUF変換
            model_a_output = Path("H:/from_D/webdataset/gguf_models/boreas_phi35_instinct_jp_bf16.gguf")

            # すでに存在する場合はスキップ
            if model_a_output.exists():
                self.model_a_path = model_a_output
                logger.info(f"Model A GGUF already exists, skipping conversion: {model_a_output}")
            else:
                cmd_a = [
                    sys.executable, "scripts/conversion/convert_phi35_to_gguf.py",
                    "--model_path", "models/Borea-Phi-3.5-mini-Instruct-Jp",
                    "--output_path", "H:/from_D/webdataset/gguf_models/boreas_phi35_instinct_jp_bf16.gguf",
                    "--quantization", "bf16"
                ]

                result_a = subprocess.run(cmd_a, cwd=self.base_path, capture_output=True, text=True)
                if result_a.returncode == 0:
                    self.model_a_path = model_a_output
                    logger.info(f"Model A GGUF conversion completed: {model_a_output}")
                else:
                    logger.error(f"Model A GGUF conversion failed: {result_a.stderr}")
                    return False

            # 新規作成モデルのGGUF変換（PPO学習済みモデル）
            model_b_output = Path("H:/from_D/webdataset/gguf_models/aegis_phi35_thinking_v2_bf16.gguf")
            ppo_model_path = "H:/from_D/webdataset/checkpoints/automatic_aegis/ppo_output"
            cmd_b = [
                sys.executable, "scripts/conversion/convert_phi35_to_gguf.py",
                "--model_path", ppo_model_path,  # PPO学習済みモデルを使用
                "--output_path", "H:/from_D/webdataset/gguf_models/aegis_phi35_thinking_v2_bf16.gguf",
                "--quantization", "bf16"
            ]

            result_b = subprocess.run(cmd_b, cwd=self.base_path, capture_output=True, text=True)
            if result_b.returncode == 0:
                self.model_b_path = model_b_output
                logger.info(f"Model B GGUF conversion completed: {model_b_output}")
                return True
            else:
                logger.error(f"Model B GGUF conversion failed: {result_b.stderr}")
                return False

        except Exception as e:
            logger.error(f"GGUF conversion error: {e}")
            return False

    def run_benchmarking(self):
        """ベンチマーク実行"""
        logger.info("Starting comprehensive benchmarking...")
        self.current_stage = "benchmarking"

        try:
            # ABテストベンチマーク実行
            cmd = [
                sys.executable, "scripts/evaluation/comprehensive_ab_benchmark.py",
                "--model_a", str(self.model_a_path),
                "--model_b", str(self.model_b_path),
                "--include_elyza", "true",
                "--elyza_full", "true"
            ]

            result = subprocess.run(cmd, cwd=self.base_path, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("Benchmarking completed successfully")

                # 結果ファイル読み込み
                results_file = self.base_path / "benchmark_results" / "ab_test_results.json"
                if results_file.exists():
                    with open(results_file, 'r', encoding='utf-8') as f:
                        self.benchmark_results = json.load(f)

                return True
            else:
                logger.error(f"Benchmarking failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Benchmarking error: {e}")
            return False

    def prepare_hf_upload(self):
        """HFアップロード用フォルダー作成"""
        logger.info("Preparing HF upload package...")
        self.current_stage = "hf_upload_preparation"

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            upload_dir = self.base_path / "hf_upload" / f"aegis_phi35_thinking_v2_{timestamp}"

            upload_dir.mkdir(parents=True, exist_ok=True)

            # モデルファイルコピー
            if self.model_a_path and self.model_a_path.exists():
                import shutil
                shutil.copy2(self.model_a_path, upload_dir / "model_a_boreas_phi35_bf16.gguf")

            if self.model_b_path and self.model_b_path.exists():
                shutil.copy2(self.model_b_path, upload_dir / "model_b_aegis_phi35_thinking_v2_bf16.gguf")

            # ベンチマーク結果コピー
            benchmark_dir = self.base_path / "benchmark_results"
            if benchmark_dir.exists():
                shutil.copytree(benchmark_dir, upload_dir / "benchmark_results", dirs_exist_ok=True)

            # README作成
            readme_content = f"""# AEGIS-Phi3.5-Thinking-v2.0

## Model Description
AEGIS-Phi3.5-Thinking-v2.0 is an advanced Japanese language model with structured thinking capabilities.

## Models
- **Model A**: Boreas-Phi3.5-Instinct-JP (Baseline)
- **Model B**: AEGIS-Phi3.5-Thinking-v2.0 (Enhanced with SO(8) reasoning)

## Benchmark Results
See `benchmark_results/` directory for comprehensive AB testing results including:
- Industry standard benchmarks
- ELYZA-100 full benchmark
- Statistical analysis with ANOVA, effect sizes, and p-values
- Error bar charts and summary statistics

## Technical Details
- Architecture: Phi-3.5 with SO(8) reasoning integration
- Quantization: BF16 GGUF
- Training: SFT + PPO with multivariate data cleansing
- Created: {datetime.now().isoformat()}

## Citation
```bibtex
@misc{{aegis-phi35-thinking-v2,
  title={{AEGIS-Phi3.5-Thinking-v2.0}},
  author={{SO8T Team}},
  year={{2025}}
}}
```
"""

            with open(upload_dir / "README.md", 'w', encoding='utf-8') as f:
                f.write(readme_content)

            logger.info(f"HF upload package prepared: {upload_dir}")
            return str(upload_dir)

        except Exception as e:
            logger.error(f"HF upload preparation error: {e}")
            return None

    def run_complete_pipeline(self):
        """完全パイプライン実行"""
        logger.info("Starting complete AEGIS pipeline...")
        self.is_running = True

        try:
            # 初期チェックポイント
            self.create_rolling_checkpoint()

            # Phase 1: SFTデータセット統合
            if not self.run_sft_integration():
                raise Exception("SFT integration failed")

            self.create_rolling_checkpoint()

            # Phase 2: PPO学習
            if not self.run_ppo_training():
                raise Exception("PPO training failed")

            self.create_rolling_checkpoint()

            # Phase 3: GGUF変換
            if not self.run_gguf_conversion():
                raise Exception("GGUF conversion failed")

            self.create_rolling_checkpoint()

            # Phase 4: ベンチマーク
            if not self.run_benchmarking():
                raise Exception("Benchmarking failed")

            self.create_rolling_checkpoint()

            # Phase 5: HFアップロード準備
            upload_path = self.prepare_hf_upload()
            if not upload_path:
                raise Exception("HF upload preparation failed")

            # 完了処理
            self.on_pipeline_completion(upload_path)

            logger.info("Complete AEGIS pipeline finished successfully")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.on_pipeline_failure(e)

        finally:
            self.is_running = False

    def on_pipeline_completion(self, upload_path: str):
        """パイプライン完了時の処理"""
        logger.info("Pipeline completed successfully!")

        # 完了通知
        try:
            subprocess.run([
                "powershell", "-ExecutionPolicy", "Bypass",
                "-File", "scripts\\utils\\play_audio_notification.ps1"
            ], check=True)
        except Exception as e:
            logger.error(f"Audio notification failed: {e}")

        print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          [SUCCESS] AEGIS PIPELINE COMPLETED!                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

📦 HF Upload Package: {upload_path}
🔬 Benchmark Results: Available in benchmark_results/
🤖 Models: BF16 GGUF format ready for deployment

Next Steps:
1. Review benchmark results
2. Upload to Hugging Face: `huggingface-cli upload {Path(upload_path).name} {upload_path} --repo-type dataset`
3. Deploy models for inference

Thank you for using SO8T Automatic Pipeline!
""")

    def on_pipeline_failure(self, error: Exception):
        """パイプライン失敗時の処理"""
        logger.error(f"Pipeline failed: {error}")

        # エラー通知
        try:
            subprocess.run([
                "powershell", "-ExecutionPolicy", "Bypass",
                "-c", "[System.Console]::Beep(800, 1000)"
            ], check=True)
        except Exception as e:
            logger.error(f"Error audio notification failed: {e}")

    def schedule_checkpoints(self):
        """チェックポイントスケジューリング"""
        def checkpoint_job():
            if self.is_running:
                self.create_rolling_checkpoint()

        schedule.every(self.checkpoint_interval).seconds.do(checkpoint_job)

        def run_scheduler():
            while self.is_running:
                schedule.run_pending()
                time.sleep(1)

        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()

    def start(self, resume: bool = False):
        """パイプライン開始"""
        logger.info("Automatic AEGIS Pipeline starting...")

        # チェックポイントスケジューラ開始
        self.schedule_checkpoints()

        if resume:
            # 最新チェックポイントから復旧
            if self.rolling_checkpoints:
                last_checkpoint = str(self.rolling_checkpoints[-1])
                if self.load_checkpoint(last_checkpoint):
                    logger.info("Resumed from checkpoint")
                else:
                    logger.info("Failed to resume, starting fresh")
                    self.run_complete_pipeline()
            else:
                logger.info("No checkpoints found, starting fresh")
                self.run_complete_pipeline()
        else:
            # 新規開始
            self.run_complete_pipeline()

    def setup_autostart(self):
        """Windows自動起動設定"""
        try:
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Run",
                0, winreg.KEY_SET_VALUE
            )

            script_path = str(Path(__file__).absolute())
            python_path = sys.executable

            command = f'"{python_path}" "{script_path}"'

            winreg.SetValueEx(key, "SO8TAutomaticAEGISPipeline", 0, winreg.REG_SZ, command)
            winreg.CloseKey(key)

            logger.info("Autostart registered successfully")

        except Exception as e:
            logger.error(f"Failed to setup autostart: {e}")

def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description='Automatic AEGIS-Phi3.5-Thinking-v2.0 Pipeline')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    parser.add_argument('--autostart', action='store_true', help='Setup Windows autostart')

    args = parser.parse_args()

    # パイプライン初期化
    pipeline = AutomaticAEGISPipeline()

    if args.autostart:
        pipeline.setup_autostart()
        print("[OK] Windows autostart configured for power-on automatic execution")
        return

    # パイプライン実行
    pipeline.start(resume=args.resume)

if __name__ == "__main__":
    main()
