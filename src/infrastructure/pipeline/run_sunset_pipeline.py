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
import signal
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, List
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

class SunsetPipelineCheckpointManager:
    """サンセットパイプライン用チェックポイントマネージャー"""
    
    def __init__(self, checkpoint_dir: Path, save_interval: int = 300):
        """
        初期化
        
        Args:
            checkpoint_dir: チェックポイント保存ディレクトリ
            save_interval: 自動保存間隔（秒、デフォルト5分）
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "pipeline_checkpoint.json"
        self.save_interval = save_interval
        self.last_save_time = time.time()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # パイプライン状態
        self.pipeline_state = {
            'session_id': self.session_id,
            'start_time': None,
            'current_phase': None,
            'phases': {
                'data': {'status': 'pending', 'progress': 0, 'start_time': None, 'end_time': None},
                'training': {'status': 'pending', 'progress': 0, 'start_time': None, 'end_time': None},
                'evaluation': {'status': 'pending', 'progress': 0, 'start_time': None, 'end_time': None},
                'abc': {'status': 'pending', 'progress': 0, 'start_time': None, 'end_time': None}
            },
            'last_update': None
        }
    
    def should_save(self) -> bool:
        """チェックポイントを保存すべきか判定"""
        current_time = time.time()
        return (current_time - self.last_save_time) >= self.save_interval
    
    def save_checkpoint(self, state: Optional[Dict] = None):
        """チェックポイントを保存"""
        try:
            if state:
                self.pipeline_state.update(state)
            
            self.pipeline_state['last_update'] = datetime.now().isoformat()
            
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(self.pipeline_state, f, indent=2, ensure_ascii=False, default=str)
            
            self.last_save_time = time.time()
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save checkpoint: {e}")
            return False
    
    def load_checkpoint(self) -> Optional[Dict]:
        """チェックポイントを読み込み"""
        if not self.checkpoint_file.exists():
            return None
        
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            # 整合性チェック
            if self._validate_checkpoint(checkpoint_data):
                self.pipeline_state = checkpoint_data
                return checkpoint_data
            else:
                print("[WARN] Checkpoint validation failed, starting fresh")
                return None
        except json.JSONDecodeError as e:
            print(f"[ERROR] Checkpoint file is corrupted: {e}")
            return None
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint: {e}")
            return None
    
    def _validate_checkpoint(self, checkpoint_data: Dict) -> bool:
        """チェックポイントの整合性を検証"""
        required_fields = ['session_id', 'phases']
        if not all(field in checkpoint_data for field in required_fields):
            return False
        
        # フェーズ構造の検証
        phases = checkpoint_data.get('phases', {})
        required_phases = ['data', 'training', 'evaluation', 'abc']
        if not all(phase in phases for phase in required_phases):
            return False
        
        # 各フェーズの必須フィールド検証
        for phase_key, phase_data in phases.items():
            if not isinstance(phase_data, dict):
                return False
            if 'status' not in phase_data or 'progress' not in phase_data:
                return False
        
        return True
    
    def update_phase_status(self, phase_name: str, status: str, progress: int = 0, 
                           start_time: Optional[str] = None, end_time: Optional[str] = None):
        """フェーズ状態を更新"""
        if phase_name not in self.pipeline_state['phases']:
            return
        
        phase = self.pipeline_state['phases'][phase_name]
        phase['status'] = status
        phase['progress'] = progress
        
        if start_time:
            phase['start_time'] = start_time
        if end_time:
            phase['end_time'] = end_time
        
        self.pipeline_state['current_phase'] = phase_name if status == 'running' else None
        
        # 自動保存チェック
        if self.should_save():
            self.save_checkpoint()
    
    def emergency_save(self):
        """緊急保存（Ctrl+C、異常終了時）"""
        try:
            self.pipeline_state['last_update'] = datetime.now().isoformat()
            self.pipeline_state['status'] = 'interrupted'
            
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(self.pipeline_state, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"[CHECKPOINT] Emergency checkpoint saved: {self.checkpoint_file}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save emergency checkpoint: {e}")
            return False
    
    def clear_checkpoint(self):
        """チェックポイントをクリア"""
        if self.checkpoint_file.exists():
            try:
                self.checkpoint_file.unlink()
                print("[CHECKPOINT] Checkpoint cleared")
            except Exception as e:
                print(f"[ERROR] Failed to clear checkpoint: {e}")

class SunsetPipelineRTX3060:
    def __init__(self, checkpoint_dir: Optional[Path] = None, resume: bool = True):
        self.project_root = Path(__file__).parent.parent
        self.config_dir = self.project_root / "config"
        self.scripts_dir = self.project_root / "scripts"

        # PowerShell風ロガー初期化
        self.logger = PowerShellLogger()

        # チェックポイントマネージャー初期化
        if checkpoint_dir is None:
            checkpoint_dir = self.project_root / "checkpoints" / "sunset_pipeline"
        self.checkpoint_manager = SunsetPipelineCheckpointManager(checkpoint_dir)
        self.resume = resume
        
        # シグナルハンドラー設定（緊急保存用）
        self._setup_signal_handlers()

        # パイプライン設定
        self.pipeline_phases = {
            'data': {'duration': 1800, 'description': 'Data Pipeline Processing'},
            'training': {'duration': 7200, 'description': 'Unsloth SO8T Training'},
            'evaluation': {'duration': 3600, 'description': 'Benchmark Evaluation'},
            'abc': {'duration': 10800, 'description': 'ABC Comparative Testing'}
        }

        # Load configuration files
        self.load_configs()
        
        # チェックポイントから復旧
        if self.resume:
            self._load_checkpoint()
    
    def _setup_signal_handlers(self):
        """シグナルハンドラーを設定（緊急保存用）"""
        def signal_handler(signum, frame):
            self.logger.warning(f"Received signal {signum}, saving emergency checkpoint...")
            self.checkpoint_manager.emergency_save()
            sys.exit(1)
        
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, 'SIGBREAK'):  # Windows
            signal.signal(signal.SIGBREAK, signal_handler)
    
    def _load_checkpoint(self) -> Optional[Dict]:
        """チェックポイントを読み込み、復旧情報を表示"""
        checkpoint_data = self.checkpoint_manager.load_checkpoint()
        if checkpoint_data:
            self.logger.info("=" * 80)
            self.logger.info("CHECKPOINT FOUND - RESUMING PIPELINE")
            self.logger.info("=" * 80)
            self.logger.info(f"Session ID: {checkpoint_data.get('session_id', 'N/A')}")
            self.logger.info(f"Start Time: {checkpoint_data.get('start_time', 'N/A')}")
            self.logger.info(f"Current Phase: {checkpoint_data.get('current_phase', 'N/A')}")
            
            phases = checkpoint_data.get('phases', {})
            self.logger.info("\nPhase Status:")
            for phase_name, phase_info in phases.items():
                status = phase_info.get('status', 'pending')
                progress = phase_info.get('progress', 0)
                self.logger.info(f"  - {phase_name}: {status} ({progress}%)")
            
            self.logger.info("=" * 80)
            return checkpoint_data
        else:
            self.logger.info("No checkpoint found, starting fresh pipeline")
            return None
    
    def _is_phase_completed(self, phase_name: str) -> bool:
        """フェーズが完了済みかチェック"""
        checkpoint_data = self.checkpoint_manager.pipeline_state
        phases = checkpoint_data.get('phases', {})
        if phase_name in phases:
            return phases[phase_name].get('status') == 'completed'
        return False

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
                f"Processing datasets...", phase_name='data'
            )
            if success:
                self.logger.success(f"{phase_config['description']} completed")
                return True
            else:
                self.logger.error(f"{phase_config['description']} failed")
                return False
        else:
            self.logger.error("Data pipeline script not found")
            return False

    def run_model_training(self):
        """Run advanced SO8T Quadrality training with Unsloth and PowerShell progress"""
        phase_config = self.pipeline_phases['training']
        self.logger.info(f"Starting {phase_config['description']}")
        self.logger.info("Techniques: SO8T + DeepSeek GRPO + MHC + imatrix + Lightning Fast Training")

        training_script = self.scripts_dir / "training" / "train_unsloth_so8t.py"
        if training_script.exists():
            success = self._run_script_with_progress(
                str(training_script), ["--phase", "full"], phase_config['duration'],
                "Unsloth SO8T training in progress...", phase_name='training'
            )
            if success:
                self.logger.success(f"{phase_config['description']} completed")
                return True
            else:
                self.logger.warning("Unsloth training failed, trying fallback")
                return self._run_fallback_training()
        else:
            self.logger.error("Unsloth training script not found, using fallback")
            return self._run_fallback_training()

    def _run_fallback_training(self):
        """Fallback training method"""
        fallback_script = self.scripts_dir / "training" / "train_quadrality_model.py"
        if fallback_script.exists():
            phase_config = self.pipeline_phases['training']
            success = self._run_script_with_progress(
                str(fallback_script), ["--phase", "full"], phase_config['duration'],
                "Standard SO8T training (fallback)...", phase_name='training'
            )
            if success:
                self.logger.success("Fallback training completed")
                return True
            else:
                self.logger.error("Fallback training failed")
                return False
        else:
            self.logger.error("Fallback training script not found")
            return False

    def run_evaluation(self):
        """Run evaluation with PowerShell progress"""
        phase_config = self.pipeline_phases['evaluation']
        self.logger.info(f"Starting {phase_config['description']}")

        eval_script = self.scripts_dir / "evaluation" / "run_benchmarks.py"
        if eval_script.exists():
            success = self._run_script_with_progress(
                str(eval_script), [], phase_config['duration'],
                "Running benchmark evaluations...", phase_name='evaluation'
            )
            if success:
                self.logger.success(f"{phase_config['description']} completed")
                return True
            else:
                self.logger.error(f"{phase_config['description']} failed")
                return False
        else:
            self.logger.error("Evaluation script not found")
            return False

    def run_abc_testing(self):
        """Run ABC testing with PowerShell progress"""
        phase_config = self.pipeline_phases['abc']
        self.logger.info(f"Starting {phase_config['description']}")
        self.logger.info("Comparing: A(Qwen-base) vs B(SO8T-trained) vs C(AEGIS-Phi3.5)")

        abc_script = self.scripts_dir / "evaluation" / "abc_testing.py"
        if abc_script.exists():
            success = self._run_script_with_progress(
                str(abc_script), [], phase_config['duration'],
                "ABC comparative testing in progress...", phase_name='abc'
            )
            if success:
                self.logger.success(f"{phase_config['description']} completed")
                return True
            else:
                self.logger.error(f"{phase_config['description']} failed")
                return False
        else:
            self.logger.error("ABC testing script not found")
            return False

    def _run_script_with_progress(self, script_path: str, args: list, estimated_duration: int, description: str, phase_name: Optional[str] = None) -> bool:
        """PowerShell風にスクリプトを実行し、進捗を表示"""
        try:
            # 進捗バー初期化
            progress_bar = PowerShellProgressBar(total=100, desc=description)

            # サブプロセス開始
            cmd = [sys.executable, script_path] + args
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            start_time = time.time()
            last_progress = 0
            last_checkpoint_update = time.time()
            output_lines = []
            error_lines = []

            # リアルタイム監視
            while process.poll() is None:
                time.sleep(1)  # 1秒ごとにチェック
                elapsed = time.time() - start_time

                # 出力を読み取り（ノンブロッキング）
                if process.stdout:
                    try:
                        line = process.stdout.readline()
                        if line:
                            output_lines.append(line.strip())
                            # 重要なログのみ表示
                            if any(keyword in line.lower() for keyword in ['error', 'warning', 'exception', 'traceback']):
                                self.logger.info(line.strip())
                    except:
                        pass

                # 推定進捗計算（実際の進捗は取得できないので時間ベース）
                if elapsed < estimated_duration:
                    progress = min(95, int((elapsed / estimated_duration) * 100))
                else:
                    progress = 95  # 推定時間を超えても95%まで

                if progress > last_progress:
                    progress_bar.update(progress - last_progress)
                    last_progress = progress
                    
                    # 進捗をチェックポイントに記録（5分間隔）
                    if phase_name and (time.time() - last_checkpoint_update) >= 300:
                        self.checkpoint_manager.update_phase_status(phase_name, 'running', progress)
                        last_checkpoint_update = time.time()

            # 残りの出力を読み取り
            if process.stdout:
                remaining = process.stdout.read()
                if remaining:
                    output_lines.extend(remaining.splitlines())
            
            if process.stderr:
                error_output = process.stderr.read()
                if error_output:
                    error_lines = error_output.splitlines()

            # 最終進捗
            if process.returncode == 0:
                progress_bar.update(100 - last_progress)
                return True
            else:
                # エラー詳細を表示
                self.logger.error(f"Script execution failed with code {process.returncode}")
                if error_lines:
                    self.logger.error("Error output:")
                    for line in error_lines[-10:]:  # 最後の10行
                        self.logger.error(f"  {line}")
                if output_lines:
                    # エラー関連の行を表示
                    error_relevant = [line for line in output_lines if any(kw in line.lower() for kw in ['error', 'exception', 'traceback', 'failed'])]
                    if error_relevant:
                        self.logger.error("Relevant output:")
                        for line in error_relevant[-10:]:
                            self.logger.error(f"  {line}")
                return False

        except Exception as e:
            self.logger.error(f"Error running script: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def run_phase_with_timing(self, phase_name: str, skip_if_completed: bool = True):
        """指定フェーズを実行し、時間を計測"""
        # 完了済みフェーズのスキップチェック
        if skip_if_completed and self._is_phase_completed(phase_name):
            self.logger.info(f"=== PHASE {phase_name.upper()} SKIPPED (already completed) ===")
            return True
        
        phase_config = self.pipeline_phases[phase_name]

        self.logger.info(f"=== PHASE {phase_name.upper()} START ===")
        self.logger.info(f"Duration: ~{phase_config['duration']//3600}h {(phase_config['duration']%3600)//60}m")
        self.logger.info(f"Description: {phase_config['description']}")

        # フェーズ開始をチェックポイントに記録
        start_time_str = datetime.now().isoformat()
        self.checkpoint_manager.update_phase_status(
            phase_name, 'running', progress=0, start_time=start_time_str
        )
        self.checkpoint_manager.save_checkpoint()

        start_time = time.time()
        success = False

        try:
            if phase_name == 'data':
                success = self.run_data_pipeline()
            elif phase_name == 'training':
                success = self.run_model_training()
            elif phase_name == 'evaluation':
                success = self.run_evaluation()
            elif phase_name == 'abc':
                success = self.run_abc_testing()
            
            elapsed = time.time() - start_time
            elapsed_str = f"{int(elapsed//3600):02d}:{int((elapsed%3600)//60):02d}:{int(elapsed%60):02d}"
            
            # フェーズ終了をチェックポイントに記録
            end_time_str = datetime.now().isoformat()
            if success:
                self.checkpoint_manager.update_phase_status(
                    phase_name, 'completed', progress=100, end_time=end_time_str
                )
                self.logger.info(f"=== PHASE {phase_name.upper()} END ===")
                self.logger.info(f"Actual duration: {elapsed_str}")
            else:
                self.checkpoint_manager.update_phase_status(
                    phase_name, 'failed', progress=0, end_time=end_time_str
                )
                self.logger.error(f"=== PHASE {phase_name.upper()} FAILED ===")
            
            self.checkpoint_manager.save_checkpoint()
            return success
            
        except KeyboardInterrupt:
            self.logger.warning(f"Phase {phase_name} interrupted by user")
            self.checkpoint_manager.emergency_save()
            raise
        except Exception as e:
            self.logger.error(f"Phase {phase_name} failed with error: {e}")
            end_time_str = datetime.now().isoformat()
            self.checkpoint_manager.update_phase_status(
                phase_name, 'failed', progress=0, end_time=end_time_str
            )
            self.checkpoint_manager.save_checkpoint()
            return False

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

        # パイプライン開始をチェックポイントに記録
        start_time_str = datetime.now().isoformat()
        self.checkpoint_manager.pipeline_state['start_time'] = start_time_str
        self.checkpoint_manager.save_checkpoint()

        pipeline_start = time.time()
        total_phases = len(self.pipeline_phases)

        try:
            # Phase 1: Data preparation
            self.logger.info("\n[PHASE 1/4] Data Pipeline")
            if not self.run_phase_with_timing('data'):
                self.logger.error("Pipeline failed at Phase 1")
                return False

            # Phase 2: Model training
            self.logger.info("\n[PHASE 2/4] Model Training")
            if not self.run_phase_with_timing('training'):
                self.logger.error("Pipeline failed at Phase 2")
                return False

            # Phase 3: Evaluation
            self.logger.info("\n[PHASE 3/4] Benchmark Evaluation")
            if not self.run_phase_with_timing('evaluation'):
                self.logger.error("Pipeline failed at Phase 3")
                return False

            # Phase 4: ABC testing
            self.logger.info("\n[PHASE 4/4] ABC Comparative Testing")
            if not self.run_phase_with_timing('abc'):
                self.logger.error("Pipeline failed at Phase 4")
                return False

            # 完了サマリー
            total_elapsed = time.time() - pipeline_start
            total_elapsed_str = f"{int(total_elapsed//3600):02d}:{int((total_elapsed%3600)//60):02d}:{int(total_elapsed%60):02d}"

            # パイプライン完了をチェックポイントに記録
            end_time_str = datetime.now().isoformat()
            self.checkpoint_manager.pipeline_state['end_time'] = end_time_str
            self.checkpoint_manager.pipeline_state['status'] = 'completed'
            self.checkpoint_manager.save_checkpoint()

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
            self.checkpoint_manager.emergency_save()
            return False
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            self.checkpoint_manager.emergency_save()
            return False

    def _show_pipeline_results(self):
        """パイプライン実行結果を表示"""
        self.logger.info("\n[PIPELINE RESULTS]")
        self.logger.info("-" * 40)

        # モデル確認
        models_dir = self.project_root / "models"
        if (models_dir / "unsloth_so8t_qwen_7b_final").exists():
            self.logger.success("[OK] Unsloth SO8T trained model: Available")
        else:
            self.logger.warning("[WARN] Unsloth SO8T trained model: Not found")

        # 評価結果確認
        results_dir = self.project_root / "results"
        if (results_dir / "benchmarks").exists():
            benchmark_files = list((results_dir / "benchmarks").glob("*.json"))
            self.logger.success(f"[OK] Benchmark results: {len(benchmark_files)} files")

        if (results_dir / "abc_testing").exists():
            abc_files = list((results_dir / "abc_testing").glob("*.json"))
            self.logger.success(f"[OK] ABC testing results: {len(abc_files)} files")

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
    parser.add_argument('--resume', action='store_true', default=True,
                       help='Resume from checkpoint (default: True)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Ignore checkpoint and start fresh')
    parser.add_argument('--checkpoint-dir', type=Path,
                       help='Checkpoint directory (default: checkpoints/sunset_pipeline)')

    args = parser.parse_args()

    # システム要件チェック
    check_system_requirements()

    # 復旧オプションの処理
    resume = args.resume and not args.no_resume

    # パイプライン初期化
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else None
    pipeline = SunsetPipelineRTX3060(checkpoint_dir=checkpoint_dir, resume=resume)

    if args.config:
        pipeline.config_dir = Path(args.config)

    # 実行
    if args.phase == 'data':
        pipeline.run_phase_with_timing('data', skip_if_completed=resume)
    elif args.phase == 'training':
        pipeline.run_phase_with_timing('training', skip_if_completed=resume)
    elif args.phase == 'evaluation':
        pipeline.run_phase_with_timing('evaluation', skip_if_completed=resume)
    elif args.phase == 'abc':
        pipeline.run_phase_with_timing('abc', skip_if_completed=resume)
    elif args.phase == 'full':
        pipeline.run_full_pipeline()

if __name__ == "__main__":
    main()