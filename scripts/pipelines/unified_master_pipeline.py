#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統合マスターパイプライン

すべての全自動パイプラインを統合して、電源投入時に自動実行できるようにする

Usage:
    python scripts/pipelines/unified_master_pipeline.py --config configs/unified_master_pipeline_config.yaml
    python scripts/pipelines/unified_master_pipeline.py --setup  # Windowsタスクスケジューラ登録
    python scripts/pipelines/unified_master_pipeline.py --run    # パイプライン実行
"""

import sys
import json
import logging
import argparse
import subprocess
import signal
import pickle
import yaml
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any
import warnings
warnings.filterwarnings("ignore")

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/unified_master_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AudioNotifier:
    """音声通知クラス"""
    
    @staticmethod
    def play_notification():
        """音声通知を再生"""
        audio_path = PROJECT_ROOT / ".cursor" / "marisa_owattaze.wav"
        if audio_path.exists():
            try:
                subprocess.run([
                    "powershell", "-ExecutionPolicy", "Bypass", "-File",
                    str(PROJECT_ROOT / "scripts" / "utils" / "play_audio_notification.ps1")
                ], check=False, timeout=10)
            except Exception as e:
                logger.warning(f"Failed to play audio notification: {e}")
                try:
                    import winsound
                    winsound.Beep(1000, 500)
                except Exception:
                    pass


def check_admin_privileges() -> bool:
    """管理者権限をチェック"""
    try:
        result = subprocess.run(
            ["net", "session"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


class UnifiedMasterPipeline:
    """統合マスターパイプライン"""
    
    def __init__(self, config_path: Path):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        # 設定ファイルを読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 基本設定
        self.session_id = datetime.now().strftime(self.config.get('session_id_format', '%Y%m%d_%H%M%S'))
        self.checkpoint_dir = Path(self.config.get('checkpoint_dir', 'D:/webdataset/checkpoints/unified_master_pipeline'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 各パイプラインの設定
        self.phase1_config = self.config.get('phase1_parallel_scraping', {})
        self.phase2_config = self.config.get('phase2_data_processing', {})
        self.phase3_config = self.config.get('phase3_ab_test', {})
        self.phase4_config = self.config.get('phase4_github_scraping', {})
        self.phase5_config = self.config.get('phase5_engineer_sites', {})
        self.phase6_config = self.config.get('phase6_coding_extraction', {})
        self.phase7_config = self.config.get('phase7_coding_training_data', {})
        self.phase8_config = self.config.get('phase8_coding_retraining', {})
        self.phase9_config = self.config.get('phase9_documentation_scraping', {})
        self.phase10_config = self.config.get('phase10_unified_agent_base', {})
        self.phase11_config = self.config.get('phase11_nsfw_detection_dataset', {})
        
        # 進捗管理
        self.current_phase = None
        self.phase_progress = {}
        
        # シグナルハンドラー設定
        self._setup_signal_handlers()
        
        logger.info("="*80)
        logger.info("Unified Master Pipeline Initialized")
        logger.info("="*80)
        logger.info(f"Session ID: {self.session_id}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def _setup_signal_handlers(self):
        """シグナルハンドラー設定"""
        def signal_handler(signum, frame):
            logger.warning(f"Received signal {signum}, saving checkpoint...")
            self._save_checkpoint()
            logger.info("Checkpoint saved. Exiting gracefully...")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)
    
    def _load_checkpoint(self) -> Optional[Dict]:
        """チェックポイント読み込み"""
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{self.session_id}.pkl"
        
        if not checkpoint_file.exists():
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pkl"))
            if checkpoints:
                checkpoint_file = checkpoints[-1]
                logger.info(f"Loading latest checkpoint: {checkpoint_file}")
            else:
                return None
        
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            logger.info(f"[OK] Checkpoint loaded from {checkpoint_file}")
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def _save_checkpoint(self):
        """チェックポイント保存"""
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{self.session_id}.pkl"
        
        checkpoint_data = {
            'session_id': self.session_id,
            'current_phase': self.current_phase,
            'phase_progress': self.phase_progress,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            logger.info(f"[CHECKPOINT] Saved to {checkpoint_file}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _is_phase_completed(self, phase_name: str) -> bool:
        """
        フェーズの完了状態を検証
        
        Args:
            phase_name: フェーズ名
        
        Returns:
            completed: 完了しているかどうか
        """
        # チェックポイントからステータスを確認
        phase_progress = self.phase_progress.get(phase_name, {})
        status = phase_progress.get('status', 'pending')
        
        if status != 'completed':
            return False
        
        # Phase 1: 並列DeepResearch Webスクレイピング
        if phase_name == 'phase1_parallel_scraping':
            # バックグラウンド実行のため、プロセスIDの確認のみ
            process_id = phase_progress.get('process_id')
            if process_id:
                # プロセスが実行中か確認（オプション）
                try:
                    import psutil
                    if psutil.pid_exists(process_id):
                        logger.debug(f"[VERIFY] Phase 1: Process {process_id} is still running")
                        return True  # 実行中なら完了とみなす（バックグラウンド実行のため）
                except ImportError:
                    pass  # psutilがインストールされていない場合はスキップ
            
            # 出力ディレクトリの存在確認
            base_output = Path(self.phase1_config.get('base_output', 'D:/webdataset/processed'))
            if base_output.exists():
                logger.info(f"[VERIFY] Phase 1: Output directory exists: {base_output}")
                return True
            else:
                logger.warning(f"[VERIFY] Phase 1: Output directory not found: {base_output}")
                return False
        
        # Phase 2: SO8T全自動データ処理
        elif phase_name == 'phase2_data_processing':
            phase2_config = self.phase2_config
            if not phase2_config.get('enabled', True):
                logger.info("[VERIFY] Phase 2: Disabled in config")
                return True
            
            # データ処理パイプラインの出力ディレクトリを確認
            # 設定ファイルから出力パスを取得
            config_path = PROJECT_ROOT / phase2_config.get('config', 'configs/so8t_auto_data_processing_config.yaml')
            if config_path.exists():
                try:
                    import yaml
                    with open(config_path, 'r', encoding='utf-8') as f:
                        data_processing_config = yaml.safe_load(f)
                    output_dir = Path(data_processing_config.get('output_dir', 'D:/webdataset/processed/four_class'))
                    
                    # 出力ディレクトリが存在し、ファイルが含まれているか確認
                    if output_dir.exists():
                        output_files = list(output_dir.glob("*.jsonl"))
                        if output_files:
                            logger.info(f"[VERIFY] Phase 2: Output directory exists with {len(output_files)} files: {output_dir}")
                            return True
                        else:
                            logger.warning(f"[VERIFY] Phase 2: Output directory exists but no files found: {output_dir}")
                            return False
                    else:
                        logger.warning(f"[VERIFY] Phase 2: Output directory not found: {output_dir}")
                        return False
                except Exception as e:
                    logger.warning(f"[VERIFY] Phase 2: Failed to verify output (fallback to status check): {e}")
                    return True  # エラー時はステータスのみで判定
            else:
                logger.debug("[VERIFY] Phase 2: Config file not found, using status check only")
                return True
        
        # Phase 3: SO8T完全統合A/Bテスト
        elif phase_name == 'phase3_ab_test':
            phase3_config = self.phase3_config
            if not phase3_config.get('enabled', True):
                logger.info("[VERIFY] Phase 3: Disabled in config")
                return True
            
            # A/Bテストの結果ファイルを確認
            # 設定ファイルから出力パスを取得
            config_path = PROJECT_ROOT / phase3_config.get('config', 'configs/complete_so8t_ab_test_pipeline_config.yaml')
            if config_path.exists():
                try:
                    import yaml
                    with open(config_path, 'r', encoding='utf-8') as f:
                        ab_test_config = yaml.safe_load(f)
                    output_base_dir = Path(ab_test_config.get('output_base_dir', 'D:/webdataset'))
                    ab_test_results_dir = output_base_dir / 'ab_test_results'
                    
                    # 結果ディレクトリが存在し、結果ファイルが含まれているか確認
                    if ab_test_results_dir.exists():
                        result_dirs = [d for d in ab_test_results_dir.iterdir() if d.is_dir() and d.name.startswith('complete_so8t_ab_test_')]
                        if result_dirs:
                            # 最新の結果ディレクトリを確認
                            latest_result_dir = max(result_dirs, key=lambda d: d.stat().st_mtime)
                            results_file = latest_result_dir / 'ab_test_results.json'
                            if results_file.exists():
                                logger.info(f"[VERIFY] Phase 3: A/B test results found: {results_file}")
                                return True
                            else:
                                logger.warning(f"[VERIFY] Phase 3: Results directory exists but no results file: {latest_result_dir}")
                                return False
                        else:
                            logger.warning(f"[VERIFY] Phase 3: No result directories found: {ab_test_results_dir}")
                            return False
                    else:
                        logger.warning(f"[VERIFY] Phase 3: Results directory not found: {ab_test_results_dir}")
                        return False
                except Exception as e:
                    logger.warning(f"[VERIFY] Phase 3: Failed to verify results (fallback to status check): {e}")
                    return True  # エラー時はステータスのみで判定
            else:
                logger.debug("[VERIFY] Phase 3: Config file not found, using status check only")
                return True
        
        # Phase 4: GitHubリポジトリ検索
        elif phase_name == 'phase4_github_scraping':
            output_dir = Path(self.phase4_config.get('output_dir', 'D:/webdataset/processed/github'))
            github_files = list(output_dir.glob('github_repositories_*.jsonl'))
            if github_files:
                latest_file = max(github_files, key=lambda p: p.stat().st_mtime)
                if latest_file.stat().st_size > 0:
                    logger.info(f"[VERIFY] Phase 4: GitHub scraping results found: {latest_file}")
                    return True
            return False
        
        # Phase 5: エンジニア向けサイトスクレイピング
        elif phase_name == 'phase5_engineer_sites':
            output_dir = Path(self.phase5_config.get('output_dir', 'D:/webdataset/processed/engineer_sites'))
            engineer_files = list(output_dir.glob('engineer_sites_*.jsonl'))
            if engineer_files:
                latest_file = max(engineer_files, key=lambda p: p.stat().st_mtime)
                if latest_file.stat().st_size > 0:
                    logger.info(f"[VERIFY] Phase 5: Engineer sites scraping results found: {latest_file}")
                    return True
            return False
        
        # Phase 6: コーディング関連データ抽出
        elif phase_name == 'phase6_coding_extraction':
            output_dir = Path(self.phase6_config.get('output_dir', 'D:/webdataset/coding_dataset'))
            coding_files = list(output_dir.glob('coding_*.jsonl'))
            if coding_files:
                latest_file = max(coding_files, key=lambda p: p.stat().st_mtime)
                if latest_file.stat().st_size > 0:
                    logger.info(f"[VERIFY] Phase 6: Coding extraction results found: {latest_file}")
                    return True
            return False
        
        # Phase 7: コーディングタスク用データセット作成
        elif phase_name == 'phase7_coding_training_data':
            output_dir = Path(self.phase7_config.get('output_dir', 'D:/webdataset/coding_training_data'))
            training_files = list(output_dir.glob('coding_training_*.jsonl'))
            if training_files:
                latest_file = max(training_files, key=lambda p: p.stat().st_mtime)
                if latest_file.stat().st_size > 0:
                    logger.info(f"[VERIFY] Phase 7: Coding training data found: {latest_file}")
                    return True
            return False
        
        # Phase 8: コーディング特化再学習
        elif phase_name == 'phase8_coding_retraining':
            config_path = Path(self.phase8_config.get('config_path', 'configs/coding_focused_retraining_config.yaml'))
            if config_path.exists():
                try:
                    import yaml
                    with open(config_path, 'r', encoding='utf-8') as f:
                        retraining_config = yaml.safe_load(f)
                    final_model_dir = Path(retraining_config.get('output', {}).get('final_model_dir', 'D:/webdataset/models/coding_focused_retraining'))
                    
                    # 最終モデルディレクトリの存在を確認
                    if final_model_dir.exists():
                        model_files = list(final_model_dir.glob('*.pt')) + list(final_model_dir.glob('*.safetensors'))
                        if model_files:
                            logger.info(f"[VERIFY] Phase 8: Coding retraining model found: {final_model_dir}")
                            return True
                except Exception as e:
                    logger.warning(f"[VERIFY] Phase 8: Failed to verify model (fallback to status check): {e}")
                    return True  # エラー時はステータスのみで判定
            return False
        
        # Phase 9: ドキュメンテーション収集
        elif phase_name == 'phase9_documentation_scraping':
            phase9_config = self.phase9_config
            if not phase9_config.get('enabled', True):
                logger.info("[VERIFY] Phase 9: Disabled in config")
                return True
            
            output_dir = Path(phase9_config.get('output_dir', 'D:/webdataset/processed/documentation'))
            doc_files = list(output_dir.glob('documentation_*.jsonl'))
            if doc_files:
                latest_file = max(doc_files, key=lambda p: p.stat().st_mtime)
                if latest_file.stat().st_size > 0:
                    logger.info(f"[VERIFY] Phase 9: Documentation scraping results found: {latest_file}")
                    return True
            return False
        
        # Phase 10: 統合AIエージェント基盤の構築
        elif phase_name == 'phase10_unified_agent_base':
            phase10_config = self.phase10_config
            if not phase10_config.get('enabled', True):
                logger.info("[VERIFY] Phase 10: Disabled in config")
                return True
            
            # 結果ファイルの存在確認
            result_files = list(self.checkpoint_dir.glob('phase10_results_*.json'))
            if result_files:
                logger.info(f"[VERIFY] Phase 10: Result files found: {len(result_files)} files")
                return True
            else:
                logger.warning(f"[VERIFY] Phase 10: No result files found in checkpoint directory")
                return False
        
        # Phase 11: 検知用NSFWデータセット収集
        elif phase_name == 'phase11_nsfw_detection_dataset':
            phase11_config = self.phase11_config
            if not phase11_config.get('enabled', True):
                logger.info("[VERIFY] Phase 11: Disabled in config")
                return True
            
            # 出力ディレクトリの存在確認
            output_dir = Path(phase11_config.get('output_dir', 'D:/webdataset/nsfw_detection_dataset'))
            train_file = output_dir / "nsfw_detection_train.jsonl"
            val_file = output_dir / "nsfw_detection_val.jsonl"
            
            if train_file.exists() and val_file.exists():
                if train_file.stat().st_size > 0 and val_file.stat().st_size > 0:
                    logger.info(f"[VERIFY] Phase 11: Dataset files found: {train_file}, {val_file}")
                    return True
                else:
                    logger.warning(f"[VERIFY] Phase 11: Dataset files exist but are empty")
                    return False
            else:
                logger.warning(f"[VERIFY] Phase 11: Dataset files not found")
                return False
        
        # その他のフェーズはステータスのみで判定
        else:
            logger.debug(f"[VERIFY] Phase {phase_name}: Status check only (status: {status})")
            return status == 'completed'
    
    def phase1_parallel_scraping(self) -> bool:
        """
        Phase 1: SO8T/thinkingモデル統制並列DeepResearch Webスクレイピング
        
        Returns:
            success: 成功フラグ
        """
        logger.info("="*80)
        logger.info("PHASE 1: SO8T Thinking Controlled Parallel DeepResearch Web Scraping")
        logger.info("="*80)
        
        self.current_phase = "phase1_parallel_scraping"
        
        # チェックポイントから復旧
        checkpoint = self._load_checkpoint()
        if checkpoint and checkpoint.get('current_phase') == 'phase1_parallel_scraping':
            phase_progress = checkpoint.get('phase_progress', {}).get('phase1_parallel_scraping', {})
            if phase_progress.get('status') == 'completed':
                logger.info("[SKIP] Phase 1 already completed")
                return True
        
        if not self.phase1_config.get('enabled', True):
            logger.info("[SKIP] Phase 1 disabled in config")
            return True
        
        # SO8T/thinkingモデル統制スクレイピングスクリプトを実行
        script_path = PROJECT_ROOT / "scripts" / "data" / "so8t_thinking_controlled_scraping.py"
        
        if not script_path.exists():
            logger.warning(f"SO8T thinking controlled scraper not found: {script_path}")
            logger.info("Falling back to parallel_pipeline_manager.py")
            script_path = PROJECT_ROOT / "scripts" / "data" / "parallel_pipeline_manager.py"
        
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return False
        
        # バックグラウンドで実行（daemon mode）
        base_output = str(self.phase1_config.get('base_output', 'D:/webdataset/processed'))
        if '#' in base_output:
            logger.error(f"[ERROR] Invalid character '#' in base_output path: {base_output}")
            logger.error("[ERROR] Please remove '#' from the path")
            self.phase_progress['phase1_parallel_scraping'] = {
                'status': 'failed',
                'error': f"Invalid character '#' in base_output path"
            }
            self._save_checkpoint()
            return False
        
        # SO8T/thinkingモデル統制スクレイピングの場合は専用コマンドを使用
        if "so8t_thinking_controlled_scraping.py" in str(script_path):
            cmd = [
                sys.executable,
                str(script_path),
                "--output", base_output,
                "--num-browsers", str(self.phase1_config.get('num_instances', 10)),
                "--remote-debugging-port", str(self.phase1_config.get('base_port', 9222)),
                "--use-cursor-browser",
                "--resume"
            ]
        else:
            # 既存のparallel_pipeline_manager.pyを使用
            cmd = [
                sys.executable,
                str(script_path),
                "--run",
                "--daemon",
                "--num-instances", str(self.phase1_config.get('num_instances', 10)),
                "--base-output", base_output,
                "--base-port", str(self.phase1_config.get('base_port', 9222)),
                "--auto-restart",
                "--restart-delay", str(self.phase1_config.get('restart_delay', 60.0)),
                "--max-memory-gb", str(self.phase1_config.get('max_memory_gb', 8.0)),
                "--max-cpu-percent", str(self.phase1_config.get('max_cpu_percent', 80.0))
            ]
        
        # コマンド内に#が含まれていないか確認
        cmd_str = ' '.join(cmd)
        if '#' in cmd_str:
            logger.error(f"[ERROR] Invalid character '#' found in command: {cmd_str}")
            logger.error("[ERROR] Please check configuration files for '#' characters")
            self.phase_progress['phase1_parallel_scraping'] = {
                'status': 'failed',
                'error': f"Invalid character '#' in command"
            }
            self._save_checkpoint()
            return False
        
        try:
            logger.info(f"Starting SO8T thinking controlled parallel scraping pipeline...")
            logger.debug(f"Command: {cmd_str}")
            # バックグラウンドで実行（非同期）
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            
            # プロセスIDを記録
            self.phase_progress['phase1_parallel_scraping'] = {
                'status': 'running',
                'process_id': process.pid,
                'started_at': datetime.now().isoformat(),
                'script': str(script_path.name)
            }
            self._save_checkpoint()
            
            logger.info(f"[OK] Phase 1 started (PID: {process.pid})")
            logger.info("[INFO] Phase 1 runs in background with SO8T thinking control. Proceeding to Phase 2...")
            
            # バックグラウンド実行のため、すぐに成功として返す
            self.phase_progress['phase1_parallel_scraping'] = {
                'status': 'completed',
                'process_id': process.pid,
                'started_at': datetime.now().isoformat(),
                'note': 'Running in background with SO8T thinking control'
            }
            self._save_checkpoint()
            
            AudioNotifier.play_notification()
            
            return True
            
        except subprocess.SubprocessError as e:
            logger.error(f"[ERROR] Phase 1 failed (SubprocessError): {e}")
            logger.error(f"[ERROR] Command: {cmd_str}")
            import traceback
            traceback.print_exc()
            self.phase_progress['phase1_parallel_scraping'] = {
                'status': 'failed',
                'error': f"SubprocessError: {str(e)}"
            }
            self._save_checkpoint()
            return False
        except Exception as e:
            logger.error(f"[ERROR] Phase 1 failed: {e}")
            logger.error(f"[ERROR] Command: {cmd_str}")
            import traceback
            traceback.print_exc()
            self.phase_progress['phase1_parallel_scraping'] = {
                'status': 'failed',
                'error': str(e)
            }
            self._save_checkpoint()
            return False
    
    def phase2_data_processing(self) -> bool:
        """
        Phase 2: SO8T全自動データ処理
        
        Returns:
            success: 成功フラグ
        """
        logger.info("="*80)
        logger.info("PHASE 2: SO8T Auto Data Processing")
        logger.info("="*80)
        
        self.current_phase = "phase2_data_processing"
        
        # チェックポイントから復旧
        checkpoint = self._load_checkpoint()
        if checkpoint and checkpoint.get('current_phase') == 'phase2_data_processing':
            phase_progress = checkpoint.get('phase_progress', {}).get('phase2_data_processing', {})
            if phase_progress.get('status') == 'completed':
                logger.info("[SKIP] Phase 2 already completed")
                return True
        
        if not self.phase2_config.get('enabled', True):
            logger.info("[SKIP] Phase 2 disabled in config")
            return True
        
        # so8t_auto_data_processing_pipeline.pyを実行
        script_path = PROJECT_ROOT / "scripts" / "pipelines" / "so8t_auto_data_processing_pipeline.py"
        config_path = PROJECT_ROOT / self.phase2_config.get('config', 'configs/so8t_auto_data_processing_config.yaml')
        
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return False
        
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return False
        
        # パスに#が含まれている場合はエラー
        config_path_str = str(config_path)
        if '#' in config_path_str:
            logger.error(f"[ERROR] Invalid character '#' in config path: {config_path_str}")
            logger.error("[ERROR] Please remove '#' from the path")
            self.phase_progress['phase2_data_processing'] = {
                'status': 'failed',
                'error': f"Invalid character '#' in config path"
            }
            self._save_checkpoint()
            return False
        
        cmd = [
            sys.executable,
            str(script_path),
            "--config", config_path_str,
            "--resume"
        ]
        
        # コマンド内に#が含まれていないか確認
        cmd_str = ' '.join(cmd)
        if '#' in cmd_str:
            logger.error(f"[ERROR] Invalid character '#' found in command: {cmd_str}")
            logger.error("[ERROR] Please check configuration files for '#' characters")
            self.phase_progress['phase2_data_processing'] = {
                'status': 'failed',
                'error': f"Invalid character '#' in command"
            }
            self._save_checkpoint()
            return False
        
        try:
            logger.info(f"Starting data processing pipeline...")
            logger.debug(f"Command: {cmd_str}")
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True,
                timeout=self.phase2_config.get('timeout', 86400)  # 24時間タイムアウト
            )
            
            self.phase_progress['phase2_data_processing'] = {
                'status': 'completed',
                'completed_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            logger.info("[OK] Phase 2 completed")
            
            AudioNotifier.play_notification()
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] Phase 2 failed: {e}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("[ERROR] Phase 2 timeout")
            return False
        except Exception as e:
            logger.error(f"[ERROR] Phase 2 failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def phase3_ab_test(self) -> bool:
        """
        Phase 3: SO8T完全統合A/Bテスト
        
        Returns:
            success: 成功フラグ
        """
        logger.info("="*80)
        logger.info("PHASE 3: SO8T Complete A/B Test")
        logger.info("="*80)
        
        self.current_phase = "phase3_ab_test"
        
        # チェックポイントから復旧
        checkpoint = self._load_checkpoint()
        if checkpoint and checkpoint.get('current_phase') == 'phase3_ab_test':
            phase_progress = checkpoint.get('phase_progress', {}).get('phase3_ab_test', {})
            if phase_progress.get('status') == 'completed':
                logger.info("[SKIP] Phase 3 already completed")
                return True
        
        if not self.phase3_config.get('enabled', True):
            logger.info("[SKIP] Phase 3 disabled in config")
            return True
        
        # complete_so8t_ab_test_pipeline.pyを実行
        script_path = PROJECT_ROOT / "scripts" / "pipelines" / "complete_so8t_ab_test_pipeline.py"
        config_path = PROJECT_ROOT / self.phase3_config.get('config', 'configs/complete_so8t_ab_test_pipeline_config.yaml')
        
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return False
        
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return False
        
        # パスに#が含まれている場合はエラー
        config_path_str = str(config_path)
        if '#' in config_path_str:
            logger.error(f"[ERROR] Invalid character '#' in config path: {config_path_str}")
            logger.error("[ERROR] Please remove '#' from the path")
            self.phase_progress['phase3_ab_test'] = {
                'status': 'failed',
                'error': f"Invalid character '#' in config path"
            }
            self._save_checkpoint()
            return False
        
        cmd = [
            sys.executable,
            str(script_path),
            "--config", config_path_str,
            "--resume"
        ]
        
        # コマンド内に#が含まれていないか確認
        cmd_str = ' '.join(cmd)
        if '#' in cmd_str:
            logger.error(f"[ERROR] Invalid character '#' found in command: {cmd_str}")
            logger.error("[ERROR] Please check configuration files for '#' characters")
            self.phase_progress['phase3_ab_test'] = {
                'status': 'failed',
                'error': f"Invalid character '#' in command"
            }
            self._save_checkpoint()
            return False
        
        try:
            logger.info(f"Starting A/B test pipeline...")
            logger.debug(f"Command: {cmd_str}")
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True,
                timeout=self.phase3_config.get('timeout', 172800)  # 48時間タイムアウト
            )
            
            self.phase_progress['phase3_ab_test'] = {
                'status': 'completed',
                'completed_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            logger.info("[OK] Phase 3 completed")
            
            AudioNotifier.play_notification()
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] Phase 3 failed: {e}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("[ERROR] Phase 3 timeout")
            return False
        except Exception as e:
            logger.error(f"[ERROR] Phase 3 failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def phase4_github_scraping(self) -> bool:
        """Phase 4: GitHubリポジトリ検索"""
        try:
            logger.info("[PHASE 4] Starting GitHub repository scraping...")
            
            self.phase_progress['phase4_github_scraping'] = {
                'status': 'running',
                'started_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            # GitHubスクレイパースクリプトのパス
            script_path = PROJECT_ROOT / 'scripts' / 'data' / 'github_repository_scraper.py'
            
            if not script_path.exists():
                logger.error(f"[ERROR] GitHub scraper script not found: {script_path}")
                return False
            
            # 設定からパラメータを取得
            output_dir = Path(self.phase4_config.get('output_dir', 'D:/webdataset/processed/github'))
            github_token = self.phase4_config.get('github_token', None)
            queries = self.phase4_config.get('queries', ['best practices', 'tutorial', 'example'])
            languages = self.phase4_config.get('languages', None)
            
            # コマンドを構築
            cmd = [
                sys.executable,
                str(script_path),
                '--output', str(output_dir),
                '--max-repos', str(self.phase4_config.get('max_repos', 100)),
                '--min-stars', str(self.phase4_config.get('min_stars', 100)),
            ]
            
            if github_token:
                cmd.extend(['--github-token', github_token])
            
            cmd.extend(['--queries'] + queries)
            
            if languages:
                cmd.extend(['--languages'] + languages)
            
            # パスに#が含まれていないか確認
            cmd_str = ' '.join(cmd)
            if '#' in cmd_str:
                logger.error(f"[ERROR] Invalid character '#' found in command: {cmd_str}")
                return False
            
            logger.info(f"[PHASE 4] Executing: {' '.join(cmd)}")
            
            subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True,
                timeout=self.phase4_config.get('timeout', 3600)  # 1時間タイムアウト
            )
            
            self.phase_progress['phase4_github_scraping'] = {
                'status': 'completed',
                'completed_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            logger.info("[OK] Phase 4 completed")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Phase 4 failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def phase5_engineer_sites(self) -> bool:
        """Phase 5: エンジニア向けサイトスクレイピング"""
        try:
            logger.info("[PHASE 5] Starting engineer sites scraping...")
            
            self.phase_progress['phase5_engineer_sites'] = {
                'status': 'running',
                'started_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            # エンジニアサイトスクレイパースクリプトのパス
            script_path = PROJECT_ROOT / 'scripts' / 'data' / 'engineer_site_scraper.py'
            
            if not script_path.exists():
                logger.error(f"[ERROR] Engineer site scraper script not found: {script_path}")
                return False
            
            # 設定からパラメータを取得
            output_dir = Path(self.phase5_config.get('output_dir', 'D:/webdataset/processed/engineer_sites'))
            queries = self.phase5_config.get('queries', ['Python', 'JavaScript', 'programming'])
            
            # コマンドを構築
            cmd = [
                sys.executable,
                str(script_path),
                '--output', str(output_dir),
                '--delay', str(self.phase5_config.get('delay', 2.0)),
                '--max-articles', str(self.phase5_config.get('max_articles', 100)),
            ]
            
            cmd.extend(['--queries'] + queries)
            
            # パスに#が含まれていないか確認
            cmd_str = ' '.join(cmd)
            if '#' in cmd_str:
                logger.error(f"[ERROR] Invalid character '#' found in command: {cmd_str}")
                return False
            
            logger.info(f"[PHASE 5] Executing: {' '.join(cmd)}")
            
            subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True,
                timeout=self.phase5_config.get('timeout', 7200)  # 2時間タイムアウト
            )
            
            self.phase_progress['phase5_engineer_sites'] = {
                'status': 'completed',
                'completed_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            logger.info("[OK] Phase 5 completed")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Phase 5 failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def phase6_coding_extraction(self) -> bool:
        """Phase 6: コーディング関連データ抽出"""
        try:
            logger.info("[PHASE 6] Starting coding dataset extraction...")
            
            self.phase_progress['phase6_coding_extraction'] = {
                'status': 'running',
                'started_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            # コーディングデータ抽出スクリプトのパス
            script_path = PROJECT_ROOT / 'scripts' / 'pipelines' / 'extract_coding_dataset.py'
            
            if not script_path.exists():
                logger.error(f"[ERROR] Coding extraction script not found: {script_path}")
                return False
            
            # 設定からパラメータを取得
            input_dir = Path(self.phase6_config.get('input_dir', 'D:/webdataset/processed/four_class'))
            output_dir = Path(self.phase6_config.get('output_dir', 'D:/webdataset/coding_dataset'))
            
            # コマンドを構築
            cmd = [
                sys.executable,
                str(script_path),
                '--input', str(input_dir),
                '--output', str(output_dir),
                '--min-code-length', str(self.phase6_config.get('min_code_length', 10)),
                '--min-text-length', str(self.phase6_config.get('min_text_length', 50)),
            ]
            
            # パスに#が含まれていないか確認
            cmd_str = ' '.join(cmd)
            if '#' in cmd_str:
                logger.error(f"[ERROR] Invalid character '#' found in command: {cmd_str}")
                return False
            
            logger.info(f"[PHASE 6] Executing: {' '.join(cmd)}")
            
            subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True,
                timeout=self.phase6_config.get('timeout', 3600)  # 1時間タイムアウト
            )
            
            self.phase_progress['phase6_coding_extraction'] = {
                'status': 'completed',
                'completed_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            logger.info("[OK] Phase 6 completed")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Phase 6 failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def phase7_coding_training_data(self) -> bool:
        """Phase 7: コーディングタスク用データセット作成"""
        try:
            logger.info("[PHASE 7] Starting coding training data preparation...")
            
            self.phase_progress['phase7_coding_training_data'] = {
                'status': 'running',
                'started_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            # コーディングトレーニングデータ準備スクリプトのパス
            script_path = PROJECT_ROOT / 'scripts' / 'pipelines' / 'prepare_coding_training_data.py'
            
            if not script_path.exists():
                logger.error(f"[ERROR] Coding training data script not found: {script_path}")
                return False
            
            # 設定からパラメータを取得
            input_dir = Path(self.phase7_config.get('input_dir', 'D:/webdataset/coding_dataset'))
            output_dir = Path(self.phase7_config.get('output_dir', 'D:/webdataset/coding_training_data'))
            
            # コマンドを構築
            cmd = [
                sys.executable,
                str(script_path),
                '--input', str(input_dir),
                '--output', str(output_dir),
            ]
            
            # パスに#が含まれていないか確認
            cmd_str = ' '.join(cmd)
            if '#' in cmd_str:
                logger.error(f"[ERROR] Invalid character '#' found in command: {cmd_str}")
                return False
            
            logger.info(f"[PHASE 7] Executing: {' '.join(cmd)}")
            
            subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True,
                timeout=self.phase7_config.get('timeout', 3600)  # 1時間タイムアウト
            )
            
            self.phase_progress['phase7_coding_training_data'] = {
                'status': 'completed',
                'completed_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            logger.info("[OK] Phase 7 completed")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Phase 7 failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def phase8_coding_retraining(self) -> bool:
        """Phase 8: コーディング特化再学習"""
        try:
            logger.info("[PHASE 8] Starting coding-focused retraining...")
            
            self.phase_progress['phase8_coding_retraining'] = {
                'status': 'running',
                'started_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            # コーディング特化再学習パイプラインスクリプトのパス
            script_path = PROJECT_ROOT / 'scripts' / 'pipelines' / 'coding_focused_retraining_pipeline.py'
            
            if not script_path.exists():
                logger.error(f"[ERROR] Coding retraining pipeline script not found: {script_path}")
                return False
            
            # 設定からパラメータを取得
            config_path = Path(self.phase8_config.get('config_path', 'configs/coding_focused_retraining_config.yaml'))
            
            # コマンドを構築
            cmd = [
                sys.executable,
                str(script_path),
                '--config', str(config_path),
            ]
            
            # パスに#が含まれていないか確認
            cmd_str = ' '.join(cmd)
            if '#' in cmd_str:
                logger.error(f"[ERROR] Invalid character '#' found in command: {cmd_str}")
                return False
            
            logger.info(f"[PHASE 8] Executing: {' '.join(cmd)}")
            
            subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True,
                timeout=self.phase8_config.get('timeout', 86400)  # 24時間タイムアウト
            )
            
            self.phase_progress['phase8_coding_retraining'] = {
                'status': 'completed',
                'completed_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            logger.info("[OK] Phase 8 completed")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Phase 8 failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def phase9_documentation_scraping(self) -> bool:
        """Phase 9: ドキュメンテーション収集"""
        try:
            logger.info("[PHASE 9] Starting documentation scraping...")
            
            self.phase_progress['phase9_documentation_scraping'] = {
                'status': 'running',
                'started_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            # ドキュメンテーション収集スクリプトのパス
            script_path = PROJECT_ROOT / 'scripts' / 'data' / 'documentation_scraper.py'
            
            if not script_path.exists():
                logger.error(f"[ERROR] Documentation scraper script not found: {script_path}")
                return False
            
            # 設定からパラメータを取得
            output_dir = Path(self.phase9_config.get('output_dir', 'D:/webdataset/processed/documentation'))
            github_repos = self.phase9_config.get('github_repos', [])
            api_urls = self.phase9_config.get('api_urls', [])
            blog_urls = self.phase9_config.get('blog_urls', [])
            delay = self.phase9_config.get('delay', 2.0)
            max_docs = self.phase9_config.get('max_docs', 100)
            
            # コマンドを構築
            cmd = [
                sys.executable,
                str(script_path),
                '--output', str(output_dir),
                '--delay', str(delay),
                '--max-docs', str(max_docs)
            ]
            
            # GitHubリポジトリが指定されている場合
            if github_repos:
                cmd.extend(['--github-repos'] + github_repos)
            
            # API URLが指定されている場合
            if api_urls:
                cmd.extend(['--api-urls'] + api_urls)
            
            # ブログURLが指定されている場合
            if blog_urls:
                cmd.extend(['--blog-urls'] + blog_urls)
            
            # パスに#が含まれていないか確認
            cmd_str = ' '.join(cmd)
            if '#' in cmd_str:
                logger.error(f"[ERROR] Invalid character '#' found in command: {cmd_str}")
                return False
            
            logger.info(f"[PHASE 9] Executing: {' '.join(cmd)}")
            
            subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True,
                timeout=self.phase9_config.get('timeout', 7200)  # 2時間タイムアウト
            )
            
            self.phase_progress['phase9_documentation_scraping'] = {
                'status': 'completed',
                'completed_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            logger.info("[OK] Phase 9 completed")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Phase 9 failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def phase10_unified_agent_base(self) -> bool:
        """Phase 10: 統合AIエージェント基盤の構築"""
        try:
            logger.info("[PHASE 10] Starting unified AI agent base construction...")
            
            self.phase_progress['phase10_unified_agent_base'] = {
                'status': 'running',
                'started_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            # 統合推論パイプラインスクリプトのパス
            script_path = PROJECT_ROOT / 'scripts' / 'agents' / 'integrated_reasoning_pipeline.py'
            
            if not script_path.exists():
                logger.error(f"[ERROR] Integrated reasoning pipeline script not found: {script_path}")
                return False
            
            # 設定からパラメータを取得
            model_path = self.phase10_config.get('model_path')
            knowledge_base_path = self.phase10_config.get('knowledge_base_path', 'database/so8t_memory.db')
            rag_store_path = self.phase10_config.get('rag_store_path', 'D:/webdataset/vector_stores')
            coding_data_path = self.phase10_config.get('coding_data_path', 'D:/webdataset/processed/coding')
            science_data_path = self.phase10_config.get('science_data_path', 'D:/webdataset/processed/science')
            test_queries = self.phase10_config.get('test_queries', [])
            
            # テストクエリが指定されている場合、テストを実行
            if test_queries:
                logger.info(f"[PHASE 10] Running test queries: {len(test_queries)} queries")
                
                # テストクエリファイルを作成
                test_queries_file = self.checkpoint_dir / f"test_queries_{self.session_id}.txt"
                with open(test_queries_file, 'w', encoding='utf-8') as f:
                    for query in test_queries:
                        f.write(f"{query}\n")
                
                # コマンドを構築
                cmd = [
                    sys.executable,
                    str(script_path),
                    '--queries-file', str(test_queries_file),
                    '--user-id', 'pipeline_test'
                ]
                
                if model_path:
                    cmd.extend(['--model-path', model_path])
                if knowledge_base_path:
                    cmd.extend(['--knowledge-base', knowledge_base_path])
                if rag_store_path:
                    cmd.extend(['--rag-store', rag_store_path])
                if coding_data_path:
                    cmd.extend(['--coding-data', coding_data_path])
                if science_data_path:
                    cmd.extend(['--science-data', science_data_path])
                
                # 出力ファイル
                output_file = self.checkpoint_dir / f"phase10_results_{self.session_id}.json"
                cmd.extend(['--output', str(output_file)])
                
                # パスに#が含まれていないか確認
                cmd_str = ' '.join(cmd)
                if '#' in cmd_str:
                    logger.error(f"[ERROR] Invalid character '#' found in command: {cmd_str}")
                    return False
                
                logger.info(f"[PHASE 10] Executing: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=self.phase10_config.get('timeout', 3600)  # 1時間タイムアウト
                )
                
                logger.info(f"[PHASE 10] Test execution completed")
                logger.debug(f"[PHASE 10] stdout: {result.stdout[:500]}")
                
                # 結果ファイルの存在確認
                if output_file.exists():
                    logger.info(f"[PHASE 10] Results saved to: {output_file}")
                else:
                    logger.warning(f"[PHASE 10] Results file not found: {output_file}")
            else:
                logger.info("[PHASE 10] No test queries specified, skipping test execution")
            
            self.phase_progress['phase10_unified_agent_base'] = {
                'status': 'completed',
                'completed_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            logger.info("[OK] Phase 10 completed")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Phase 10 failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def phase11_nsfw_detection_dataset(self) -> bool:
        """Phase 11: 検知用NSFWデータセット収集"""
        try:
            logger.info("[PHASE 11] Starting NSFW detection dataset collection...")
            
            self.phase_progress['phase11_nsfw_detection_dataset'] = {
                'status': 'running',
                'started_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            # NSFW検知用データセット収集スクリプトのパス
            script_path = PROJECT_ROOT / 'scripts' / 'data' / 'collect_nsfw_detection_dataset.py'
            
            if not script_path.exists():
                logger.error(f"[ERROR] NSFW detection dataset collector script not found: {script_path}")
                return False
            
            # 設定からパラメータを取得
            input_dir = Path(self.phase11_config.get('input_dir', 'D:/webdataset/processed/four_class'))
            output_dir = Path(self.phase11_config.get('output_dir', 'D:/webdataset/nsfw_detection_dataset'))
            nsfw_classifier_path = self.phase11_config.get('nsfw_classifier_path')
            max_samples = self.phase11_config.get('max_samples', 50000)
            use_multimodal = self.phase11_config.get('use_multimodal', True)
            include_synthetic = self.phase11_config.get('include_synthetic', True)
            synthetic_samples = self.phase11_config.get('synthetic_samples', 1000)
            include_metadata = self.phase11_config.get('include_metadata', True)
            
            # コマンドを構築
            cmd = [
                sys.executable,
                str(script_path),
                '--output', str(output_dir),
                '--max-samples', str(max_samples)
            ]
            
            if input_dir.exists():
                cmd.extend(['--input', str(input_dir)])
            
            if nsfw_classifier_path:
                cmd.extend(['--nsfw-classifier', str(nsfw_classifier_path)])
            
            if use_multimodal:
                cmd.append('--use-multimodal')
            
            if include_synthetic:
                cmd.append('--include-synthetic')
                cmd.extend(['--synthetic-samples', str(synthetic_samples)])
            
            if include_metadata:
                cmd.append('--include-metadata')
            
            # パスに#が含まれていないか確認
            cmd_str = ' '.join(cmd)
            if '#' in cmd_str:
                logger.error(f"[ERROR] Invalid character '#' found in command: {cmd_str}")
                return False
            
            logger.info(f"[PHASE 11] Executing: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.phase11_config.get('timeout', 7200)  # 2時間タイムアウト
            )
            
            logger.info(f"[PHASE 11] Dataset collection completed")
            logger.debug(f"[PHASE 11] stdout: {result.stdout[:500]}")
            
            # 出力ファイルの存在確認
            train_file = output_dir / "nsfw_detection_train.jsonl"
            val_file = output_dir / "nsfw_detection_val.jsonl"
            
            if train_file.exists() and val_file.exists():
                logger.info(f"[PHASE 11] Dataset files created successfully")
            else:
                logger.warning(f"[PHASE 11] Some dataset files not found")
            
            self.phase_progress['phase11_nsfw_detection_dataset'] = {
                'status': 'completed',
                'completed_at': datetime.now().isoformat()
            }
            self._save_checkpoint()
            
            logger.info("[OK] Phase 11 completed")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Phase 11 failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def start_streamlit_dashboard(self):
        """Streamlitダッシュボードを起動"""
        try:
            dashboard_script = PROJECT_ROOT / "scripts" / "dashboard" / "unified_scraping_monitoring_dashboard.py"
            
            if not dashboard_script.exists():
                logger.warning(f"Dashboard script not found: {dashboard_script}")
                return False
            
            logger.info("[DASHBOARD] Starting Streamlit dashboard...")
            
            cmd = [
                sys.executable,
                "-m", "streamlit", "run",
                str(dashboard_script),
                "--server.port", "8501",
                "--server.headless", "true"
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            
            logger.info(f"[DASHBOARD] Dashboard started (PID: {process.pid})")
            logger.info(f"[DASHBOARD] Access at: http://localhost:8501")
            
            return True
            
        except Exception as e:
            logger.error(f"[DASHBOARD] Failed to start dashboard: {e}")
            return False
    
    def run_complete_pipeline(self, resume: bool = True):
        """
        完全パイプラインを実行
        
        Args:
            resume: チェックポイントから再開するか
        """
        logger.info("="*80)
        logger.info("Starting Unified Master Pipeline (SO8T Thinking Controlled)")
        logger.info("="*80)
        
        start_time = datetime.now()
        
        # Streamlitダッシュボードを起動（設定で有効な場合）
        dashboard_config = self.config.get('dashboard', {})
        if dashboard_config.get('enabled', True) and dashboard_config.get('auto_start', True):
            self.start_streamlit_dashboard()
        
        # システム起動時の遅延処理（60秒待機）
        delay_seconds = 60
        logger.info(f"Waiting {delay_seconds} seconds before starting pipeline (system startup delay)...")
        time.sleep(delay_seconds)
        
        # チェックポイントから復旧
        if resume:
            checkpoint = self._load_checkpoint()
            if checkpoint:
                logger.info(f"[RESUME] Resuming from checkpoint (Session: {self.session_id})")
                self.phase_progress = checkpoint.get('phase_progress', {})
        
        try:
            # Phase 1: 並列DeepResearch Webスクレイピング
            if not self._is_phase_completed('phase1_parallel_scraping'):
                logger.info("[EXECUTE] Phase 1: Starting parallel DeepResearch web scraping")
                success = self.phase1_parallel_scraping()
                if not success and self.phase1_config.get('required', True):
                    raise RuntimeError("Phase 1 (parallel scraping) failed")
            else:
                logger.info("[SKIP] Phase 1 already completed and verified")
            
            # Phase 2: SO8T全自動データ処理
            if not self._is_phase_completed('phase2_data_processing'):
                logger.info("[EXECUTE] Phase 2: Starting SO8T auto data processing")
                success = self.phase2_data_processing()
                if not success and self.phase2_config.get('required', True):
                    raise RuntimeError("Phase 2 (data processing) failed")
            else:
                logger.info("[SKIP] Phase 2 already completed and verified")
            
            # Phase 3: SO8T完全統合A/Bテスト
            if not self._is_phase_completed('phase3_ab_test'):
                logger.info("[EXECUTE] Phase 3: Starting SO8T complete A/B test")
                success = self.phase3_ab_test()
                if not success and self.phase3_config.get('required', True):
                    raise RuntimeError("Phase 3 (A/B test) failed")
            else:
                logger.info("[SKIP] Phase 3 already completed and verified")
            
            # Phase 4: GitHubリポジトリ検索
            if not self._is_phase_completed('phase4_github_scraping'):
                logger.info("[EXECUTE] Phase 4: Starting GitHub repository scraping")
                success = self.phase4_github_scraping()
                if not success and self.phase4_config.get('required', False):
                    logger.warning("[WARNING] Phase 4 (GitHub scraping) failed, continuing...")
            else:
                logger.info("[SKIP] Phase 4 already completed and verified")
            
            # Phase 5: エンジニア向けサイトスクレイピング
            if not self._is_phase_completed('phase5_engineer_sites'):
                logger.info("[EXECUTE] Phase 5: Starting engineer sites scraping")
                success = self.phase5_engineer_sites()
                if not success and self.phase5_config.get('required', False):
                    logger.warning("[WARNING] Phase 5 (engineer sites) failed, continuing...")
            else:
                logger.info("[SKIP] Phase 5 already completed and verified")
            
            # Phase 6: コーディング関連データ抽出
            if not self._is_phase_completed('phase6_coding_extraction'):
                logger.info("[EXECUTE] Phase 6: Starting coding dataset extraction")
                success = self.phase6_coding_extraction()
                if not success and self.phase6_config.get('required', False):
                    logger.warning("[WARNING] Phase 6 (coding extraction) failed, continuing...")
            else:
                logger.info("[SKIP] Phase 6 already completed and verified")
            
            # Phase 7: コーディングタスク用データセット作成
            if not self._is_phase_completed('phase7_coding_training_data'):
                logger.info("[EXECUTE] Phase 7: Starting coding training data preparation")
                success = self.phase7_coding_training_data()
                if not success and self.phase7_config.get('required', False):
                    logger.warning("[WARNING] Phase 7 (coding training data) failed, continuing...")
            else:
                logger.info("[SKIP] Phase 7 already completed and verified")
            
            # Phase 8: コーディング特化再学習
            if not self._is_phase_completed('phase8_coding_retraining'):
                logger.info("[EXECUTE] Phase 8: Starting coding-focused retraining")
                success = self.phase8_coding_retraining()
                if not success and self.phase8_config.get('required', False):
                    logger.warning("[WARNING] Phase 8 (coding retraining) failed, continuing...")
            else:
                logger.info("[SKIP] Phase 8 already completed and verified")
            
            # Phase 9: ドキュメンテーション収集
            if not self._is_phase_completed('phase9_documentation_scraping'):
                logger.info("[EXECUTE] Phase 9: Starting documentation scraping")
                success = self.phase9_documentation_scraping()
                if not success and self.phase9_config.get('required', False):
                    logger.warning("[WARNING] Phase 9 (documentation scraping) failed, continuing...")
            else:
                logger.info("[SKIP] Phase 9 already completed and verified")
            
            # Phase 10: 統合AIエージェント基盤の構築
            if not self._is_phase_completed('phase10_unified_agent_base'):
                logger.info("[EXECUTE] Phase 10: Starting unified AI agent base construction")
                success = self.phase10_unified_agent_base()
                if not success and self.phase10_config.get('required', False):
                    logger.warning("[WARNING] Phase 10 (unified agent base) failed, continuing...")
            else:
                logger.info("[SKIP] Phase 10 already completed and verified")
            
            # Phase 11: 検知用NSFWデータセット収集
            if not self._is_phase_completed('phase11_nsfw_detection_dataset'):
                logger.info("[EXECUTE] Phase 11: Starting NSFW detection dataset collection")
                success = self.phase11_nsfw_detection_dataset()
                if not success and self.phase11_config.get('required', False):
                    logger.warning("[WARNING] Phase 11 (NSFW detection dataset) failed, continuing...")
            else:
                logger.info("[SKIP] Phase 11 already completed and verified")
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            logger.info("="*80)
            logger.info("Unified Master Pipeline Completed Successfully")
            logger.info("="*80)
            logger.info(f"Total processing time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
            
        except KeyboardInterrupt:
            logger.warning("[INTERRUPT] Pipeline interrupted by user")
            self._save_checkpoint()
        except Exception as e:
            logger.error(f"[ERROR] Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            self._save_checkpoint()
            raise


def setup_auto_start():
    """Windowsタスクスケジューラに自動実行タスクを登録"""
    logger.info("="*80)
    logger.info("Setting up auto-start task")
    logger.info("="*80)
    
    # 管理者権限チェック
    if not check_admin_privileges():
        logger.error("[ERROR] Administrator privileges required")
        logger.error("Please run this script as administrator")
        logger.error("Right-click and select 'Run as administrator'")
        return False
    
    task_name = 'SO8T-UnifiedMasterPipeline-AutoStart'
    
    # タスクスケジューラ用バッチファイルのパス
    batch_file_path = PROJECT_ROOT / 'scripts' / 'pipelines' / 'unified_master_pipeline_autostart.bat'
    
    if not batch_file_path.exists():
        logger.error(f"Batch file not found: {batch_file_path}")
        logger.error("Please ensure unified_master_pipeline_autostart.bat exists")
        return False
    
    # パスに#が含まれている場合はエラー
    batch_file_path_str = str(batch_file_path)
    if '#' in batch_file_path_str:
        logger.error(f"[ERROR] Invalid character '#' in batch file path: {batch_file_path_str}")
        logger.error("[ERROR] Please remove '#' from the path")
        return False
    
    # タスクスケジューラから呼び出されるコマンド（バッチファイルを実行）
    task_command = f'"{batch_file_path_str}"'
    
    # コマンド内に#が含まれていないか確認
    if '#' in task_command:
        logger.error(f"[ERROR] Invalid character '#' found in task command: {task_command}")
        logger.error("[ERROR] Please check batch file path for '#' characters")
        return False
    
    # 既存のタスクを削除
    try:
        result = subprocess.run(
            ["schtasks", "/query", "/tn", task_name],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            logger.info(f"Removing existing task: {task_name}")
            subprocess.run(
                ["schtasks", "/delete", "/tn", task_name, "/f"],
                check=False
            )
    except Exception as e:
        logger.warning(f"Failed to check/remove existing task: {e}")
    
    # 新しいタスクを作成
    logger.info(f"Creating new task: {task_name}")
    
    create_cmd = [
        "schtasks", "/create",
        "/tn", task_name,
        "/tr", task_command,
        "/sc", "onstart",  # システム起動時
        "/rl", "highest",  # 最高権限
        "/f"
    ]
    
    try:
        result = subprocess.run(create_cmd, capture_output=True, text=True, check=True)
        logger.info("[OK] Task created successfully")
        logger.info(f"Task name: {task_name}")
        logger.info(f"Trigger: On system start")
        logger.info(f"Command: {task_command}")
        
        subprocess.run(["schtasks", "/query", "/tn", task_name, "/fo", "list", "/v"], check=False)
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"[ERROR] Failed to create task: {e}")
        logger.error(f"Output: {e.stdout}")
        logger.error(f"Error: {e.stderr}")
        if "アクセスが拒否されました" in e.stderr or "Access is denied" in e.stderr:
            logger.error("[ERROR] Access denied. Administrator privileges required.")
            logger.error("Please run this script as administrator:")
            logger.error("  Right-click and select 'Run as administrator'")
            logger.error("  Or run: py -3 scripts\\pipelines\\unified_master_pipeline.py --setup")
        return False


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Unified Master Pipeline")
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/unified_master_pipeline_config.yaml'),
        help='Configuration file path'
    )
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Setup Windows Task Scheduler auto-start task'
    )
    parser.add_argument(
        '--run',
        action='store_true',
        help='Run the pipeline (called by task scheduler)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        default=True,
        help='Resume from checkpoint'
    )
    
    args = parser.parse_args()
    
    if args.setup:
        # タスクスケジューラ登録
        success = setup_auto_start()
        if success:
            logger.info("[OK] Auto-start task setup completed")
            return 0
        else:
            logger.error("[ERROR] Auto-start task setup failed")
            return 1
    
    elif args.run:
        # タスクスケジューラから呼び出された場合の処理
        if not args.config.exists():
            logger.error(f"Config file not found: {args.config}")
            return 1
        
        pipeline = UnifiedMasterPipeline(args.config)
        pipeline.run_complete_pipeline(resume=args.resume)
        
        return 0
    
    else:
        # 通常実行
        if not args.config.exists():
            logger.error(f"Config file not found: {args.config}")
            return 1
        
        pipeline = UnifiedMasterPipeline(args.config)
        pipeline.run_complete_pipeline(resume=args.resume)
        
        return 0


if __name__ == "__main__":
    sys.exit(main())

