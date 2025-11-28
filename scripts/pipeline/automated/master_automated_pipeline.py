#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T完全自動化マスターパイプライン

電源投入時・再起動時に自動実行される統合パイプライン
依存関係チェック → データ収集 → 訓練 → 評価 → A/Bテスト → GGUF変換 → Ollamaインポート → テストまで一貫実行

Usage:
    python scripts/pipelines/master_automated_pipeline.py --config configs/master_automated_pipeline.yaml
    python scripts/pipelines/master_automated_pipeline.py --setup  # Windowsタスクスケジューラ登録
    python scripts/pipelines/master_automated_pipeline.py --run    # パイプライン実行
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import warnings
warnings.filterwarnings("ignore")

import yaml

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/master_automated_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 進捗管理システムのインポート
try:
    from scripts.utils.progress_manager import ProgressManager
    PROGRESS_MANAGER_AVAILABLE = True
except ImportError:
    PROGRESS_MANAGER_AVAILABLE = False
    logger.warning("Progress manager not available")

# 依存関係自動インストールのインポート
try:
    from scripts.utils.auto_install_dependencies import (
        check_and_install_all,
        REQUIRED_PACKAGES
    )
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    logger.warning("Auto install dependencies not available")

# リソースバランサーのインポート
try:
    from scripts.utils.resource_balancer import ResourceBalancer
    RESOURCE_BALANCER_AVAILABLE = True
except ImportError:
    RESOURCE_BALANCER_AVAILABLE = False
    logger.warning("Resource balancer not available")


class AudioNotifier:
    """音声通知クラス"""
    
    @staticmethod
    def play_notification(audio_file: Optional[Path] = None, fallback: bool = True):
        """
        音声通知を再生
        
        Args:
            audio_file: 音声ファイルパス（デフォルト: marisa_owattaze.wav）
            fallback: フォールバック（ビープ音）を使用するか
        """
        if audio_file is None:
            audio_file = PROJECT_ROOT / ".cursor" / "marisa_owattaze.wav"
        
        audio_played = False
        
        # 方法1: PowerShell SoundPlayer（PRIMARY METHOD）
        if audio_file.exists():
            try:
                ps_cmd = f"""
                if (Test-Path '{audio_file}') {{
                    Add-Type -AssemblyName System.Windows.Forms
                    $player = New-Object System.Media.SoundPlayer('{audio_file}')
                    $player.PlaySync()
                    Write-Host '[OK] marisa_owattaze.wav played successfully' -ForegroundColor Green
                }}
                """
                result = subprocess.run(
                    ["powershell", "-Command", ps_cmd],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    logger.info("[AUDIO] Audio notification played successfully")
                    audio_played = True
            except Exception as e:
                logger.warning(f"[AUDIO] Failed to play audio: {e}")
        
        # 方法2: フォールバック（ビープ音）
        if not audio_played and fallback:
            try:
                import winsound
                winsound.Beep(1000, 500)
                logger.info("[AUDIO] Fallback beep played successfully")
            except Exception as e:
                logger.warning(f"[AUDIO] Fallback beep also failed: {e}")


class MasterAutomatedPipeline:
    """SO8T完全自動化マスターパイプラインクラス"""
    
    def __init__(self, config_path: str, resume_from_checkpoint: Optional[str] = None, 
                 reset_checkpoint: bool = False, start_from_phase: Optional[str] = None):
        """
        初期化
        
        Args:
            config_path: 設定ファイルパス
            resume_from_checkpoint: チェックポイントパス（復旧時）
            reset_checkpoint: チェックポイントをクリアするか
            start_from_phase: 特定のフェーズから開始する場合のフェーズ名
        """
        self.config = self._load_config(config_path)
        self.session_id = datetime.now().strftime(self.config.get('pipeline', {}).get('session_id_format', '%Y%m%d_%H%M%S'))
        
        # チェックポイント設定
        checkpoint_config = self.config.get('checkpoint', {})
        self.checkpoint_dir = Path(checkpoint_config.get('save_dir', 'D:/webdataset/checkpoints/master_pipeline'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = checkpoint_config.get('interval_seconds', 300)  # デフォルト5分
        self.max_checkpoints = checkpoint_config.get('max_checkpoints', 10)
        self.last_checkpoint_time = time.time()
        self.auto_recovery = checkpoint_config.get('auto_recovery', True)
        
        # チェックポイントリセット処理
        if reset_checkpoint:
            logger.info("[RESET] Clearing all checkpoints...")
            self._clear_checkpoints()
        
        # 特定フェーズからの開始処理
        self.start_from_phase = start_from_phase
        
        # 進捗管理システム
        if PROGRESS_MANAGER_AVAILABLE:
            progress_config = self.config.get('progress', {})
            log_interval = progress_config.get('log_interval', 1800)
            self.progress_manager = ProgressManager(session_id=self.session_id, log_interval=log_interval)
            self.progress_manager.start_logging()
        else:
            self.progress_manager = None
        
        # リソースバランサー
        if RESOURCE_BALANCER_AVAILABLE:
            self.resource_balancer = ResourceBalancer(self.config)
        else:
            self.resource_balancer = None
        
        # フェーズ状態
        self.current_phase = None
        self.phase_results = {}
        self.resume_from_checkpoint = resume_from_checkpoint
        
        # エラーハンドリング設定
        error_config = self.config.get('error_handling', {})
        self.max_retries = error_config.get('max_retries', 3)
        self.retry_delay = error_config.get('retry_delay', 60)
        self.continue_on_error = error_config.get('continue_on_error', False)
        self.continue_on_optional_error = error_config.get('continue_on_optional_error', True)
        
        # 通知設定
        notification_config = self.config.get('notifications', {})
        self.audio_notification = notification_config.get('audio_notification', True)
        self.play_on_phase_completion = notification_config.get('play_on_phase_completion', True)
        
        # シグナルハンドラー設定
        self._setup_signal_handlers()
        
        logger.info("="*80)
        logger.info("Master Automated Pipeline Initialized")
        logger.info("="*80)
        logger.info(f"Session ID: {self.session_id}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        logger.info(f"Resume from checkpoint: {resume_from_checkpoint}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_signal_handlers(self):
        """シグナルハンドラー設定"""
        def signal_handler(signum, frame):
            logger.warning(f"Received signal {signum}, saving checkpoint...")
            self._save_checkpoint()
            if self.resource_balancer:
                self.resource_balancer.stop_monitoring()
            if self.progress_manager:
                self.progress_manager.stop_logging()
            sys.exit(1)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)
    
    def _save_checkpoint(self):
        """チェックポイント保存"""
        checkpoint_data = {
            'session_id': self.session_id,
            'current_phase': self.current_phase,
            'phase_results': self.phase_results,
            'timestamp': datetime.now().isoformat(),
            'checkpoint_interval': self.checkpoint_interval
        }
        
        checkpoint_path = self.checkpoint_dir / f"{self.session_id}_checkpoint.json"
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        
        # 古いチェックポイントを削除
        checkpoint_files = sorted(self.checkpoint_dir.glob("*_checkpoint.json"), key=lambda p: p.stat().st_mtime)
        if len(checkpoint_files) > self.max_checkpoints:
            for old_checkpoint in checkpoint_files[:-self.max_checkpoints]:
                old_checkpoint.unlink()
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _clear_checkpoints(self):
        """チェックポイントをクリア"""
        checkpoint_files = list(self.checkpoint_dir.glob("*_checkpoint.json"))
        if checkpoint_files:
            for checkpoint_file in checkpoint_files:
                checkpoint_file.unlink()
                logger.info(f"Deleted checkpoint: {checkpoint_file}")
            logger.info(f"[OK] Cleared {len(checkpoint_files)} checkpoint(s)")
        else:
            logger.info("[INFO] No checkpoints to clear")
        
        # フェーズ結果もリセット
        self.phase_results = {}
        self.current_phase = None
    
    def _reset_from_phase(self, start_phase: str):
        """特定のフェーズから再実行するために、それ以前のフェーズ結果をクリア"""
        phase_order = [
            'phase0_dependencies',
            'phase1_web_scraping',
            'phase2_data_cleansing',
            'phase3_modeling_so8t',
            'phase4_integration',
            'phase5_qlora_training',
            'phase6_evaluation',
            'phase7_ab_test',
            'phase8_post_processing',
            'phase9_japanese_test'
        ]
        
        try:
            start_index = phase_order.index(start_phase)
            # 開始フェーズ以降のフェーズ結果をクリア
            phases_to_clear = phase_order[start_index:]
            
            for phase in phases_to_clear:
                if phase in self.phase_results:
                    del self.phase_results[phase]
                    logger.info(f"[RESET] Cleared phase result: {phase}")
            
            logger.info(f"[OK] Reset from phase: {start_phase}")
            return True
        except ValueError:
            logger.error(f"[ERROR] Invalid phase name: {start_phase}")
            return False
    
    def _load_checkpoint(self) -> Optional[Dict]:
        """チェックポイント読み込み"""
        if self.resume_from_checkpoint:
            checkpoint_path = Path(self.resume_from_checkpoint)
        else:
            # 最新のチェックポイントを検索
            checkpoint_files = list(self.checkpoint_dir.glob("*_checkpoint.json"))
            if not checkpoint_files:
                return None
            
            # 有効期限チェック
            recovery_timeout = self.config.get('checkpoint', {}).get('recovery_timeout', 86400)
            checkpoint_path = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
            checkpoint_age = time.time() - checkpoint_path.stat().st_mtime
            
            if checkpoint_age > recovery_timeout:
                logger.warning(f"Checkpoint expired (age: {checkpoint_age/3600:.1f} hours)")
                return None
        
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def _check_and_install_dependencies(self) -> bool:
        """依存関係チェック・インストール"""
        phase_config = self.config.get('phase0_dependencies', {})
        if not phase_config.get('enabled', True):
            logger.info("[PHASE 0] Dependencies check skipped (disabled)")
            return True
        
        logger.info("="*80)
        logger.info("[PHASE 0] Checking and Installing Dependencies")
        logger.info("="*80)
        
        if not DEPENDENCIES_AVAILABLE:
            logger.error("[PHASE 0] Dependencies module not available")
            if phase_config.get('continue_on_failure', False):
                return True
            return False
        
        try:
            required_packages = self.config.get('dependencies', {}).get('required_packages', [])
            if not required_packages:
                # デフォルトパッケージリストを使用
                required_packages_dict = REQUIRED_PACKAGES
            else:
                # 設定ファイルからパッケージリストを作成
                required_packages_dict = {}
                for pkg_spec in required_packages:
                    if '>=' in pkg_spec:
                        pkg_name, version = pkg_spec.split('>=')
                        required_packages_dict[pkg_name] = pkg_spec
                    else:
                        required_packages_dict[pkg_spec] = pkg_spec
            
            results = check_and_install_all(required_packages_dict, show_progress=True)
            
            success_count = sum(1 for v in results.values() if v)
            total_count = len(results)
            
            if success_count < total_count:
                failed = [k for k, v in results.items() if not v]
                logger.error(f"[PHASE 0] Failed packages: {', '.join(failed)}")
                if not phase_config.get('continue_on_failure', False):
                    return False
            
            logger.info(f"[OK] Phase 0 completed ({success_count}/{total_count} packages)")
            return True
            
        except Exception as e:
            logger.error(f"[PHASE 0] Failed: {e}")
            if not phase_config.get('continue_on_failure', False):
                return False
            return True
    
    def _run_phase_with_retry(
        self,
        phase_name: str,
        phase_func,
        required: bool = True,
        *args,
        **kwargs
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        フェーズをリトライ付きで実行
        
        Args:
            phase_name: フェーズ名
            phase_func: フェーズ実行関数
            required: 必須フェーズかどうか
            *args, **kwargs: フェーズ関数への引数
        
        Returns:
            (success, result): 成功フラグと結果
        """
        self.current_phase = phase_name
        
        if self.progress_manager:
            self.progress_manager.update_phase_status(phase_name, "running", progress=0.0)
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"[{phase_name}] Attempt {attempt + 1}/{self.max_retries + 1}")
                result = phase_func(*args, **kwargs)
                
                if self.progress_manager:
                    self.progress_manager.update_phase_status(phase_name, "completed", progress=1.0)
                
                self.phase_results[phase_name] = {
                    'status': 'completed',
                    'result': result,
                    'attempt': attempt + 1
                }
                
                self._save_checkpoint()
                
                if self.audio_notification and self.play_on_phase_completion:
                    AudioNotifier.play_notification()
                
                logger.info(f"[OK] {phase_name} completed")
                return True, result
                
            except Exception as e:
                logger.error(f"[{phase_name}] Attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries:
                    logger.info(f"[{phase_name}] Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"[{phase_name}] All retries failed")
                    
                    if self.progress_manager:
                        self.progress_manager.update_phase_status(
                            phase_name, "failed", progress=0.0,
                            error_message=str(e)
                        )
                    
                    self.phase_results[phase_name] = {
                        'status': 'failed',
                        'error': str(e),
                        'attempts': attempt + 1
                    }
                    
                    self._save_checkpoint()
                    
                    if required:
                        if self.continue_on_error:
                            logger.warning(f"[{phase_name}] Required phase failed, but continuing...")
                            return False, None
                        else:
                            logger.error(f"[{phase_name}] Required phase failed, stopping pipeline")
                            raise
                    else:
                        if self.continue_on_optional_error:
                            logger.warning(f"[{phase_name}] Optional phase failed, continuing...")
                            return False, None
                        else:
                            logger.error(f"[{phase_name}] Optional phase failed, stopping pipeline")
                            raise
        
        return False, None
    
    def run_phase1_web_scraping(self) -> Dict[str, Any]:
        """Phase 1: Webスクレイピングとデータ収集"""
        phase_config = self.config.get('phase1_web_scraping', {})
        if not phase_config.get('enabled', True):
            logger.info("[PHASE 1] Skipped (disabled)")
            return {'status': 'skipped'}
        
        logger.info("="*80)
        logger.info("[PHASE 1] Webスクレイピングとデータ収集")
        logger.info("="*80)
        
        # Playwrightクローラーを使用するかどうか
        use_playwright = phase_config.get('use_playwright', True)
        use_cursor_browser = phase_config.get('use_cursor_browser', True)
        
        if use_playwright:
            # Playwrightクローラーを使用
            logger.info("[PHASE 1] Using Playwright crawler")
            
            try:
                import asyncio
                from scripts.data.playwright_crawler import PlaywrightCrawler
                
                output_dir = Path(phase_config.get('output_dir', 'D:/webdataset/processed/four_class'))
                remote_debugging_port = phase_config.get('remote_debugging_port', 9222)
                
                # 設定ファイルを読み込んでクローラーに渡す
                pipeline_config_path = phase_config.get('pipeline_config', 'configs/data_pipeline_config.yaml')
                config_dict = {}
                if Path(pipeline_config_path).exists():
                    import yaml
                    with open(pipeline_config_path, 'r', encoding='utf-8') as f:
                        config_dict = yaml.safe_load(f)
                
                # クローラー作成
                crawler = PlaywrightCrawler(
                    output_dir=output_dir,
                    use_cursor_browser=use_cursor_browser,
                    remote_debugging_port=remote_debugging_port,
                    headless=False,  # Cursorブラウザ使用時はheadless=False
                    delay_per_request=phase_config.get('delay', 1.0),
                    timeout=phase_config.get('timeout', 30000),
                    max_pages=phase_config.get('target_samples', 100000)
                )
                
                # クロール実行
                output_file = asyncio.run(crawler.crawl_from_config(config_dict))
                
                return {'status': 'completed', 'output': str(output_file)}
                
            except ImportError:
                logger.warning("[PHASE 1] Playwright not available, falling back to requests-based crawler")
                use_playwright = False
            except Exception as e:
                logger.error(f"[PHASE 1] Playwright crawler failed: {e}")
                logger.info("[PHASE 1] Falling back to requests-based crawler")
                use_playwright = False
        
        if not use_playwright:
            # 既存のrequestsベースのパイプラインを使用
            logger.info("[PHASE 1] Using requests-based crawler")
            
            pipeline_script = PROJECT_ROOT / phase_config.get('script', 'scripts/pipelines/complete_data_pipeline.py')
            pipeline_config = phase_config.get('pipeline_config', 'configs/data_pipeline_config.yaml')
            
            cmd = [
                sys.executable,
                str(pipeline_script),
                '--config', str(pipeline_config)
            ]
            
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                check=True
            )
            
            # 出力ファイルを探す
            output_dir = Path(phase_config.get('output_dir', 'D:/webdataset/processed/four_class'))
            jsonl_files = list(output_dir.glob("four_class_*.jsonl"))
            if jsonl_files:
                latest_file = max(jsonl_files, key=lambda p: p.stat().st_mtime)
                return {'status': 'completed', 'output': str(latest_file)}
            
            return {'status': 'completed', 'output': str(output_dir)}
    
    def run_phase2_data_cleansing(self) -> Dict[str, Any]:
        """Phase 2: データクレンジングと前処理"""
        phase_config = self.config.get('phase2_data_cleansing', {})
        if not phase_config.get('enabled', True):
            logger.info("[PHASE 2] Skipped (disabled)")
            return {'status': 'skipped'}
        
        logger.info("="*80)
        logger.info("[PHASE 2] データクレンジングと前処理")
        logger.info("="*80)
        
        # Phase 1の出力を使用
        phase1_output = self.phase_results.get('phase1_web_scraping', {}).get('result', {}).get('output')
        if phase1_output:
            return {'status': 'completed', 'output': phase1_output}
        
        # フォールバック: 直接探す
        output_dir = Path(phase_config.get('output_dir', 'D:/webdataset/processed/finetuning'))
        jsonl_files = list(output_dir.glob("*.jsonl"))
        if jsonl_files:
            latest_file = max(jsonl_files, key=lambda p: p.stat().st_mtime)
            return {'status': 'completed', 'output': str(latest_file)}
        
        return {'status': 'completed', 'output': str(output_dir)}
    
    def run_phase3_modeling_so8t(self) -> Dict[str, Any]:
        """Phase 3: modeling_phi3.pyのSO8T統合"""
        phase_config = self.config.get('phase3_modeling_so8t', {})
        if not phase_config.get('enabled', True):
            logger.info("[PHASE 3] Skipped (disabled)")
            return {'status': 'skipped'}
        
        logger.info("="*80)
        logger.info("[PHASE 3] modeling_phi3.pyのSO8T統合")
        logger.info("="*80)
        
        modeling_file = PROJECT_ROOT / phase_config.get('modeling_file', 'models/Borea-Phi-3.5-mini-Instruct-Jp/modeling_phi3_so8t.py')
        
        if modeling_file.exists():
            return {'status': 'completed', 'output': str(modeling_file)}
        else:
            raise FileNotFoundError(f"SO8T modeling file not found: {modeling_file}")
    
    def run_phase4_integration(self) -> Dict[str, Any]:
        """Phase 4: SO8T統合スクリプト"""
        phase_config = self.config.get('phase4_integration', {})
        if not phase_config.get('enabled', True):
            logger.info("[PHASE 4] Skipped (disabled)")
            return {'status': 'skipped'}
        
        logger.info("="*80)
        logger.info("[PHASE 4] SO8T統合スクリプト")
        logger.info("="*80)
        
        integration_script = PROJECT_ROOT / phase_config.get('script', 'scripts/conversion/integrate_phi3_so8t.py')
        model_path = phase_config.get('model_path', 'models/Borea-Phi-3.5-mini-Instruct-Jp')
        output_path = phase_config.get('output_path', 'D:/webdataset/models/so8t_integrated/phi3_so8t')
        
        # メモリ使用量をチェックして待機
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            logger.info(f"[MEMORY] Current memory usage: {memory_percent:.1f}%")
            
            # メモリ使用量が高い場合は待機
            max_wait_time = 300  # 最大5分待機
            wait_interval = 10  # 10秒ごとにチェック
            waited_time = 0
            
            while memory_percent > 85 and waited_time < max_wait_time:
                logger.warning(f"[WARNING] High memory usage ({memory_percent:.1f}%), waiting for memory to free up...")
                logger.info(f"[INFO] Waiting {wait_interval} seconds... (waited {waited_time}/{max_wait_time} seconds)")
                time.sleep(wait_interval)
                waited_time += wait_interval
                memory_percent = psutil.virtual_memory().percent
            
            if memory_percent > 85:
                logger.error(f"[ERROR] Memory usage still high ({memory_percent:.1f}%) after waiting {max_wait_time} seconds")
                logger.error("[ERROR] Cannot proceed with Phase 4 due to insufficient memory")
                raise RuntimeError(f"Insufficient memory: {memory_percent:.1f}% used")
            
            # メモリ使用量が高い場合はCPUモードにフォールバック
            device = phase_config.get('device', 'cuda')
            if memory_percent > 80:
                logger.warning(f"[WARNING] Memory usage is high ({memory_percent:.1f}%)")
                logger.warning("[WARNING] Falling back to CPU mode to avoid memory errors")
                device = 'cpu'
        except ImportError:
            device = phase_config.get('device', 'cuda')
            logger.warning("[WARNING] psutil not available, cannot check memory usage")
        
        cmd = [
            sys.executable,
            str(integration_script),
            '--model_path', str(model_path),
            '--output_path', str(output_path),
            '--device', device,  # メモリ状況に応じて調整
            '--torch_dtype', phase_config.get('torch_dtype', 'bfloat16'),
        ]
        if not phase_config.get('verify', True):
            cmd.append('--no-verify')
        
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True
        )
        
        return {'status': 'completed', 'output': str(output_path)}
    
    def run_phase5_qlora_training(self) -> Dict[str, Any]:
        """Phase 5: QLoRA 8bitファインチューニング"""
        phase_config = self.config.get('phase5_qlora_training', {})
        if not phase_config.get('enabled', True):
            logger.info("[PHASE 5] Skipped (disabled)")
            return {'status': 'skipped'}
        
        logger.info("="*80)
        logger.info("[PHASE 5] QLoRA 8bitファインチューニング")
        logger.info("="*80)
        
        training_script = PROJECT_ROOT / phase_config.get('script', 'scripts/training/train_so8t_phi3_qlora.py')
        training_config = phase_config.get('config', 'configs/train_so8t_phi3_qlora.yaml')
        
        cmd = [
            sys.executable,
            str(training_script),
            '--config', str(training_config),
        ]
        
        if phase_config.get('resume', True):
            # 最新のチェックポイントから再開
            checkpoint_dir = Path(phase_config.get('output_dir', 'D:/webdataset/checkpoints/finetuning/so8t_phi3'))
            if checkpoint_dir.exists():
                checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[1]) if x.name.split("-")[1].isdigit() else 0)
                if checkpoints:
                    cmd.extend(['--resume', str(checkpoints[-1])])
        
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=False,
            text=True,
            check=True
        )
        
        output_dir = Path(phase_config.get('output_dir', 'D:/webdataset/checkpoints/finetuning/so8t_phi3'))
        return {'status': 'completed', 'output': str(output_dir)}
    
    def run_phase6_evaluation(self) -> Dict[str, Any]:
        """Phase 6: モデル評価"""
        phase_config = self.config.get('phase6_evaluation', {})
        if not phase_config.get('enabled', True):
            logger.info("[PHASE 6] Skipped (disabled)")
            return {'status': 'skipped'}
        
        logger.info("="*80)
        logger.info("[PHASE 6] モデル評価")
        logger.info("="*80)
        
        eval_script = PROJECT_ROOT / phase_config.get('script', 'scripts/evaluation/evaluate_so8t_phi3.py')
        trained_model_path = self.phase_results.get('phase5_qlora_training', {}).get('result', {}).get('output')
        
        if not trained_model_path:
            raise ValueError("Trained model path not found")
        
        output_dir = Path(phase_config.get('output_dir', 'D:/webdataset/evaluation_results'))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"evaluation_{self.session_id}.json"
        
        cmd = [
            sys.executable,
            str(eval_script),
            '--model-path', str(trained_model_path),
            '--output', str(output_path),
            '--device', 'cuda'
        ]
        
        eval_dataset = phase_config.get('eval_dataset')
        if eval_dataset:
            cmd.extend(['--eval-dataset', str(eval_dataset)])
        
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=False,
            text=True,
            check=True
        )
        
        return {'status': 'completed', 'output': str(output_path)}
    
    def run_phase7_ab_test(self) -> Dict[str, Any]:
        """Phase 7: A/Bテスト実行"""
        phase_config = self.config.get('phase7_ab_test', {})
        if not phase_config.get('enabled', True):
            logger.info("[PHASE 7] Skipped (disabled)")
            return {'status': 'skipped'}
        
        logger.info("="*80)
        logger.info("[PHASE 7] A/Bテスト実行")
        logger.info("="*80)
        
        ab_test_script = PROJECT_ROOT / phase_config.get('script', 'scripts/evaluation/ab_test_borea_phi35_original_vs_so8t.py')
        base_model = phase_config.get('base_model', 'models/Borea-Phi-3.5-mini-Instruct-Jp')
        retrained_model = self.phase_results.get('phase5_qlora_training', {}).get('result', {}).get('output')
        test_data = phase_config.get('test_data', 'D:/webdataset/processed/finetuning/eval.jsonl')
        output_dir = Path(phase_config.get('output_dir', 'D:/webdataset/ab_test_results'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            sys.executable,
            str(ab_test_script),
            '--base-model', str(base_model),
            '--retrained-model', str(retrained_model),
            '--test-data', str(test_data),
            '--output-dir', str(output_dir),
            '--device', 'cuda'
        ]
        
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True
        )
        
        results_path = output_dir / "ab_test_results.json"
        return {'status': 'completed', 'output': str(results_path)}
    
    def run_phase8_post_processing(self) -> Dict[str, Any]:
        """Phase 8: A/Bテスト後処理（勝者判定・GGUF変換・Ollamaインポート）"""
        phase_config = self.config.get('phase8_post_processing', {})
        if not phase_config.get('enabled', True):
            logger.info("[PHASE 8] Skipped (disabled)")
            return {'status': 'skipped'}
        
        logger.info("="*80)
        logger.info("[PHASE 8] A/Bテスト後処理")
        logger.info("="*80)
        
        post_processing_script = PROJECT_ROOT / phase_config.get('script', 'scripts/pipelines/complete_ab_test_post_processing_pipeline.py')
        post_processing_config = phase_config.get('config', 'configs/complete_ab_test_post_processing_config.yaml')
        
        cmd = [
            sys.executable,
            str(post_processing_script),
            '--config', str(post_processing_config)
        ]
        
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=False,
            text=True,
            check=True
        )
        
        return {'status': 'completed', 'output': 'post_processing_completed'}
    
    def run_phase9_japanese_test(self) -> Dict[str, Any]:
        """Phase 9: 日本語パフォーマンステスト"""
        phase_config = self.config.get('phase9_japanese_test', {})
        if not phase_config.get('enabled', True):
            logger.info("[PHASE 9] Skipped (disabled)")
            return {'status': 'skipped'}
        
        logger.info("="*80)
        logger.info("[PHASE 9] 日本語パフォーマンステスト")
        logger.info("="*80)
        
        from scripts.testing.japanese_llm_performance_test import JapaneseLLMPerformanceTester
        
        ollama_model_name = self.config.get('ollama', {}).get('model_b_name', 'so8t-borea-phi35-mini-q8_0')
        output_dir = Path(phase_config.get('output_dir', '_docs'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        tester = JapaneseLLMPerformanceTester(
            model_name=f"{ollama_model_name}:latest",
            output_dir=output_dir
        )
        
        results_path = tester.run_all_tests()
        return {'status': 'completed', 'output': str(results_path)}
    
    def _check_phase_dependencies(self, phase_name: str) -> bool:
        """フェーズの依存関係をチェック"""
        dependencies = self.config.get('dependencies_phases', {}).get(phase_name, [])
        
        for dep_phase in dependencies:
            dep_result = self.phase_results.get(dep_phase, {})
            
            # 複数の形式に対応（statusフィールドまたはresult内のstatus）
            status = dep_result.get('status')
            if not status and isinstance(dep_result, dict) and 'result' in dep_result:
                status = dep_result.get('result', {}).get('status')
            
            if status != 'completed' and status != 'skipped':
                logger.warning(f"[{phase_name}] Dependency {dep_phase} not completed (status: {status})")
                logger.debug(f"[{phase_name}] Dependency {dep_phase} result: {dep_result}")
                return False
        
        return True
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """完全パイプライン実行"""
        logger.info("="*80)
        logger.info("Starting Master Automated Pipeline")
        logger.info("="*80)
        
        start_time = time.time()
        
        # 特定フェーズからの開始処理
        if self.start_from_phase:
            logger.info(f"[RESET] Starting from phase: {self.start_from_phase}")
            # チェックポイントを読み込んでから、指定フェーズ以降をリセット
            checkpoint = self._load_checkpoint()
            if checkpoint:
                self.session_id = checkpoint.get('session_id', self.session_id)
                self.phase_results = checkpoint.get('phase_results', {})
                # フェーズ結果の構造を正規化
                for phase_name, phase_result in self.phase_results.items():
                    if isinstance(phase_result, dict):
                        if 'status' not in phase_result and 'result' in phase_result:
                            phase_result['status'] = phase_result.get('result', {}).get('status', 'unknown')
                # 指定フェーズ以降をリセット
                self._reset_from_phase(self.start_from_phase)
            else:
                logger.warning(f"[WARNING] No checkpoint found, starting from phase {self.start_from_phase} anyway")
        
        # チェックポイントから復旧（特定フェーズ指定がない場合のみ）
        elif not self.start_from_phase:
            checkpoint = self._load_checkpoint()
            if checkpoint:
                logger.info(f"Resuming from checkpoint: {checkpoint.get('current_phase')}")
                self.session_id = checkpoint.get('session_id', self.session_id)
                self.phase_results = checkpoint.get('phase_results', {})
                
                # 完了済みフェーズをログに出力
                completed_phases = [name for name, result in self.phase_results.items() 
                                   if result.get('status') == 'completed' or 
                                      (isinstance(result, dict) and result.get('result', {}).get('status') == 'completed')]
                if completed_phases:
                    logger.info(f"Completed phases from checkpoint: {', '.join(completed_phases)}")
                
                # フェーズ結果の構造を正規化（後方互換性のため）
                for phase_name, phase_result in self.phase_results.items():
                    if isinstance(phase_result, dict):
                        # statusフィールドがない場合、result内のstatusを使用
                        if 'status' not in phase_result and 'result' in phase_result:
                            phase_result['status'] = phase_result.get('result', {}).get('status', 'unknown')
        
        # リソースバランス監視開始
        if self.resource_balancer:
            self.resource_balancer.start_monitoring()
        
        try:
            # 特定フェーズからの開始処理
            start_phase_index = None
            if self.start_from_phase:
                phase_order = [
                    'phase0_dependencies',
                    'phase1_web_scraping',
                    'phase2_data_cleansing',
                    'phase3_modeling_so8t',
                    'phase4_integration',
                    'phase5_qlora_training',
                    'phase6_evaluation',
                    'phase7_ab_test',
                    'phase8_post_processing',
                    'phase9_japanese_test'
                ]
                try:
                    start_phase_index = phase_order.index(self.start_from_phase)
                    logger.info(f"[START] Starting from phase index: {start_phase_index} ({self.start_from_phase})")
                except ValueError:
                    logger.error(f"[ERROR] Invalid start phase: {self.start_from_phase}")
                    raise ValueError(f"Invalid start phase: {self.start_from_phase}")
            
            # Phase 0: 依存関係チェック・インストール
            if start_phase_index is None or start_phase_index <= 0:
                phase0_status = self.phase_results.get('phase0_dependencies', {}).get('status')
                if phase0_status != 'completed':
                    success, _ = self._run_phase_with_retry(
                        'phase0_dependencies',
                        self._check_and_install_dependencies,
                        required=True
                    )
                    if not success:
                        raise RuntimeError("Phase 0 (dependencies) failed")
                else:
                    logger.info("[PHASE 0] Already completed, skipping")
            
            # Phase 1: Webスクレイピング
            if start_phase_index is None or start_phase_index <= 1:
                phase1_status = self.phase_results.get('phase1_web_scraping', {}).get('status')
                if phase1_status != 'completed':
                    if self._check_phase_dependencies('phase1_web_scraping'):
                        phase_config = self.config.get('phase1_web_scraping', {})
                        success, _ = self._run_phase_with_retry(
                            'phase1_web_scraping',
                            self.run_phase1_web_scraping,
                            required=phase_config.get('required', True)
                        )
                        if not success and phase_config.get('required', True):
                            raise RuntimeError("Phase 1 (web scraping) failed")
                else:
                    logger.info("[PHASE 1] Already completed, skipping")
            
            # Phase 2: データクレンジング
            if start_phase_index is None or start_phase_index <= 2:
                phase2_status = self.phase_results.get('phase2_data_cleansing', {}).get('status')
                if phase2_status != 'completed':
                    if self._check_phase_dependencies('phase2_data_cleansing'):
                        phase_config = self.config.get('phase2_data_cleansing', {})
                        success, _ = self._run_phase_with_retry(
                            'phase2_data_cleansing',
                            self.run_phase2_data_cleansing,
                            required=phase_config.get('required', True)
                        )
                        if not success and phase_config.get('required', True):
                            raise RuntimeError("Phase 2 (data cleansing) failed")
                else:
                    logger.info("[PHASE 2] Already completed, skipping")
            
            # Phase 3: SO8T統合
            if start_phase_index is None or start_phase_index <= 3:
                phase3_status = self.phase_results.get('phase3_modeling_so8t', {}).get('status')
                if phase3_status != 'completed':
                    if self._check_phase_dependencies('phase3_modeling_so8t'):
                        phase_config = self.config.get('phase3_modeling_so8t', {})
                        success, _ = self._run_phase_with_retry(
                            'phase3_modeling_so8t',
                            self.run_phase3_modeling_so8t,
                            required=phase_config.get('required', True)
                        )
                        if not success and phase_config.get('required', True):
                            raise RuntimeError("Phase 3 (SO8T modeling) failed")
                else:
                    logger.info("[PHASE 3] Already completed, skipping")
            
            # Phase 4: SO8T統合スクリプト
            if start_phase_index is None or start_phase_index <= 4:
                phase4_status = self.phase_results.get('phase4_integration', {}).get('status')
                if phase4_status != 'completed':
                    if self._check_phase_dependencies('phase4_integration'):
                        phase_config = self.config.get('phase4_integration', {})
                        success, _ = self._run_phase_with_retry(
                            'phase4_integration',
                            self.run_phase4_integration,
                            required=phase_config.get('required', True)
                        )
                        if not success and phase_config.get('required', True):
                            raise RuntimeError("Phase 4 (integration) failed")
                else:
                    logger.info("[PHASE 4] Already completed, skipping")
            
            # Phase 5: QLoRA訓練
            if start_phase_index is None or start_phase_index <= 5:
                phase5_status = self.phase_results.get('phase5_qlora_training', {}).get('status')
                if phase5_status != 'completed':
                    if self._check_phase_dependencies('phase5_qlora_training'):
                        phase_config = self.config.get('phase5_qlora_training', {})
                        success, _ = self._run_phase_with_retry(
                            'phase5_qlora_training',
                            self.run_phase5_qlora_training,
                            required=phase_config.get('required', True)
                        )
                        if not success and phase_config.get('required', True):
                            raise RuntimeError("Phase 5 (QLoRA training) failed")
                else:
                    logger.info("[PHASE 5] Already completed, skipping")
            
            # Phase 6: 評価
            if start_phase_index is None or start_phase_index <= 6:
                phase6_status = self.phase_results.get('phase6_evaluation', {}).get('status')
                if phase6_status != 'completed':
                    if self._check_phase_dependencies('phase6_evaluation'):
                        phase_config = self.config.get('phase6_evaluation', {})
                        success, _ = self._run_phase_with_retry(
                            'phase6_evaluation',
                            self.run_phase6_evaluation,
                            required=phase_config.get('required', False)
                        )
                else:
                    logger.info("[PHASE 6] Already completed, skipping")
            
            # Phase 7: A/Bテスト
            if start_phase_index is None or start_phase_index <= 7:
                phase7_status = self.phase_results.get('phase7_ab_test', {}).get('status')
                if phase7_status != 'completed':
                    if self._check_phase_dependencies('phase7_ab_test'):
                        phase_config = self.config.get('phase7_ab_test', {})
                        success, _ = self._run_phase_with_retry(
                            'phase7_ab_test',
                            self.run_phase7_ab_test,
                            required=phase_config.get('required', False)
                        )
                else:
                    logger.info("[PHASE 7] Already completed, skipping")
            
            # Phase 8: 後処理
            if start_phase_index is None or start_phase_index <= 8:
                phase8_status = self.phase_results.get('phase8_post_processing', {}).get('status')
                if phase8_status != 'completed':
                    if self._check_phase_dependencies('phase8_post_processing'):
                        phase_config = self.config.get('phase8_post_processing', {})
                        success, _ = self._run_phase_with_retry(
                            'phase8_post_processing',
                            self.run_phase8_post_processing,
                            required=phase_config.get('required', False)
                        )
                else:
                    logger.info("[PHASE 8] Already completed, skipping")
            
            # Phase 9: 日本語テスト
            if start_phase_index is None or start_phase_index <= 9:
                phase9_status = self.phase_results.get('phase9_japanese_test', {}).get('status')
                if phase9_status != 'completed':
                    if self._check_phase_dependencies('phase9_japanese_test'):
                        phase_config = self.config.get('phase9_japanese_test', {})
                        success, _ = self._run_phase_with_retry(
                            'phase9_japanese_test',
                            self.run_phase9_japanese_test,
                            required=phase_config.get('required', False)
                        )
                else:
                    logger.info("[PHASE 9] Already completed, skipping")
            
            # リソースバランス監視停止
            if self.resource_balancer:
                self.resource_balancer.stop_monitoring()
                self.resource_balancer.save_metrics_history()
            
            duration = time.time() - start_time
            
            result = {
                'status': 'completed',
                'duration': duration,
                'session_id': self.session_id,
                'phase_results': self.phase_results,
                'timestamp': datetime.now().isoformat()
            }
            
            # 最終チェックポイント保存
            self._save_checkpoint()
            
            logger.info("="*80)
            logger.info("[SUCCESS] Master Automated Pipeline Completed!")
            logger.info("="*80)
            logger.info(f"Total duration: {duration/3600:.2f} hours")
            logger.info(f"Session ID: {self.session_id}")
            
            # 進捗管理停止
            if self.progress_manager:
                self.progress_manager.stop_logging()
            
            # 音声通知
            if self.audio_notification:
                AudioNotifier.play_notification()
            
            return result
            
        except Exception as e:
            logger.error("="*80)
            logger.error(f"[ERROR] Master Automated Pipeline Failed: {e}")
            logger.error("="*80)
            logger.exception(e)
            
            if self.resource_balancer:
                self.resource_balancer.stop_monitoring()
            
            if self.progress_manager:
                self.progress_manager.stop_logging()
            
            # エラー時もチェックポイント保存
            self._save_checkpoint()
            
            # エラー時の音声通知
            if self.audio_notification:
                AudioNotifier.play_notification()
            
            raise


def setup_auto_start():
    """Windowsタスクスケジューラに自動実行タスクを登録"""
    logger.info("="*80)
    logger.info("Setting up auto-start task")
    logger.info("="*80)
    
    task_config = Path("configs/master_automated_pipeline.yaml")
    if not task_config.exists():
        logger.error(f"Config file not found: {task_config}")
        return False
    
    with open(task_config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    scheduler_config = config.get('task_scheduler', {})
    task_name = scheduler_config.get('task_name', 'SO8T-MasterAutomatedPipeline-AutoStart')
    
    script_path = PROJECT_ROOT / scheduler_config.get('script_path', 'scripts/pipelines/master_automated_pipeline.py')
    python_exe = sys.executable
    
    task_command = f'"{python_exe}" "{script_path}" --run --config "{task_config}"'
    
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
    
    # /delayパラメータは/sc onstartと一緒に使えないため、削除
    # 遅延処理はPythonスクリプト内で実装
    create_cmd = [
        "schtasks", "/create",
        "/tn", task_name,
        "/tr", task_command,
        "/sc", scheduler_config.get('trigger', 'onstart'),
        "/ru", scheduler_config.get('user', 'SYSTEM'),
        "/rl", scheduler_config.get('priority', 'highest'),
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
        logger.info("Note: Task creation may require administrator privileges")
        return False


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="SO8T Master Automated Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/master_automated_pipeline.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Setup Windows Task Scheduler auto-start task"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the pipeline (called by task scheduler)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint file path"
    )
    
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Clear all checkpoints and start from the beginning"
    )
    
    parser.add_argument(
        "--start-from-phase",
        type=str,
        default=None,
        help="Start from a specific phase (e.g., 'phase1_web_scraping')"
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
        # パイプライン実行
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return 1
        
        # システム起動時の遅延処理（設定ファイルから取得）
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        delay_seconds = config.get('task_scheduler', {}).get('delay_seconds', 30)
        
        if delay_seconds > 0:
            logger.info(f"Waiting {delay_seconds} seconds before starting pipeline (system startup delay)...")
            time.sleep(delay_seconds)
        
        pipeline = MasterAutomatedPipeline(
            str(config_path),
            resume_from_checkpoint=args.resume,
            reset_checkpoint=args.reset_checkpoint,
            start_from_phase=args.start_from_phase
        )
        
        try:
            result = pipeline.run_complete_pipeline()
            logger.info("Master Automated Pipeline completed successfully!")
            return 0
            
        except KeyboardInterrupt:
            logger.warning("[WARNING] Pipeline interrupted by user")
            return 1
        except Exception as e:
            logger.error(f"[FAILED] Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

