#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学習とクロールの並列実行スクリプト

学習処理（train_so8t_qwen.py）とクロール処理（collect_japanese_data.py）を
並列実行し、進捗を監視・管理します。

Usage:
    python scripts/run_parallel_train_crawl.py --train-config configs/training_config.yaml --crawl-target 10000
"""

import os
import sys
import json
import time
import signal
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
import argparse

from tqdm import tqdm

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/parallel_train_crawl.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 音声ファイルパス
AUDIO_FILE = Path("C:/Users/downl/Desktop/SO8T/.cursor/marisa_owattaze.wav")

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class ProcessStatus:
    """プロセス状態"""
    name: str
    pid: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    status: str = "pending"  # pending, running, completed, failed
    error: Optional[str] = None
    progress: float = 0.0
    
    def to_dict(self):
        return asdict(self)


class ParallelTrainCrawl:
    """学習とクロールの並列実行管理"""
    
    def __init__(
        self,
        train_config: str = "configs/training_config.yaml",
        crawl_target: int = 10000,
        crawl_workers: int = 4,
        enable_web_crawl: bool = True,
        auto_resume: bool = True
    ):
        self.train_config = train_config
        self.crawl_target = crawl_target
        self.crawl_workers = crawl_workers
        self.enable_web_crawl = enable_web_crawl
        self.auto_resume = auto_resume
        
        # プロセス状態
        self.train_status = ProcessStatus(name="training")
        self.crawl_status = ProcessStatus(name="crawling")
        
        # プロセスオブジェクト
        self.train_process: Optional[subprocess.Popen] = None
        self.crawl_process: Optional[subprocess.Popen] = None
        
        # ログディレクトリ
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # ドキュメントディレクトリ
        self.docs_dir = Path("_docs")
        self.docs_dir.mkdir(exist_ok=True)
        
        # シグナルハンドラー設定
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """シグナルハンドラー（電源断対応）"""
        logger.warning(f"[WARNING] Signal {signum} received. Terminating processes...")
        self.cleanup()
        sys.exit(0)
    
    def _run_training(self) -> Tuple[int, Optional[str]]:
        """学習プロセス実行"""
        try:
            train_script = PROJECT_ROOT / "scripts" / "train_so8t_qwen.py"
            if not train_script.exists():
                return -1, f"Training script not found: {train_script}"
            
            logger.info(f"[TRAIN] Starting training process...")
            logger.info(f"[TRAIN] Config: {self.train_config}")
            
            # 学習プロセス起動
            cmd = [
                sys.executable, str(train_script),
                "--config", str(self.train_config)
            ]
            
            log_file = self.log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            with open(log_file, 'w', encoding='utf-8') as f:
                self.train_process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=str(PROJECT_ROOT),
                    encoding='utf-8'
                )
            
            self.train_status.pid = self.train_process.pid
            self.train_status.start_time = time.time()
            self.train_status.status = "running"
            
            logger.info(f"[TRAIN] Process started (PID: {self.train_process.pid})")
            logger.info(f"[TRAIN] Log file: {log_file}")
            
            # プロセス完了待機
            return_code = self.train_process.wait()
            
            self.train_status.end_time = time.time()
            
            if return_code == 0:
                self.train_status.status = "completed"
                logger.info(f"[TRAIN] Process completed successfully")
                return 0, None
            else:
                self.train_status.status = "failed"
                error_msg = f"Training process exited with code {return_code}"
                self.train_status.error = error_msg
                logger.error(f"[TRAIN] {error_msg}")
                return return_code, error_msg
                
        except Exception as e:
            self.train_status.status = "failed"
            self.train_status.error = str(e)
            logger.error(f"[TRAIN] Error: {e}")
            return -1, str(e)
    
    def _run_crawling(self) -> Tuple[int, Optional[str]]:
        """クロールプロセス実行"""
        try:
            crawl_script = PROJECT_ROOT / "so8t-mmllm" / "scripts" / "data" / "collect_japanese_data.py"
            if not crawl_script.exists():
                return -1, f"Crawl script not found: {crawl_script}"
            
            logger.info(f"[CRAWL] Starting crawling process...")
            logger.info(f"[CRAWL] Target: {self.crawl_target:,} samples")
            
            # クロールプロセス起動
            cmd = [
                sys.executable, str(crawl_script),
                "--target", str(self.crawl_target),
                "--workers", str(self.crawl_workers)
            ]
            
            if self.enable_web_crawl:
                cmd.append("--web-crawl")
            else:
                cmd.append("--no-web-crawl")
            
            if self.auto_resume:
                cmd.append("--auto-resume")
            
            log_file = self.log_dir / f"crawl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            with open(log_file, 'w', encoding='utf-8') as f:
                self.crawl_process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=str(PROJECT_ROOT),
                    encoding='utf-8'
                )
            
            self.crawl_status.pid = self.crawl_process.pid
            self.crawl_status.start_time = time.time()
            self.crawl_status.status = "running"
            
            logger.info(f"[CRAWL] Process started (PID: {self.crawl_process.pid})")
            logger.info(f"[CRAWL] Log file: {log_file}")
            
            # プロセス完了待機
            return_code = self.crawl_process.wait()
            
            self.crawl_status.end_time = time.time()
            
            if return_code == 0:
                self.crawl_status.status = "completed"
                logger.info(f"[CRAWL] Process completed successfully")
                return 0, None
            else:
                self.crawl_status.status = "failed"
                error_msg = f"Crawl process exited with code {return_code}"
                self.crawl_status.error = error_msg
                logger.error(f"[CRAWL] {error_msg}")
                return return_code, error_msg
                
        except Exception as e:
            self.crawl_status.status = "failed"
            self.crawl_status.error = str(e)
            logger.error(f"[CRAWL] Error: {e}")
            return -1, str(e)
    
    def _monitor_processes(self):
        """プロセス監視（進捗表示）"""
        logger.info("[MONITOR] Starting process monitoring...")
        
        # プログレスバー設定
        with tqdm(total=2, desc="Parallel Execution", unit="process") as pbar:
            while (self.train_process and self.train_process.poll() is None) or \
                  (self.crawl_process and self.crawl_process.poll() is None):
                time.sleep(1)
                
                # プロセス状態更新
                if self.train_process and self.train_process.poll() is not None:
                    if self.train_status.status == "running":
                        self.train_status.status = "completed"
                        pbar.update(1)
                        pbar.set_postfix({"train": "completed"})
                
                if self.crawl_process and self.crawl_process.poll() is not None:
                    if self.crawl_status.status == "running":
                        self.crawl_status.status = "completed"
                        pbar.update(1)
                        pbar.set_postfix({"crawl": "completed"})
        
        logger.info("[MONITOR] All processes completed")
    
    def run_parallel(self):
        """並列実行メイン関数"""
        logger.info("="*80)
        logger.info("SO8T Parallel Train & Crawl")
        logger.info("="*80)
        logger.info(f"Training Config: {self.train_config}")
        logger.info(f"Crawl Target: {self.crawl_target:,} samples")
        logger.info(f"Crawl Workers: {self.crawl_workers}")
        logger.info(f"Web Crawl: {'Enabled' if self.enable_web_crawl else 'Disabled'}")
        logger.info(f"Auto Resume: {'Enabled' if self.auto_resume else 'Disabled'}")
        logger.info("="*80)
        
        start_time = time.time()
        
        try:
            logger.info("[START] Starting parallel execution...")
            
            # 学習プロセスを別スレッドで起動
            def train_thread():
                self._run_training()
            
            # クロールプロセスを別スレッドで起動
            def crawl_thread():
                self._run_crawling()
            
            # スレッド起動
            train_thread_obj = threading.Thread(target=train_thread, name="TrainThread", daemon=False)
            crawl_thread_obj = threading.Thread(target=crawl_thread, name="CrawlThread", daemon=False)
            
            train_thread_obj.start()
            crawl_thread_obj.start()
            
            logger.info("[START] Both processes started")
            
            # 監視
            self._monitor_processes()
            
            # スレッド完了待機
            train_thread_obj.join()
            crawl_thread_obj.join()
            
            # 結果確認
            train_success = self.train_status.status == "completed"
            crawl_success = self.crawl_status.status == "completed"
            
            end_time = time.time()
            total_time = end_time - start_time
            
            logger.info("="*80)
            logger.info("Execution Summary")
            logger.info("="*80)
            logger.info(f"Training: {self.train_status.status}")
            if self.train_status.error:
                logger.info(f"  Error: {self.train_status.error}")
            if self.train_status.start_time and self.train_status.end_time:
                train_duration = self.train_status.end_time - self.train_status.start_time
                logger.info(f"  Duration: {train_duration/3600:.2f} hours")
            
            logger.info(f"Crawling: {self.crawl_status.status}")
            if self.crawl_status.error:
                logger.info(f"  Error: {self.crawl_status.error}")
            if self.crawl_status.start_time and self.crawl_status.end_time:
                crawl_duration = self.crawl_status.end_time - self.crawl_status.start_time
                logger.info(f"  Duration: {crawl_duration/3600:.2f} hours")
            
            logger.info(f"Total Time: {total_time/3600:.2f} hours")
            logger.info("="*80)
            
            # レポート生成
            self._generate_report(total_time, train_success, crawl_success)
            
            # 音声通知
            self._play_completion_sound()
            
            return train_success and crawl_success
            
        except KeyboardInterrupt:
            logger.warning("\n[WARNING] Interrupted by user")
            self.cleanup()
            return False
        except Exception as e:
            logger.error(f"\n[ERROR] Parallel execution failed: {e}")
            self.cleanup()
            raise
    
    def cleanup(self):
        """クリーンアップ（プロセス終了）"""
        logger.info("[CLEANUP] Terminating processes...")
        
        if self.train_process and self.train_process.poll() is None:
            self.train_process.terminate()
            try:
                self.train_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.train_process.kill()
        
        if self.crawl_process and self.crawl_process.poll() is None:
            self.crawl_process.terminate()
            try:
                self.crawl_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.crawl_process.kill()
        
        logger.info("[CLEANUP] Cleanup completed")
    
    def _generate_report(self, total_time: float, train_success: bool, crawl_success: bool):
        """完了レポート生成"""
        report_file = self.docs_dir / f"{datetime.now().strftime('%Y-%m-%d')}_parallel_train_crawl.md"
        
        report = f"""# 学習とクロール並列実行レポート

## 実行概要
- **実行日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **総実行時間**: {total_time/3600:.2f}時間
- **学習設定**: {self.train_config}
- **クロール目標**: {self.crawl_target:,}サンプル

## 実行結果

### 学習プロセス
- **ステータス**: {self.train_status.status}
- **PID**: {self.train_status.pid}
"""
        
        if self.train_status.start_time and self.train_status.end_time:
            train_duration = self.train_status.end_time - self.train_status.start_time
            report += f"- **実行時間**: {train_duration/3600:.2f}時間\n"
        
        if self.train_status.error:
            report += f"- **エラー**: {self.train_status.error}\n"
        
        report += f"""
### クロールプロセス
- **ステータス**: {self.crawl_status.status}
- **PID**: {self.crawl_status.pid}
"""
        
        if self.crawl_status.start_time and self.crawl_status.end_time:
            crawl_duration = self.crawl_status.end_time - self.crawl_status.start_time
            report += f"- **実行時間**: {crawl_duration/3600:.2f}時間\n"
        
        if self.crawl_status.error:
            report += f"- **エラー**: {self.crawl_status.error}\n"
        
        report += f"""
## 設定
- **クロールワーカー数**: {self.crawl_workers}
- **Webクロール**: {'有効' if self.enable_web_crawl else '無効'}
- **自動再開**: {'有効' if self.auto_resume else '無効'}

## 結果サマリー
- **学習**: {'[OK] 成功' if train_success else '[NG] 失敗'}
- **クロール**: {'[OK] 成功' if crawl_success else '[NG] 失敗'}
- **全体**: {'[OK] 成功' if (train_success and crawl_success) else '[NG] 一部失敗'}

## ログファイル
- 学習ログ: `logs/train_*.log`
- クロールログ: `logs/crawl_*.log`
- 統合ログ: `logs/parallel_train_crawl.log`
"""
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"[REPORT] Report saved to {report_file}")
    
    def _play_completion_sound(self):
        """完了音声通知"""
        if not AUDIO_FILE.exists():
            logger.warning(f"[AUDIO] Audio file not found: {AUDIO_FILE}")
            return
        
        try:
            logger.info("[AUDIO] Playing completion notification...")
            
            # PowerShellで音声再生
            ps_cmd = f"""
            if (Test-Path '{AUDIO_FILE}') {{
                Add-Type -AssemblyName System.Windows.Forms
                $player = New-Object System.Media.SoundPlayer '{AUDIO_FILE}'
                $player.PlaySync()
                Write-Host '[OK] Audio notification played successfully' -ForegroundColor Green
            }} else {{
                Write-Host '[WARNING] Audio file not found' -ForegroundColor Yellow
            }}
            """
            
            subprocess.run(
                ["powershell", "-Command", ps_cmd],
                cwd=str(PROJECT_ROOT),
                check=False
            )
            
            logger.info("[AUDIO] Audio notification completed")
            
        except Exception as e:
            logger.warning(f"[AUDIO] Failed to play audio: {e}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Parallel Training and Crawling Execution"
    )
    parser.add_argument(
        "--train-config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--crawl-target",
        type=int,
        default=10000,
        help="Target number of samples for crawling"
    )
    parser.add_argument(
        "--crawl-workers",
        type=int,
        default=4,
        help="Number of workers for crawling"
    )
    parser.add_argument(
        "--no-web-crawl",
        action="store_true",
        help="Disable web crawling"
    )
    parser.add_argument(
        "--no-auto-resume",
        action="store_true",
        help="Disable auto-resume for crawling"
    )
    
    args = parser.parse_args()
    
    # 並列実行オブジェクト作成
    runner = ParallelTrainCrawl(
        train_config=args.train_config,
        crawl_target=args.crawl_target,
        crawl_workers=args.crawl_workers,
        enable_web_crawl=not args.no_web_crawl,
        auto_resume=not args.no_auto_resume
    )
    
    # 実行
    try:
        success = runner.run_parallel()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"[ERROR] Execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

