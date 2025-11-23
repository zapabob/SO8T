#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO8T統制によるArxiv・オープンアクセス論文完全自動バックグラウンドスクレイピング

SO8Tモデルの四重推論を使って、完全自動でバックグラウンド実行される
Arxiv・オープンアクセス論文スクレイピングスクリプト。

Usage:
    python scripts/data/so8t_arxiv_background_scraping.py --daemon
"""

import sys
import json
import logging
import asyncio
import argparse
import signal
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import subprocess
import os

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Arxivスクレイピングスクリプトをインポート
from scripts.data.arxiv_open_access_scraping import ArxivOpenAccessScraper

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/so8t_arxiv_background_scraping.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SO8TArxivBackgroundScraper:
    """SO8T統制による完全自動バックグラウンドArxivスクレイパー"""
    
    def __init__(
        self,
        output_dir: Path,
        config_file: Optional[Path] = None,
        daemon_mode: bool = False,
        auto_restart: bool = True,
        max_restarts: int = 10,
        restart_delay: float = 3600.0  # 1時間
    ):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリ
            config_file: 設定ファイルパス
            daemon_mode: デーモンモード（バックグラウンド実行）
            auto_restart: 自動再起動
            max_restarts: 最大再起動回数
            restart_delay: 再起動待機時間（秒）
        """
        self.output_dir = Path(output_dir)
        self.config_file = config_file
        self.daemon_mode = daemon_mode
        self.auto_restart = auto_restart
        self.max_restarts = max_restarts
        self.restart_delay = restart_delay
        
        self.restart_count = 0
        self.running = True
        self.scraper = None
        
        # シグナルハンドラー設定
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, self._signal_handler)
        
        logger.info("="*80)
        logger.info("SO8T Arxiv Background Scraper Initialized")
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Daemon mode: {self.daemon_mode}")
        logger.info(f"Auto restart: {self.auto_restart}")
    
    def _signal_handler(self, signum, frame):
        """シグナルハンドラー"""
        logger.info(f"[SIGNAL] Received signal {signum}, shutting down gracefully...")
        self.running = False
        if self.scraper:
            pass
    
    def load_config(self) -> Dict:
        """設定ファイルを読み込み"""
        if self.config_file and self.config_file.exists():
            import yaml
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"[CONFIG] Loaded config from: {self.config_file}")
            return config
        else:
            # デフォルト設定
            return {
                'use_cursor_browser': True,
                'remote_debugging_port': 9222,
                'delay_per_action': 2.0,
                'timeout': 30000,
                'max_papers_per_category': 50,
                'use_so8t_control': True,
                'so8t_model_path': None,
                'output_dir': str(self.output_dir)
            }
    
    async def run_scraping_session(self) -> bool:
        """スクレイピングセッションを実行"""
        try:
            # 設定を読み込み
            config = self.load_config()
            
            # スクレイパーを作成
            self.scraper = ArxivOpenAccessScraper(
                output_dir=Path(config.get('output_dir', self.output_dir)),
                use_cursor_browser=config.get('use_cursor_browser', True),
                remote_debugging_port=config.get('remote_debugging_port', 9222),
                delay_per_action=config.get('delay_per_action', 2.0),
                timeout=config.get('timeout', 30000),
                max_papers_per_category=config.get('max_papers_per_category', 50),
                use_so8t_control=config.get('use_so8t_control', True),
                so8t_model_path=config.get('so8t_model_path', None)
            )
            
            # スクレイピング実行
            logger.info("[SESSION] Starting scraping session...")
            await self.scraper.run_scraping()
            
            # 結果を保存
            output_file = self.scraper.save_papers()
            
            logger.info(f"[SESSION] Scraping session completed. Output: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"[SESSION] Scraping session failed: {e}", exc_info=True)
            return False
    
    async def run_continuous(self):
        """連続実行ループ"""
        logger.info("[CONTINUOUS] Starting continuous scraping loop...")
        
        while self.running:
            try:
                # スクレイピングセッションを実行
                success = await self.run_scraping_session()
                
                if not success:
                    self.restart_count += 1
                    if self.restart_count >= self.max_restarts:
                        logger.error(f"[CONTINUOUS] Max restarts ({self.max_restarts}) reached, stopping...")
                        break
                    
                    if self.auto_restart:
                        logger.info(f"[CONTINUOUS] Restarting in {self.restart_delay} seconds... (attempt {self.restart_count}/{self.max_restarts})")
                        await asyncio.sleep(self.restart_delay)
                    else:
                        break
                else:
                    # 成功時は再起動カウントをリセット
                    self.restart_count = 0
                    
                    # 次のセッションまでの待機（24時間）
                    if self.running:
                        wait_time = 86400.0  # 24時間待機
                        logger.info(f"[CONTINUOUS] Waiting {wait_time/3600:.1f} hours before next session...")
                        await asyncio.sleep(wait_time)
                
            except KeyboardInterrupt:
                logger.info("[CONTINUOUS] Keyboard interrupt received, stopping...")
                self.running = False
                break
            except Exception as e:
                logger.error(f"[CONTINUOUS] Unexpected error: {e}", exc_info=True)
                self.restart_count += 1
                if self.restart_count >= self.max_restarts:
                    logger.error(f"[CONTINUOUS] Max restarts reached, stopping...")
                    break
                
                if self.auto_restart:
                    logger.info(f"[CONTINUOUS] Restarting in {self.restart_delay} seconds...")
                    await asyncio.sleep(self.restart_delay)
                else:
                    break
        
        logger.info("[CONTINUOUS] Continuous scraping loop stopped")
    
    def run_as_daemon(self):
        """デーモンとして実行（Windows対応）"""
        if self.daemon_mode:
            logger.info("[DAEMON] Running as daemon (background process)...")
        
        # 連続実行ループを開始（既存のイベントループを使用）
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 既にイベントループが実行中の場合は、タスクとして実行
                loop.create_task(self.run_continuous())
            else:
                # イベントループが実行されていない場合は、run()を使用
                asyncio.run(self.run_continuous())
        except RuntimeError:
            # イベントループが存在しない場合は、run()を使用
            asyncio.run(self.run_continuous())


async def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="SO8T Arxiv Background Scraping")
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('D:/webdataset/processed'),
        help='Output directory'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help='Configuration file path'
    )
    parser.add_argument(
        '--daemon',
        action='store_true',
        default=False,
        help='Run as daemon (background process)'
    )
    parser.add_argument(
        '--auto-restart',
        action='store_true',
        default=True,
        help='Auto restart on failure'
    )
    parser.add_argument(
        '--max-restarts',
        type=int,
        default=10,
        help='Maximum restart attempts'
    )
    parser.add_argument(
        '--restart-delay',
        type=float,
        default=3600.0,
        help='Delay between restarts (seconds)'
    )
    
    args = parser.parse_args()
    
    # スクレイパー作成
    scraper = SO8TArxivBackgroundScraper(
        output_dir=args.output,
        config_file=args.config,
        daemon_mode=args.daemon,
        auto_restart=args.auto_restart,
        max_restarts=args.max_restarts,
        restart_delay=args.restart_delay
    )
    
    # 実行
    if args.daemon:
        scraper.run_as_daemon()
    else:
        await scraper.run_continuous()


if __name__ == "__main__":
    asyncio.run(main())

