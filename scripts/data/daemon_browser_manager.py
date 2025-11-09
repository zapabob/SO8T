#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
10個ブラウザバックグラウンドデーモン起動

Playwrightで10個のブラウザをバックグラウンドでデーモンとして起動し、
各ブラウザにリモートデバッグポートを割り当てて管理します。

Usage:
    from scripts.data.daemon_browser_manager import DaemonBrowserManager
    
    manager = DaemonBrowserManager(num_browsers=10, base_port=9222)
    await manager.start_all_browsers()
    browsers = await manager.get_all_browsers()
"""

import sys
import logging
import asyncio
import subprocess
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any
import os
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Playwrightインポート
try:
    from playwright.async_api import async_playwright, Browser, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.error("[ERROR] Playwright not installed. Install with: pip install playwright")

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/daemon_browser_manager.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# psutil警告をログに記録
if not PSUTIL_AVAILABLE:
    logger.warning("[WARNING] psutil not available, resource monitoring will be limited")


class DaemonBrowserManager:
    """10個ブラウザバックグラウンドデーモンマネージャー"""
    
    def __init__(
        self,
        num_browsers: int = 10,
        base_port: int = 9222,
        headless: bool = False,
        use_cursor_browser: bool = True,
        auto_restart: bool = True,
        restart_delay: float = 60.0,
        max_memory_gb: float = 8.0,
        max_cpu_percent: float = 80.0
    ):
        """
        初期化
        
        Args:
            num_browsers: ブラウザ数
            base_port: ベースリモートデバッグポート
            headless: ヘッドレスモード
            use_cursor_browser: Cursorブラウザを使用するか
            auto_restart: 自動再起動を有効にするか
            restart_delay: 再起動遅延（秒）
            max_memory_gb: 最大メモリ使用量（GB）
            max_cpu_percent: 最大CPU使用率（%）
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError("Playwright not available")
        
        self.num_browsers = num_browsers
        self.base_port = base_port
        self.headless = headless
        self.use_cursor_browser = use_cursor_browser
        self.auto_restart = auto_restart
        self.restart_delay = restart_delay
        self.max_memory_gb = max_memory_gb
        self.max_cpu_percent = max_cpu_percent
        
        # ブラウザ管理
        self.browsers: Dict[int, Browser] = {}
        self.browser_contexts: Dict[int, BrowserContext] = {}
        self.browser_status: Dict[int, Dict[str, Any]] = {}
        self.browser_processes: Dict[int, subprocess.Popen] = {}
        
        # Playwrightインスタンス
        self.playwright = None
        
        logger.info("="*80)
        logger.info("Daemon Browser Manager Initialized")
        logger.info("="*80)
        logger.info(f"Number of browsers: {self.num_browsers}")
        logger.info(f"Base port: {self.base_port}")
        logger.info(f"Headless: {self.headless}")
        logger.info(f"Use Cursor browser: {self.use_cursor_browser}")
        logger.info(f"Auto restart: {self.auto_restart}")
    
    async def initialize_playwright(self):
        """Playwrightを初期化"""
        if self.playwright is None:
            self.playwright = await async_playwright().start()
            logger.info("[PLAYWRIGHT] Playwright initialized")
    
    async def launch_browser_background(self, browser_index: int) -> bool:
        """
        ブラウザをバックグラウンドで起動
        
        Args:
            browser_index: ブラウザインデックス（0-9）
        
        Returns:
            success: 成功フラグ
        """
        if browser_index < 0 or browser_index >= self.num_browsers:
            logger.error(f"[BROWSER {browser_index}] Invalid browser index")
            return False
        
        port = self.base_port + browser_index
        
        try:
            logger.info(f"[BROWSER {browser_index}] Launching browser on port {port}...")
            
            if self.use_cursor_browser:
                # Cursorブラウザをバックグラウンドで起動
                success = await self._launch_cursor_browser_background(port)
                if not success:
                    logger.warning(f"[BROWSER {browser_index}] Failed to launch Cursor browser, falling back to regular browser")
                    self.use_cursor_browser = False
            
            if not self.use_cursor_browser:
                # 通常のブラウザを起動
                await self.initialize_playwright()
                browser = await self.playwright.chromium.launch(
                    headless=self.headless,
                    args=[
                        f"--remote-debugging-port={port}",
                        "--disable-blink-features=AutomationControlled",
                        "--disable-dev-shm-usage"
                    ]
                )
                self.browsers[browser_index] = browser
                logger.info(f"[OK] Browser {browser_index} launched on port {port}")
            
            # ブラウザに接続
            await self.initialize_playwright()
            browser = await self._connect_to_browser(port, browser_index)
            
            if browser is None:
                logger.error(f"[BROWSER {browser_index}] Failed to connect to browser")
                return False
            
            self.browsers[browser_index] = browser
            
            # ブラウザコンテキストを作成
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            self.browser_contexts[browser_index] = context
            
            # 状態を保存
            self.browser_status[browser_index] = {
                'browser_index': browser_index,
                'port': port,
                'started_at': datetime.now().isoformat(),
                'status': 'running',
                'headless': self.headless,
                'use_cursor_browser': self.use_cursor_browser
            }
            
            logger.info(f"[OK] Browser {browser_index} connected and ready")
            return True
            
        except Exception as e:
            logger.error(f"[BROWSER {browser_index}] Failed to launch: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    async def _launch_cursor_browser_background(self, port: int) -> bool:
        """
        Cursorブラウザをバックグラウンドで起動
        
        Args:
            port: リモートデバッグポート
        
        Returns:
            success: 成功フラグ
        """
        try:
            # Cursorブラウザのパスを検出
            cursor_browser_paths = [
                r"C:\Users\{}\AppData\Local\Programs\cursor\resources\app.asar.unpacked\node_modules\@cursor\browser\chrome-win\chrome.exe".format(os.getenv('USERNAME')),
                r"C:\Users\{}\AppData\Local\Programs\Cursor\resources\app.asar.unpacked\node_modules\@cursor\browser\chrome-win\chrome.exe".format(os.getenv('USERNAME')),
            ]
            
            cursor_browser_path = None
            for path in cursor_browser_paths:
                if Path(path).exists():
                    cursor_browser_path = path
                    break
            
            if cursor_browser_path is None:
                logger.warning("[CURSOR] Cursor browser path not found")
                return False
            
            # Cursorブラウザをバックグラウンドで起動
            cmd = [
                cursor_browser_path,
                f"--remote-debugging-port={port}",
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-first-run",
                "--no-default-browser-check"
            ]
            
            logger.info(f"[CURSOR] Launching Cursor browser in background on port {port}...")
            
            if platform.system() == "Windows":
                # WindowsではCREATE_NO_WINDOWフラグを使用してバックグラウンド起動
                subprocess.Popen(
                    cmd,
                    creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True
                )
            else:
                subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True
                )
            
            # ブラウザが起動するまで待機（最大10秒）
            for i in range(20):
                await asyncio.sleep(0.5)
                if await self._check_browser_running(port):
                    logger.info(f"[OK] Cursor browser launched successfully on port {port}")
                    return True
            
            logger.warning(f"[CURSOR] Cursor browser may not have started properly on port {port}")
            return False
            
        except Exception as e:
            logger.error(f"[CURSOR] Failed to launch Cursor browser: {e}")
            return False
    
    async def _check_browser_running(self, port: int) -> bool:
        """
        ブラウザが起動しているかチェック
        
        Args:
            port: リモートデバッグポート
        
        Returns:
            running: 起動中フラグ
        """
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://127.0.0.1:{port}/json/version", timeout=aiohttp.ClientTimeout(total=1)) as response:
                    return response.status == 200
        except Exception:
            return False
    
    async def _connect_to_browser(self, port: int, browser_index: int) -> Optional[Browser]:
        """
        ブラウザに接続
        
        Args:
            port: リモートデバッグポート
            browser_index: ブラウザインデックス
        
        Returns:
            browser: ブラウザインスタンス
        """
        try:
            await self.initialize_playwright()
            
            # CDPエンドポイントURL
            cdp_endpoint = f"http://127.0.0.1:{port}"
            
            browser = await self.playwright.chromium.connect_over_cdp(cdp_endpoint)
            
            # 接続確認
            contexts = browser.contexts
            if contexts:
                logger.info(f"[BROWSER {browser_index}] Connected (found {len(contexts)} contexts)")
            else:
                logger.info(f"[BROWSER {browser_index}] Connected (no existing contexts)")
            
            return browser
            
        except Exception as e:
            logger.error(f"[BROWSER {browser_index}] Failed to connect: {e}")
            return None
    
    async def start_all_browsers(self) -> bool:
        """
        すべてのブラウザを起動
        
        Returns:
            success: 成功フラグ
        """
        logger.info(f"[START] Starting all {self.num_browsers} browsers...")
        
        success_count = 0
        for browser_index in range(self.num_browsers):
            success = await self.launch_browser_background(browser_index)
            if success:
                success_count += 1
            # ブラウザ間で少し待機（リソース競合を避ける）
            await asyncio.sleep(1.0)
        
        logger.info(f"[OK] Started {success_count}/{self.num_browsers} browsers")
        return success_count == self.num_browsers
    
    async def stop_browser(self, browser_index: int) -> bool:
        """
        ブラウザを停止
        
        Args:
            browser_index: ブラウザインデックス
        
        Returns:
            success: 成功フラグ
        """
        if browser_index not in self.browsers:
            logger.warning(f"[BROWSER {browser_index}] Not started")
            return False
        
        try:
            logger.info(f"[BROWSER {browser_index}] Stopping browser...")
            
            # コンテキストを閉じる
            if browser_index in self.browser_contexts:
                await self.browser_contexts[browser_index].close()
                del self.browser_contexts[browser_index]
            
            # ブラウザを閉じる
            browser = self.browsers[browser_index]
            await browser.close()
            del self.browsers[browser_index]
            
            # 状態を更新
            if browser_index in self.browser_status:
                self.browser_status[browser_index]['status'] = 'stopped'
                self.browser_status[browser_index]['stopped_at'] = datetime.now().isoformat()
            
            logger.info(f"[OK] Browser {browser_index} stopped")
            return True
            
        except Exception as e:
            logger.error(f"[BROWSER {browser_index}] Failed to stop: {e}")
            return False
    
    async def stop_all_browsers(self) -> bool:
        """
        すべてのブラウザを停止
        
        Returns:
            success: 成功フラグ
        """
        logger.info("[STOP] Stopping all browsers...")
        
        success_count = 0
        for browser_index in list(self.browsers.keys()):
            success = await self.stop_browser(browser_index)
            if success:
                success_count += 1
        
        # Playwrightを終了
        if self.playwright:
            await self.playwright.stop()
            self.playwright = None
        
        logger.info(f"[OK] Stopped {success_count} browsers")
        return True
    
    def get_browser(self, browser_index: int) -> Optional[Browser]:
        """
        ブラウザを取得
        
        Args:
            browser_index: ブラウザインデックス
        
        Returns:
            browser: ブラウザインスタンス
        """
        return self.browsers.get(browser_index)
    
    def get_browser_context(self, browser_index: int) -> Optional[BrowserContext]:
        """
        ブラウザコンテキストを取得
        
        Args:
            browser_index: ブラウザインデックス
        
        Returns:
            context: ブラウザコンテキスト
        """
        return self.browser_contexts.get(browser_index)
    
    def get_all_browsers(self) -> Dict[int, Browser]:
        """
        すべてのブラウザを取得
        
        Returns:
            browsers: すべてのブラウザ
        """
        return self.browsers.copy()
    
    def get_browser_status(self, browser_index: int) -> Optional[Dict[str, Any]]:
        """
        ブラウザの状態を取得
        
        Args:
            browser_index: ブラウザインデックス
        
        Returns:
            status: ブラウザ状態
        """
        return self.browser_status.get(browser_index)
    
    def get_all_browsers_status(self) -> Dict[int, Dict[str, Any]]:
        """
        すべてのブラウザの状態を取得
        
        Returns:
            statuses: すべてのブラウザ状態
        """
        return self.browser_status.copy()
    
    async def monitor_resources(self):
        """
        リソース監視（メモリ、CPU使用率）
        """
        if not PSUTIL_AVAILABLE:
            logger.warning("[MONITOR] psutil not available, skipping resource monitoring")
            return
        
        while True:
            try:
                for browser_index, browser in self.browsers.items():
                    # リソース使用状況を取得（簡易版）
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    cpu_percent = process.cpu_percent(interval=1)
                    
                    if browser_index in self.browser_status:
                        self.browser_status[browser_index]['memory_mb'] = memory_mb
                        self.browser_status[browser_index]['cpu_percent'] = cpu_percent
                    
                    # リソース制限チェック
                    if memory_mb > self.max_memory_gb * 1024:
                        logger.warning(f"[BROWSER {browser_index}] Memory limit exceeded: {memory_mb:.1f}MB > {self.max_memory_gb * 1024}MB")
                        if self.auto_restart:
                            logger.info(f"[BROWSER {browser_index}] Auto-restarting due to memory limit...")
                            await self.stop_browser(browser_index)
                            await asyncio.sleep(self.restart_delay)
                            await self.launch_browser_background(browser_index)
                    
                    if cpu_percent > self.max_cpu_percent:
                        logger.warning(f"[BROWSER {browser_index}] CPU limit exceeded: {cpu_percent:.1f}% > {self.max_cpu_percent}%")
                
                await asyncio.sleep(10)  # 10秒ごとにチェック
                
            except Exception as e:
                logger.error(f"[MONITOR] Resource monitoring error: {e}")
                await asyncio.sleep(10)


async def main():
    """メイン関数（テスト用）"""
    manager = DaemonBrowserManager(num_browsers=10, base_port=9222)
    
    try:
        # すべてのブラウザを起動
        await manager.start_all_browsers()
        
        # 状態を確認
        statuses = manager.get_all_browsers_status()
        logger.info(f"[STATUS] Active browsers: {len(statuses)}")
        
        # リソース監視を開始（バックグラウンド）
        monitor_task = asyncio.create_task(manager.monitor_resources())
        
        # 30秒待機
        await asyncio.sleep(30)
        
        # 監視タスクをキャンセル
        monitor_task.cancel()
        
        # すべてのブラウザを停止
        await manager.stop_all_browsers()
        
    except KeyboardInterrupt:
        logger.warning("[INTERRUPT] Interrupted by user")
        await manager.stop_all_browsers()
    except Exception as e:
        logger.error(f"[ERROR] Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        await manager.stop_all_browsers()


if __name__ == "__main__":
    asyncio.run(main())

