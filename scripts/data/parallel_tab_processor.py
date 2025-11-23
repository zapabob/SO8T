#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
10個タブ並列処理機能

各ブラウザで10個のタブを並列処理し、タブごとの独立した処理フローを実現します。

Usage:
    from scripts.data.parallel_tab_processor import ParallelTabProcessor
    
    processor = ParallelTabProcessor(browser_context, num_tabs=10)
    await processor.process_tabs(urls)
"""

import sys
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Callable
from urllib.parse import urljoin

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Playwrightインポート
try:
    from playwright.async_api import BrowserContext, Page, TimeoutError as PlaywrightTimeoutError
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.error("[ERROR] Playwright not installed. Install with: pip install playwright")

# BeautifulSoupインポート
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("[WARNING] BeautifulSoup not installed. Install with: pip install beautifulsoup4")

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/parallel_tab_processor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ParallelTabProcessor:
    """10個タブ並列処理クラス"""
    
    def __init__(
        self,
        browser_context: BrowserContext,
        num_tabs: int = 10,
        delay_per_action: float = 1.5,
        timeout: int = 30000,
        max_retries: int = 3
    ):
        """
        初期化
        
        Args:
            browser_context: ブラウザコンテキスト
            num_tabs: タブ数
            delay_per_action: アクション間の遅延（秒）
            timeout: タイムアウト（ミリ秒）
            max_retries: 最大リトライ回数
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError("Playwright not available")
        
        self.browser_context = browser_context
        self.num_tabs = num_tabs
        self.delay_per_action = delay_per_action
        self.timeout = timeout
        self.max_retries = max_retries
        
        # タブ管理
        self.tabs: List[Page] = []
        self.tab_status: Dict[int, Dict[str, Any]] = {}
        self.visited_urls: Set[str] = set()
        
        logger.info("="*80)
        logger.info("Parallel Tab Processor Initialized")
        logger.info("="*80)
        logger.info(f"Number of tabs: {self.num_tabs}")
        logger.info(f"Delay per action: {self.delay_per_action}s")
        logger.info(f"Timeout: {self.timeout}ms")
    
    async def initialize_tabs(self) -> bool:
        """
        タブを初期化
        
        Returns:
            success: 成功フラグ
        """
        try:
            logger.info(f"[TABS] Initializing {self.num_tabs} tabs...")
            
            for tab_index in range(self.num_tabs):
                page = await self.browser_context.new_page()
                self.tabs.append(page)
                
                self.tab_status[tab_index] = {
                    'tab_index': tab_index,
                    'status': 'ready',
                    'current_url': None,
                    'created_at': datetime.now().isoformat(),
                    'processed_count': 0,
                    'error_count': 0
                }
                
                logger.info(f"[TAB {tab_index}] Initialized")
            
            logger.info(f"[OK] Initialized {len(self.tabs)} tabs")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize tabs: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    async def process_tab(
        self,
        tab_index: int,
        urls: List[str],
        process_func: Optional[Callable] = None
    ) -> List[Dict]:
        """
        タブでURLを処理
        
        Args:
            tab_index: タブインデックス
            urls: 処理するURLのリスト
            process_func: カスタム処理関数
        
        Returns:
            results: 処理結果のリスト
        """
        if tab_index >= len(self.tabs):
            logger.error(f"[TAB {tab_index}] Tab not initialized")
            return []
        
        page = self.tabs[tab_index]
        results = []
        
        try:
            self.tab_status[tab_index]['status'] = 'processing'
            
            for url in urls:
                if url in self.visited_urls:
                    logger.debug(f"[TAB {tab_index}] URL already visited: {url}")
                    continue
                
                try:
                    # URLを処理
                    result = await self._process_url(page, tab_index, url, process_func)
                    if result:
                        results.append(result)
                        self.visited_urls.add(url)
                        self.tab_status[tab_index]['processed_count'] += 1
                    
                    # アクション間の遅延
                    await asyncio.sleep(self.delay_per_action)
                    
                except Exception as e:
                    logger.error(f"[TAB {tab_index}] Failed to process URL {url}: {e}")
                    self.tab_status[tab_index]['error_count'] += 1
                    
                    # リトライ
                    if self.tab_status[tab_index]['error_count'] <= self.max_retries:
                        logger.info(f"[TAB {tab_index}] Retrying... ({self.tab_status[tab_index]['error_count']}/{self.max_retries})")
                        await asyncio.sleep(self.delay_per_action * 2)
                        continue
                    else:
                        logger.warning(f"[TAB {tab_index}] Max retries reached, skipping URL")
            
            self.tab_status[tab_index]['status'] = 'completed'
            
        except Exception as e:
            logger.error(f"[TAB {tab_index}] Tab processing failed: {e}")
            self.tab_status[tab_index]['status'] = 'error'
            import traceback
            logger.error(traceback.format_exc())
        
        return results
    
    async def _process_url(
        self,
        page: Page,
        tab_index: int,
        url: str,
        process_func: Optional[Callable] = None
    ) -> Optional[Dict]:
        """
        URLを処理
        
        Args:
            page: ページインスタンス
            tab_index: タブインデックス
            url: 処理するURL
            process_func: カスタム処理関数
        
        Returns:
            result: 処理結果
        """
        try:
            logger.info(f"[TAB {tab_index}] Processing URL: {url}")
            
            # ページに移動
            await page.goto(url, timeout=self.timeout, wait_until='networkidle')
            
            # 現在のURLを更新
            current_url = page.url
            self.tab_status[tab_index]['current_url'] = current_url
            
            # カスタム処理関数が指定されている場合はそれを使用
            if process_func:
                result = await process_func(page, tab_index, url)
            else:
                # デフォルト処理: ページコンテンツを取得
                result = await self._extract_page_content(page, tab_index, url)
            
            return result
            
        except PlaywrightTimeoutError:
            logger.warning(f"[TAB {tab_index}] Timeout while processing URL: {url}")
            return None
        except Exception as e:
            logger.error(f"[TAB {tab_index}] Error processing URL {url}: {e}")
            return None
    
    async def _extract_page_content(
        self,
        page: Page,
        tab_index: int,
        url: str
    ) -> Dict:
        """
        ページコンテンツを抽出
        
        Args:
            page: ページインスタンス
            tab_index: タブインデックス
            url: URL
        
        Returns:
            content: 抽出されたコンテンツ
        """
        try:
            # ページのHTMLを取得
            html = await page.content()
            
            # BeautifulSoupでパース
            if BS4_AVAILABLE:
                soup = BeautifulSoup(html, 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
                title = soup.title.string if soup.title else ""
            else:
                text = html
                title = ""
            
            # メタデータを取得
            metadata = {
                'url': url,
                'title': title,
                'timestamp': datetime.now().isoformat(),
                'tab_index': tab_index
            }
            
            # リンクを抽出
            links = []
            if BS4_AVAILABLE:
                for link in soup.find_all('a', href=True):
                    href = link.get('href')
                    if href:
                        absolute_url = urljoin(url, href)
                        links.append(absolute_url)
            
            result = {
                'url': url,
                'title': title,
                'text': text[:10000],  # テキストを10000文字に制限
                'links': links[:100],  # リンクを100個に制限
                'metadata': metadata
            }
            
            return result
            
        except Exception as e:
            logger.error(f"[TAB {tab_index}] Failed to extract content: {e}")
            return {
                'url': url,
                'error': str(e),
                'metadata': {
                    'url': url,
                    'timestamp': datetime.now().isoformat(),
                    'tab_index': tab_index
                }
            }
    
    async def process_tabs_parallel(
        self,
        urls: List[str],
        process_func: Optional[Callable] = None
    ) -> List[Dict]:
        """
        タブを並列処理
        
        Args:
            urls: 処理するURLのリスト
            process_func: カスタム処理関数
        
        Returns:
            results: すべての処理結果
        """
        if not self.tabs:
            logger.warning("[TABS] Tabs not initialized, initializing now...")
            await self.initialize_tabs()
        
        # URLをタブ数で分割
        urls_per_tab = len(urls) // self.num_tabs
        url_chunks = [urls[i:i + urls_per_tab] for i in range(0, len(urls), urls_per_tab)]
        
        # 余りのURLを最初のタブに追加
        if len(url_chunks) > self.num_tabs:
            url_chunks[self.num_tabs - 1].extend(url_chunks[self.num_tabs:])
            url_chunks = url_chunks[:self.num_tabs]
        
        # タブ数よりURLチャンクが少ない場合は空リストで埋める
        while len(url_chunks) < self.num_tabs:
            url_chunks.append([])
        
        logger.info(f"[TABS] Processing {len(urls)} URLs across {self.num_tabs} tabs")
        logger.info(f"[TABS] URLs per tab: {[len(chunk) for chunk in url_chunks]}")
        
        # すべてのタブを並列処理
        tasks = []
        for tab_index, url_chunk in enumerate(url_chunks):
            if url_chunk:
                task = self.process_tab(tab_index, url_chunk, process_func)
                tasks.append(task)
        
        # 並列実行
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 結果を統合
        all_results = []
        for results in results_list:
            if isinstance(results, Exception):
                logger.error(f"[TABS] Tab processing exception: {results}")
            elif isinstance(results, list):
                all_results.extend(results)
        
        logger.info(f"[OK] Processed {len(all_results)} results from {self.num_tabs} tabs")
        return all_results
    
    async def close_tabs(self):
        """タブを閉じる"""
        logger.info(f"[TABS] Closing {len(self.tabs)} tabs...")
        
        for tab_index, page in enumerate(self.tabs):
            try:
                await page.close()
                logger.debug(f"[TAB {tab_index}] Closed")
            except Exception as e:
                logger.warning(f"[TAB {tab_index}] Failed to close: {e}")
        
        self.tabs.clear()
        self.tab_status.clear()
        logger.info("[OK] All tabs closed")
    
    def get_tab_status(self, tab_index: int) -> Optional[Dict[str, Any]]:
        """
        タブの状態を取得
        
        Args:
            tab_index: タブインデックス
        
        Returns:
            status: タブ状態
        """
        return self.tab_status.get(tab_index)
    
    def get_all_tabs_status(self) -> Dict[int, Dict[str, Any]]:
        """
        すべてのタブの状態を取得
        
        Returns:
            statuses: すべてのタブ状態
        """
        return self.tab_status.copy()


async def main():
    """メイン関数（テスト用）"""
    from scripts.data.daemon_browser_manager import DaemonBrowserManager
    
    manager = DaemonBrowserManager(num_browsers=1, base_port=9222)
    
    try:
        # ブラウザを起動
        await manager.start_all_browsers()
        
        # ブラウザコンテキストを取得
        browser_context = manager.get_browser_context(0)
        if browser_context is None:
            logger.error("[ERROR] Failed to get browser context")
            return
        
        # タブプロセッサーを初期化
        processor = ParallelTabProcessor(browser_context, num_tabs=10)
        await processor.initialize_tabs()
        
        # テストURL
        test_urls = [
            "https://example.com",
            "https://www.google.com",
            "https://github.com"
        ] * 5  # 15個のURL
        
        # 並列処理
        results = await processor.process_tabs_parallel(test_urls)
        
        logger.info(f"[OK] Processed {len(results)} results")
        
        # タブを閉じる
        await processor.close_tabs()
        
        # ブラウザを停止
        await manager.stop_all_browsers()
        
    except Exception as e:
        logger.error(f"[ERROR] Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        await manager.stop_all_browsers()


if __name__ == "__main__":
    asyncio.run(main())

