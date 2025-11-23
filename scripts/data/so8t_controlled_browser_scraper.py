#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
SO8T統制機能統合

ScrapingReasoningAgentを使用したSO8T統制により、各タブでのスクレイピング判断を
SO8Tで実行し、四重推論と四値分類による判断結果に基づいてアクションを制御します。

Usage:
    from scripts.data.so8t_controlled_browser_scraper import SO8TControlledBrowserScraper
    
    scraper = SO8TControlledBrowserScraper(browser_context, agent)
    results = await scraper.scrape_with_so8t_control(urls)
"""

import sys
import json
import logging
import asyncio
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from collections import deque

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

# SO8T統制エージェントインポート
try:
    from scripts.agents.scraping_reasoning_agent import ScrapingReasoningAgent
    SO8T_AGENT_AVAILABLE = True
except ImportError:
    SO8T_AGENT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("[WARNING] ScrapingReasoningAgent not available")

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/so8t_controlled_browser_scraper.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SO8TControlledBrowserScraper:
    """SO8T統制ブラウザスクレイパー"""
    
    def __init__(
        self,
        browser_context: BrowserContext,
        agent: Optional[ScrapingReasoningAgent] = None,
        num_tabs: int = 10,
        delay_per_action: float = 1.5,
        timeout: int = 30000
    ):
        """
        初期化
        
        Args:
            browser_context: ブラウザコンテキスト
            agent: SO8T統制エージェント（Noneの場合は自動初期化）
            num_tabs: タブ数
            delay_per_action: アクション間の遅延（秒）
            timeout: タイムアウト（ミリ秒）
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError("Playwright not available")
        
        self.browser_context = browser_context
        self.num_tabs = num_tabs
        self.delay_per_action = delay_per_action
        self.timeout = timeout
        
        # SO8T統制エージェント
        if agent is None:
            if SO8T_AGENT_AVAILABLE:
                try:
                    self.agent = ScrapingReasoningAgent()
                    logger.info("[SO8T] ScrapingReasoningAgent initialized")
                except Exception as e:
                    logger.warning(f"[SO8T] Failed to initialize agent: {e}")
                    self.agent = None
            else:
                self.agent = None
                logger.warning("[SO8T] ScrapingReasoningAgent not available, continuing without SO8T control")
        else:
            self.agent = agent
        
        # タブ管理
        self.tabs: List[Page] = []
        self.tab_status: Dict[int, Dict[str, Any]] = {}
        self.visited_urls: Set[str] = set()
        self.scraped_samples: List[Dict] = []
        self.url_queue: deque = deque()  # 代替URLキュー
        
        logger.info("="*80)
        logger.info("SO8T Controlled Browser Scraper Initialized")
        logger.info("="*80)
        logger.info(f"Number of tabs: {self.num_tabs}")
        logger.info(f"SO8T control: {self.agent is not None}")
        logger.info(f"Delay per action: {self.delay_per_action}s")
    
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
                    'allowed_count': 0,
                    'denied_count': 0,
                    'escalated_count': 0,
                    'refused_count': 0
                }
                
                logger.info(f"[TAB {tab_index}] Initialized")
            
            logger.info(f"[OK] Initialized {len(self.tabs)} tabs")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize tabs: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    async def should_scrape_url(self, url: str, keyword: Optional[str] = None) -> Dict[str, Any]:
        """
        SO8TでURLのスクレイピング可否を判断
        
        Args:
            url: スクレイピング対象URL
            keyword: キーワード（オプション）
        
        Returns:
            decision: 判断結果
        """
        if self.agent is None:
            # SO8T統制が無効な場合は常に許可
            return {
                'decision': 'ALLOW',
                'confidence': 1.0,
                'reasoning': 'SO8T control disabled',
                'method': 'fallback'
            }
        
        try:
            # SO8T統制エージェントで判断
            result = self.agent.should_scrape(url=url, keyword=keyword)
            
            decision = result.get('decision', 'ALLOW')
            four_class_label = result.get('four_class_label', 'ALLOW')
            
            logger.info(f"[SO8T] URL: {url}, Decision: {decision}, Four-class: {four_class_label}")
            
            return {
                'decision': decision,
                'four_class_label': four_class_label,
                'confidence': result.get('confidence', 0.5),
                'reasoning': result.get('reasoning', ''),
                'method': 'so8t',
                'quadruple_reasoning': result.get('quadruple_reasoning', {})
            }
            
        except Exception as e:
            logger.error(f"[SO8T] Failed to get decision: {e}")
            # エラー時は許可（安全側に倒す）
            return {
                'decision': 'ALLOW',
                'confidence': 0.5,
                'reasoning': f'Error in SO8T decision: {str(e)}',
                'method': 'error_fallback'
            }
    
    async def scrape_with_so8t_control(
        self,
        urls: List[str],
        keywords: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        SO8T統制でスクレイピング
        
        Args:
            urls: スクレイピング対象URLのリスト
            keywords: キーワードのリスト（オプション）
        
        Returns:
            results: スクレイピング結果のリスト
        """
        if not self.tabs:
            logger.warning("[TABS] Tabs not initialized, initializing now...")
            await self.initialize_tabs()
        
        if keywords is None:
            keywords = [None] * len(urls)
        
        logger.info(f"[SCRAPE] Starting SO8T-controlled scraping for {len(urls)} URLs...")
        
        # URLをタブ数で分割
        urls_per_tab = len(urls) // self.num_tabs
        url_chunks = [urls[i:i + urls_per_tab] for i in range(0, len(urls), urls_per_tab)]
        keyword_chunks = [keywords[i:i + urls_per_tab] for i in range(0, len(keywords), urls_per_tab)]
        
        # 余りのURLを最初のタブに追加
        if len(url_chunks) > self.num_tabs:
            url_chunks[self.num_tabs - 1].extend(url_chunks[self.num_tabs:])
            keyword_chunks[self.num_tabs - 1].extend(keyword_chunks[self.num_tabs:])
            url_chunks = url_chunks[:self.num_tabs]
            keyword_chunks = keyword_chunks[:self.num_tabs]
        
        # タブ数よりURLチャンクが少ない場合は空リストで埋める
        while len(url_chunks) < self.num_tabs:
            url_chunks.append([])
            keyword_chunks.append([])
        
        logger.info(f"[SCRAPE] Processing {len(urls)} URLs across {self.num_tabs} tabs")
        logger.info(f"[SCRAPE] URLs per tab: {[len(chunk) for chunk in url_chunks]}")
        
        # すべてのタブを並列処理
        tasks = []
        for tab_index, (url_chunk, keyword_chunk) in enumerate(zip(url_chunks, keyword_chunks)):
            if url_chunk:
                task = self._process_tab_with_so8t(tab_index, url_chunk, keyword_chunk)
                tasks.append(task)
        
        # 並列実行
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 結果を統合
        all_results = []
        for results in results_list:
            if isinstance(results, Exception):
                logger.error(f"[SCRAPE] Tab processing exception: {results}")
            elif isinstance(results, list):
                all_results.extend(results)
        
        logger.info(f"[OK] Scraped {len(all_results)} samples with SO8T control")
        
        # 統計を表示
        self._print_statistics()
        
        return all_results
    
    async def _process_tab_with_so8t(
        self,
        tab_index: int,
        urls: List[str],
        keywords: List[Optional[str]]
    ) -> List[Dict]:
        """
        タブでSO8T統制スクレイピング
        
        Args:
            tab_index: タブインデックス
            urls: 処理するURLのリスト
            keywords: キーワードのリスト
        
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
            
            for url, keyword in zip(urls, keywords):
                if url in self.visited_urls:
                    logger.debug(f"[TAB {tab_index}] URL already visited: {url}")
                    continue
                
                try:
                    # SO8Tでスクレイピング可否を判断
                    decision = await self.should_scrape_url(url, keyword)
                    decision_type = decision.get('decision', 'ALLOW')
                    four_class_label = decision.get('four_class_label', 'ALLOW')
                    
                    # 判断結果に基づいて処理
                    if decision_type == 'ALLOW' or four_class_label == 'ALLOW':
                        # スクレイピング実行
                        result = await self._scrape_page(page, tab_index, url, keyword, decision)
                        if result:
                            results.append(result)
                            self.visited_urls.add(url)
                            self.tab_status[tab_index]['processed_count'] += 1
                            self.tab_status[tab_index]['allowed_count'] += 1
                    
                    elif decision_type == 'DENY' or four_class_label == 'DENY':
                        logger.info(f"[TAB {tab_index}] Denied by SO8T: {url}")
                        self.tab_status[tab_index]['denied_count'] += 1
                        # 拒否されたURLはスキップ
                        continue
                    
                    elif decision_type == 'ESCALATION' or four_class_label == 'ESCALATION':
                        logger.info(f"[TAB {tab_index}] Escalated by SO8T: {url}")
                        self.tab_status[tab_index]['escalated_count'] += 1
                        # エスカレーションされたURLは慎重に処理
                        result = await self._scrape_page(page, tab_index, url, keyword, decision)
                        if result:
                            results.append(result)
                            self.visited_urls.add(url)
                            self.tab_status[tab_index]['processed_count'] += 1
                    
                    elif decision_type == 'REFUSE' or four_class_label == 'REFUSE':
                        logger.info(f"[TAB {tab_index}] Refused by SO8T: {url}")
                        self.tab_status[tab_index]['refused_count'] += 1
                        # 拒否されたURLはスキップ
                        continue
                    
                    # アクション間の遅延
                    await asyncio.sleep(self.delay_per_action)
                    
                except Exception as e:
                    logger.error(f"[TAB {tab_index}] Failed to process URL {url}: {e}")
                    continue
            
            self.tab_status[tab_index]['status'] = 'completed'
            
        except Exception as e:
            logger.error(f"[TAB {tab_index}] Tab processing failed: {e}")
            self.tab_status[tab_index]['status'] = 'error'
            import traceback
            logger.error(traceback.format_exc())
        
        return results
    
    async def human_like_mouse_move(self, page: Page, start_x: int, start_y: int, end_x: int, end_y: int):
        """人間のようなマウス移動（滑らかな軌跡）"""
        try:
            steps = random.randint(5, 15)
            for i in range(steps):
                t = i / steps
                # ベジェ曲線のような滑らかな移動
                x = int(start_x + (end_x - start_x) * t + random.randint(-10, 10))
                y = int(start_y + (end_y - start_y) * t + random.randint(-10, 10))
                await page.mouse.move(x, y)
                await asyncio.sleep(random.uniform(0.02, 0.08))
        except Exception as e:
            logger.debug(f"[MOUSE] Mouse movement failed: {e}")
    
    async def enhanced_human_behavior(self, page: Page):
        """より高度な人間を模倣した動作（検知回避）"""
        try:
            viewport = page.viewport_size
            if viewport:
                # 1. ランダムなマウス軌跡（より複雑な動き）
                for _ in range(random.randint(3, 6)):
                    x = random.randint(0, viewport['width'])
                    y = random.randint(0, viewport['height'])
                    await self.human_like_mouse_move(page, 100, 100, x, y)
                    await asyncio.sleep(random.uniform(0.2, 0.5))
            
            # 2. キーボード入力のシミュレート（タブキー、矢印キーなど）
            await page.keyboard.press('Tab')
            await asyncio.sleep(random.uniform(0.3, 0.6))
            await page.keyboard.press('Tab')
            await asyncio.sleep(random.uniform(0.3, 0.6))
            
            # 3. ウィンドウフォーカスのシミュレート
            await page.evaluate("window.focus()")
            await asyncio.sleep(random.uniform(0.5, 1.0))
            
            # 4. スクロールの不規則な動き
            for _ in range(random.randint(2, 4)):
                scroll_amount = random.randint(-300, 300)
                await page.mouse.wheel(0, scroll_amount)
                await asyncio.sleep(random.uniform(0.3, 0.8))
            
            # 5. ページ要素への複数回ホバー
            elements = await page.query_selector_all('a, button, input')
            if elements:
                hover_elements = random.sample(elements, min(random.randint(1, 3), len(elements)))
                for element in hover_elements:
                    try:
                        await element.hover()
                        await asyncio.sleep(random.uniform(0.3, 0.8))
                    except Exception:
                        continue
            
            logger.debug("[BEHAVIOR] Enhanced human behavior completed")
        except Exception as e:
            logger.debug(f"[BEHAVIOR] Enhanced human behavior failed: {e}")
    
    async def detect_bot_checks(self, page: Page) -> Dict[str, bool]:
        """ボット検知チェックを検出"""
        checks = {
            'captcha': False,
            'access_denied': False,
            'rate_limit': False,
            'cloudflare': False,
            'bot_detection': False
        }
        
        try:
            # ページのHTMLを取得
            html = await page.content()
            html_lower = html.lower()
            
            # CAPTCHA検出
            captcha_indicators = ['captcha', 'recaptcha', 'hcaptcha', 'turnstile', 'challenge']
            if any(indicator in html_lower for indicator in captcha_indicators):
                checks['captcha'] = True
                logger.warning("[CHECK] CAPTCHA detected")
            
            # アクセス拒否検出
            access_denied_indicators = ['access denied', 'forbidden', '403', 'blocked', 'unauthorized']
            if any(indicator in html_lower for indicator in access_denied_indicators):
                checks['access_denied'] = True
                logger.warning("[CHECK] Access denied detected")
            
            # Cloudflare検出
            cloudflare_indicators = ['cloudflare', 'checking your browser', 'ddos protection', 'ray id']
            if any(indicator in html_lower for indicator in cloudflare_indicators):
                checks['cloudflare'] = True
                logger.warning("[CHECK] Cloudflare protection detected")
            
            # ボット検知検出
            bot_indicators = ['bot detection', 'automated', 'verify you are human', 'suspicious activity']
            if any(indicator in html_lower for indicator in bot_indicators):
                checks['bot_detection'] = True
                logger.warning("[CHECK] Bot detection detected")
            
            # レート制限検出
            rate_limit_indicators = ['rate limit', 'too many requests', '429', 'slow down', 'try again later']
            if any(indicator in html_lower for indicator in rate_limit_indicators):
                checks['rate_limit'] = True
                logger.warning("[CHECK] Rate limit detected")
            
        except Exception as e:
            logger.debug(f"[CHECK] Detection failed: {e}")
        
        return checks
    
    async def bypass_bot_checks(self, page: Page, checks: Dict[str, bool]) -> bool:
        """ボット検知チェックを回避"""
        try:
            # CAPTCHA検出時の待機処理
            if checks.get('captcha'):
                logger.info("[BYPASS] Waiting for CAPTCHA...")
                await asyncio.sleep(random.uniform(5.0, 10.0))
                # 再度チェック
                checks = await self.detect_bot_checks(page)
                if checks.get('captcha'):
                    return False
            
            # Cloudflare検出時の待機処理
            if checks.get('cloudflare'):
                logger.info("[BYPASS] Waiting for Cloudflare check...")
                await asyncio.sleep(random.uniform(3.0, 6.0))
                # 再度チェック
                checks = await self.detect_bot_checks(page)
                if checks.get('cloudflare'):
                    return False
            
            # レート制限検出時の指数バックオフ
            if checks.get('rate_limit'):
                backoff_time = random.uniform(10.0, 30.0)
                logger.info(f"[BYPASS] Rate limit detected, waiting {backoff_time:.1f}s...")
                await asyncio.sleep(backoff_time)
                return True
            
            # アクセス拒否検出時のユーザーエージェント変更
            if checks.get('access_denied'):
                logger.info("[BYPASS] Access denied detected, changing user agent...")
                user_agents = [
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                ]
                await page.set_extra_http_headers({
                    'User-Agent': random.choice(user_agents)
                })
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"[BYPASS] Failed to bypass checks: {e}")
            return False
    
    async def handle_check_failure(self, page: Page, checks: Dict[str, bool], url_queue: deque) -> bool:
        """チェック失敗時の処理（ブラウザバックと別ページへの遷移）"""
        if any(checks.values()):
            logger.info("[RECOVERY] Check failure detected, attempting recovery...")
            
            try:
                # ブラウザバック
                if page.url != "about:blank":
                    await page.go_back()
                    await asyncio.sleep(random.uniform(1.0, 2.0))
                    logger.info("[RECOVERY] Browser back executed")
                
                # 別のページへ遷移
                if url_queue:
                    next_url = url_queue.popleft()
                    logger.info(f"[RECOVERY] Navigating to alternative URL: {next_url}")
                    await page.goto(next_url, wait_until="networkidle", timeout=self.timeout)
                    await asyncio.sleep(random.uniform(1.0, 2.0))
                    return True
                else:
                    logger.warning("[RECOVERY] No alternative URLs available")
                    return False
                    
            except Exception as e:
                logger.error(f"[RECOVERY] Recovery failed: {e}")
                return False
        
        return False
    
    async def _scrape_page(
        self,
        page: Page,
        tab_index: int,
        url: str,
        keyword: Optional[str],
        decision: Dict[str, Any]
    ) -> Optional[Dict]:
        """
        ページをスクレイピング
        
        Args:
            page: ページインスタンス
            tab_index: タブインデックス
            url: スクレイピング対象URL
            keyword: キーワード
            decision: SO8T判断結果
        
        Returns:
            result: スクレイピング結果
        """
        try:
            logger.info(f"[TAB {tab_index}] Scraping URL: {url}")
            
            # ページに移動
            await page.goto(url, timeout=self.timeout, wait_until='networkidle')
            await asyncio.sleep(random.uniform(1.0, 2.0))
            
            # ボット検知チェック
            checks = await self.detect_bot_checks(page)
            if any(checks.values()):
                logger.warning(f"[TAB {tab_index}] Bot checks detected")
                # 回避を試みる
                bypass_success = await self.bypass_bot_checks(page, checks)
                if not bypass_success:
                    # 回避失敗時はリカバリー処理
                    recovery_success = await self.handle_check_failure(page, checks, self.url_queue)
                    if not recovery_success:
                        return None
                    # リカバリー後、再度チェック
                    checks = await self.detect_bot_checks(page)
                    if any(checks.values()):
                        return None
            
            # 人間を模倣した動作
            await self.enhanced_human_behavior(page)
            
            # 現在のURLを更新
            current_url = page.url
            self.tab_status[tab_index]['current_url'] = current_url
            
            # ページコンテンツを取得
            html = await page.content()
            title = await page.title()
            
            # テキストを抽出
            text = await page.evaluate("() => document.body.innerText")
            
            # リンクを抽出
            links = await page.evaluate("""
                () => {
                    const links = Array.from(document.querySelectorAll('a[href]'));
                    return links.map(link => link.href).filter(href => href && href.startsWith('http'));
                }
            """)
            
            # リンクをURLキューに追加（代替URLとして使用）
            for link in links[:20]:  # 最初の20個のリンクを追加
                if link not in self.visited_urls and link not in self.url_queue:
                    self.url_queue.append(link)
            
            result = {
                'url': url,
                'title': title,
                'text': text[:10000],  # テキストを10000文字に制限
                'links': links[:100],  # リンクを100個に制限
                'keyword': keyword,
                'so8t_decision': decision,
                'scraped_at': datetime.now().isoformat(),
                'tab_index': tab_index,
                'bot_checks': checks,
                'metadata': {
                    'url': url,
                    'timestamp': datetime.now().isoformat(),
                    'tab_index': tab_index
                }
            }
            
            self.scraped_samples.append(result)
            return result
            
        except PlaywrightTimeoutError:
            logger.warning(f"[TAB {tab_index}] Timeout while scraping URL: {url}")
            # タイムアウト時もリカバリーを試みる
            if self.url_queue:
                next_url = self.url_queue.popleft()
                logger.info(f"[RECOVERY] Timeout recovery: navigating to {next_url}")
                try:
                    await page.goto(next_url, wait_until="networkidle", timeout=self.timeout)
                    await asyncio.sleep(random.uniform(1.0, 2.0))
                except Exception:
                    pass
            return None
        except Exception as e:
            logger.error(f"[TAB {tab_index}] Error scraping URL {url}: {e}")
            return None
    
    def _print_statistics(self):
        """統計を表示"""
        logger.info("="*80)
        logger.info("SO8T Control Statistics")
        logger.info("="*80)
        
        total_allowed = sum(status.get('allowed_count', 0) for status in self.tab_status.values())
        total_denied = sum(status.get('denied_count', 0) for status in self.tab_status.values())
        total_escalated = sum(status.get('escalated_count', 0) for status in self.tab_status.values())
        total_refused = sum(status.get('refused_count', 0) for status in self.tab_status.values())
        total_processed = sum(status.get('processed_count', 0) for status in self.tab_status.values())
        
        logger.info(f"Total processed: {total_processed}")
        logger.info(f"  ALLOW: {total_allowed}")
        logger.info(f"  DENY: {total_denied}")
        logger.info(f"  ESCALATION: {total_escalated}")
        logger.info(f"  REFUSE: {total_refused}")
        logger.info(f"Total samples scraped: {len(self.scraped_samples)}")
        logger.info("="*80)
    
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
    
    def get_scraped_samples(self) -> List[Dict]:
        """
        スクレイピングされたサンプルを取得
        
        Returns:
            samples: スクレイピングされたサンプルのリスト
        """
        return self.scraped_samples.copy()
    
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
        
        # SO8T統制スクレイパーを初期化
        scraper = SO8TControlledBrowserScraper(browser_context, num_tabs=10)
        await scraper.initialize_tabs()
        
        # テストURL
        test_urls = [
            "https://example.com",
            "https://www.google.com",
            "https://github.com"
        ] * 5  # 15個のURL
        
        test_keywords = ["example", "search", "code"] * 5
        
        # SO8T統制でスクレイピング
        results = await scraper.scrape_with_so8t_control(test_urls, test_keywords)
        
        logger.info(f"[OK] Scraped {len(results)} samples")
        
        # タブを閉じる
        await scraper.close_tabs()
        
        # ブラウザを停止
        await manager.stop_all_browsers()
        
    except Exception as e:
        logger.error(f"[ERROR] Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        await manager.stop_all_browsers()


if __name__ == "__main__":
    asyncio.run(main())

