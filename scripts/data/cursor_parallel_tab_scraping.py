#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Cursorブラウザ並列タブスクレイピングスクリプト

Cursorブラウザを使って10個のタブを並列処理し、各タブで10ページずつ計100ページを
人間の動きを忠実に再現してスクレイピングします。

Usage:
    python scripts/data/cursor_parallel_tab_scraping.py --output D:\webdataset\processed --num-tabs 10 --pages-per-tab 10
"""

import sys
import json
import logging
import asyncio
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse, urljoin
from collections import deque

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Playwrightインポート
try:
    from playwright.async_api import async_playwright, Browser, Page, TimeoutError as PlaywrightTimeoutError
except ImportError:
    print("[ERROR] Playwright not installed. Install with: pip install playwright")
    sys.exit(1)

# BeautifulSoupインポート
try:
    from bs4 import BeautifulSoup
except ImportError:
    print("[ERROR] BeautifulSoup not installed. Install with: pip install beautifulsoup4")
    sys.exit(1)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/cursor_parallel_tab_scraping.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CursorParallelTabScraper:
    """Cursorブラウザ並列タブスクレイピングクラス（MCP Chrome DevTools対応）"""
    
    def __init__(
        self,
        output_dir: Path,
        num_tabs: int = 10,
        pages_per_tab: int = 10,
        use_cursor_browser: bool = True,
        use_mcp_chrome_devtools: bool = True,  # MCP Chrome DevToolsを使用するか
        remote_debugging_port: int = 9222,
        delay_per_action: float = 1.5,
        timeout: int = 30000
    ):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリ
            num_tabs: タブ数
            pages_per_tab: タブあたりのページ数
            use_cursor_browser: Cursorのブラウザを使用するか
            remote_debugging_port: リモートデバッグポート
            delay_per_action: アクション間の遅延（秒）
            timeout: タイムアウト（ミリ秒）
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_tabs = num_tabs
        self.pages_per_tab = pages_per_tab
        self.use_cursor_browser = use_cursor_browser
        self.use_mcp_chrome_devtools = use_mcp_chrome_devtools
        self.remote_debugging_port = remote_debugging_port
        self.delay_per_action = delay_per_action
        self.timeout = timeout
        
        self.all_samples: List[Dict] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.visited_urls: Set[str] = set()
        
        # MCP Chrome DevToolsラッパーをインポート（オプション）
        self.mcp_wrapper = None
        if self.use_mcp_chrome_devtools:
            try:
                from scripts.utils.mcp_chrome_devtools_wrapper import MCPChromeDevTools
                # MCPサーバー設定を読み込み（設定ファイルから取得）
                import yaml
                config_path = PROJECT_ROOT / "configs" / "unified_master_pipeline_config.yaml"
                mcp_config = {}
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                        mcp_config = config.get('mcp_server', {})
                
                # MCPサーバーが有効な場合のみ初期化
                if mcp_config.get('enabled', True):
                    self.mcp_wrapper = MCPChromeDevTools(
                        transport=mcp_config.get('transport', 'stdio'),
                        command=mcp_config.get('command', 'npx'),
                        args=mcp_config.get('args', ['-y', '@modelcontextprotocol/server-chrome-devtools']),
                        url=mcp_config.get('url'),
                        timeout=mcp_config.get('timeout', 30000)
                    )
                    logger.info("[MCP] Chrome DevTools wrapper initialized with MCP client")
                else:
                    logger.info("[MCP] MCP server disabled in config, using fallback")
                    self.use_mcp_chrome_devtools = False
            except ImportError as e:
                logger.warning(f"[MCP] Failed to import MCP Chrome DevTools wrapper: {e}")
                logger.warning("[MCP] Falling back to Playwright")
                self.use_mcp_chrome_devtools = False
            except Exception as e:
                logger.warning(f"[MCP] Failed to initialize MCP Chrome DevTools wrapper: {e}")
                logger.warning("[MCP] Falling back to Playwright")
                self.use_mcp_chrome_devtools = False
        
        logger.info("="*80)
        logger.info("Cursor Parallel Tab Scraper Initialized")
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Number of tabs: {self.num_tabs}")
        logger.info(f"Pages per tab: {self.pages_per_tab}")
        logger.info(f"Total pages: {self.num_tabs * self.pages_per_tab}")
        logger.info(f"Use Cursor browser: {self.use_cursor_browser}")
        logger.info(f"Use MCP Chrome DevTools: {self.use_mcp_chrome_devtools}")
        logger.info(f"Remote debugging port: {self.remote_debugging_port}")
    
    async def connect_to_cursor_browser(self, playwright, tab_index: int) -> Optional[Browser]:
        """Cursorブラウザに接続"""
        if not self.use_cursor_browser:
            logger.info(f"[TAB {tab_index}] Launching new browser...")
            browser = await playwright.chromium.launch(headless=False)
            logger.info(f"[TAB {tab_index}] Browser launched")
            return browser
        
        try:
            port = self.remote_debugging_port + tab_index
            logger.info(f"[TAB {tab_index}] Connecting to Cursor browser on port {port}...")
            cdp_endpoint = f"http://127.0.0.1:{port}"
            browser = await playwright.chromium.connect_over_cdp(cdp_endpoint)
            
            contexts = browser.contexts
            if contexts:
                logger.info(f"[TAB {tab_index}] Connected to Cursor browser (found {len(contexts)} contexts)")
            else:
                logger.info(f"[TAB {tab_index}] No existing contexts found, creating new context...")
                await browser.new_context()
                logger.info(f"[TAB {tab_index}] New context created")
            
            return browser
            
        except Exception as e:
            logger.warning(f"[TAB {tab_index}] Failed to connect to Cursor browser: {e}")
            logger.info(f"[TAB {tab_index}] Falling back to launching new browser...")
            browser = await playwright.chromium.launch(headless=False)
            logger.info(f"[TAB {tab_index}] New browser launched")
            return browser
    
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
    
    async def extract_links(self, page: Page, base_url: str, max_links: int = 20) -> List[str]:
        """ページからリンクを抽出"""
        try:
            links = await page.evaluate("""
                () => {
                    const links = Array.from(document.querySelectorAll('a[href]'));
                    return links.map(link => link.href).filter(href => href && href.startsWith('http'));
                }
            """)
            
            # URLを正規化
            normalized_links = []
            base_domain = urlparse(base_url).netloc
            
            for link in links[:max_links]:
                try:
                    parsed = urlparse(link)
                    # 同じドメイン内のリンクのみ
                    if parsed.netloc == base_domain or not parsed.netloc:
                        if not parsed.netloc:
                            link = urljoin(base_url, link)
                        if link not in self.visited_urls:
                            normalized_links.append(link)
                except Exception:
                    continue
            
            return normalized_links
        except Exception as e:
            logger.debug(f"[LINKS] Failed to extract links: {e}")
            return []
    
    async def extract_code_blocks(self, soup: BeautifulSoup) -> List[str]:
        """コードブロックを抽出"""
        code_blocks = []
        
        # <pre><code>タグからコードを抽出
        pre_code_blocks = soup.find_all(['pre', 'code'])
        for block in pre_code_blocks:
            code_text = block.get_text(strip=True)
            if code_text and len(code_text) > 10:  # 最小長をチェック
                code_blocks.append(code_text)
        
        # GitHubのコードブロック
        github_code_blocks = soup.find_all('div', class_=lambda x: x and 'highlight' in str(x).lower())
        for block in github_code_blocks:
            code_text = block.get_text(strip=True)
            if code_text and len(code_text) > 10:
                code_blocks.append(code_text)
        
        return code_blocks
    
    def is_coding_related(self, url: str, title: str, text: str) -> bool:
        """コーディング関連コンテンツかどうかを判定"""
        coding_keywords = [
            'code', 'programming', 'developer', 'software', 'algorithm',
            'function', 'class', 'variable', 'syntax', 'api', 'framework',
            'library', 'package', 'module', 'import', 'export', 'git',
            'github', 'stackoverflow', 'qiita', 'zenn', 'dev.to',
            'python', 'javascript', 'typescript', 'java', 'cpp', 'rust',
            'go', 'ruby', 'php', 'sql', 'html', 'css', 'react', 'vue',
            'angular', 'node', 'npm', 'yarn', 'docker', 'kubernetes',
            'aws', 'azure', 'gcp', 'linux', 'unix', 'bash', 'shell'
        ]
        
        url_lower = url.lower()
        title_lower = title.lower()
        text_lower = text.lower()[:500]  # 最初の500文字をチェック
        
        # URLまたはタイトルにコーディング関連キーワードが含まれるか
        for keyword in coding_keywords:
            if keyword in url_lower or keyword in title_lower or keyword in text_lower:
                return True
        
        return False
    
    async def scrape_page(self, page: Page, url: str, tab_index: int, url_queue: deque) -> Optional[Dict]:
        """1ページをスクレイピング"""
        try:
            logger.info(f"[TAB {tab_index}] Scraping: {url}")
            
            # ページに移動
            await page.goto(url, wait_until="networkidle", timeout=self.timeout)
            await asyncio.sleep(random.uniform(1.0, 2.0))
            
            # ボット検知チェック
            checks = await self.detect_bot_checks(page)
            if any(checks.values()):
                logger.warning(f"[TAB {tab_index}] Bot checks detected")
                # リカバリー処理
                recovery_success = await self.handle_check_failure(page, checks, url_queue)
                if not recovery_success:
                    return None
                # リカバリー後、再度チェック
                checks = await self.detect_bot_checks(page)
                if any(checks.values()):
                    return None
            
            # 人間を模倣した動作
            await self.enhanced_human_behavior(page)
            
            # リンクを抽出してURLキューに追加（コーディング関連サイトを優先）
            links = await self.extract_links(page, url, max_links=30)
            coding_links = []
            other_links = []
            
            for link in links:
                if link not in self.visited_urls and link not in url_queue:
                    # コーディング関連リンクを優先
                    if any(keyword in link.lower() for keyword in ['github', 'stackoverflow', 'qiita', 'zenn', 'dev.to', 'code', 'programming', 'developer']):
                        coding_links.append(link)
                    else:
                        other_links.append(link)
            
            # コーディング関連リンクを先に追加
            for link in coding_links:
                url_queue.append(link)
            for link in other_links:
                url_queue.append(link)
            
            # ページコンテンツを取得
            html = await page.content()
            soup = BeautifulSoup(html, 'html.parser')
            
            # メタデータを取得
            title = soup.find('title')
            title_text = title.get_text(strip=True) if title else ""
            
            # コードブロックを抽出
            code_blocks = await self.extract_code_blocks(soup)
            
            # テキストを抽出
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text(separator='\n', strip=True)
            
            # コーディング関連かどうかを判定
            is_coding = self.is_coding_related(url, title_text, text)
            
            # サンプルを作成
            sample = {
                'url': url,
                'title': title_text,
                'text': text,
                'code_blocks': code_blocks,
                'is_coding_related': is_coding,
                'code_block_count': len(code_blocks),
                'timestamp': datetime.now().isoformat(),
                'tab_index': tab_index,
                'session_id': self.session_id
            }
            
            logger.info(f"[TAB {tab_index}] Successfully scraped: {url} (coding: {is_coding}, code blocks: {len(code_blocks)})")
            return sample
            
        except PlaywrightTimeoutError:
            logger.warning(f"[TAB {tab_index}] Timeout while scraping: {url}")
            return None
        except Exception as e:
            logger.error(f"[TAB {tab_index}] Failed to scrape {url}: {e}")
            return None
    
    async def scrape_tab(
        self,
        tab_index: int,
        start_url: str,
        url_queue: deque,
        pages_per_tab: int
    ) -> List[Dict]:
        """1つのタブでスクレイピング"""
        samples = []
        current_url = start_url
        
        async with async_playwright() as playwright:
            browser = await self.connect_to_cursor_browser(playwright, tab_index)
            
            if not browser:
                logger.error(f"[TAB {tab_index}] Failed to connect to browser")
                return samples
            
            try:
                # コンテキストとページを作成
                context = await browser.new_context()
                page = await context.new_page()
                
                # User-Agentを設定
                await page.set_extra_http_headers({
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                })
                
                logger.info(f"[TAB {tab_index}] Starting scraping (target: {pages_per_tab} pages)...")
                
                pages_scraped = 0
                consecutive_failures = 0
                max_consecutive_failures = 3
                
                while pages_scraped < pages_per_tab:
                    if current_url in self.visited_urls:
                        # 既に訪問済みの場合は次のURLを取得
                        if url_queue:
                            current_url = url_queue.popleft()
                            continue
                        else:
                            logger.warning(f"[TAB {tab_index}] No more URLs available")
                            break
                    
                    # ページをスクレイピング
                    sample = await self.scrape_page(page, current_url, tab_index, url_queue)
                    
                    if sample:
                        samples.append(sample)
                        self.visited_urls.add(current_url)
                        pages_scraped += 1
                        consecutive_failures = 0
                        
                        # 次のURLを取得
                        if url_queue:
                            current_url = url_queue.popleft()
                        else:
                            logger.warning(f"[TAB {tab_index}] No more URLs in queue")
                            break
                    else:
                        consecutive_failures += 1
                        
                        # 連続失敗時のリカバリー処理
                        if consecutive_failures >= max_consecutive_failures:
                            logger.warning(f"[TAB {tab_index}] Too many consecutive failures, attempting recovery...")
                            
                            # ボット検知チェック
                            checks = await self.detect_bot_checks(page)
                            recovery_success = await self.handle_check_failure(page, checks, url_queue)
                            
                            if recovery_success:
                                consecutive_failures = 0
                                if url_queue:
                                    current_url = url_queue.popleft()
                                else:
                                    break
                            else:
                                logger.error(f"[TAB {tab_index}] Recovery failed, stopping tab")
                                break
                        else:
                            # 次のURLを取得
                            if url_queue:
                                current_url = url_queue.popleft()
                            else:
                                break
                    
                    # アクション間の待機
                    await asyncio.sleep(random.uniform(self.delay_per_action, self.delay_per_action * 2))
                
                logger.info(f"[TAB {tab_index}] Completed scraping: {pages_scraped} pages")
                
            except Exception as e:
                logger.error(f"[TAB {tab_index}] Tab scraping failed: {e}")
            finally:
                await context.close()
                if not self.use_cursor_browser:
                    await browser.close()
        
        return samples
    
    async def scrape_with_parallel_tabs(
        self,
        start_urls: List[str],
        pages_per_tab: int = 10,
        num_tabs: int = 10
    ) -> List[Dict]:
        """並列タブスクレイピング"""
        logger.info("="*80)
        logger.info("Starting Parallel Tab Scraping")
        logger.info("="*80)
        logger.info(f"Number of tabs: {num_tabs}")
        logger.info(f"Pages per tab: {pages_per_tab}")
        logger.info(f"Total pages: {num_tabs * pages_per_tab}")
        logger.info(f"Start URLs: {len(start_urls)}")
        
        # URLキューを作成（各タブで使用）
        url_queues = []
        for i in range(num_tabs):
            queue = deque()
            # 開始URLを追加
            start_url = start_urls[i % len(start_urls)]
            queue.append(start_url)
            url_queues.append(queue)
        
        # 並列タスクを作成
        tasks = []
        for tab_index in range(num_tabs):
            start_url = start_urls[tab_index % len(start_urls)]
            task = self.scrape_tab(
                tab_index=tab_index,
                start_url=start_url,
                url_queue=url_queues[tab_index],
                pages_per_tab=pages_per_tab
            )
            tasks.append(task)
        
        # 並列実行
        logger.info("[PARALLEL] Starting parallel tab scraping...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 結果を統合
        all_samples = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"[TAB {i}] Exception occurred: {result}")
            else:
                all_samples.extend(result)
                logger.info(f"[TAB {i}] Collected {len(result)} samples")
        
        logger.info(f"[PARALLEL] Total samples collected: {len(all_samples)}")
        return all_samples
    
    def save_samples(self, samples: List[Dict]):
        """サンプルを保存"""
        if not samples:
            logger.warning("[SAVE] No samples to save")
            return
        
        # 全サンプルを保存
        output_file = self.output_dir / f"parallel_tab_scraped_{self.session_id}.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"[SAVE] Saved {len(samples)} samples to {output_file}")
        
        # コーディング関連サンプルを別ファイルに保存
        coding_samples = [s for s in samples if s.get('is_coding_related', False)]
        if coding_samples:
            coding_output_file = self.output_dir / f"parallel_tab_coding_{self.session_id}.jsonl"
            with open(coding_output_file, 'w', encoding='utf-8') as f:
                for sample in coding_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            logger.info(f"[SAVE] Saved {len(coding_samples)} coding-related samples to {coding_output_file}")
            
            # 統計情報をログ出力
            total_code_blocks = sum(s.get('code_block_count', 0) for s in coding_samples)
            logger.info(f"[STATS] Coding samples: {len(coding_samples)}/{len(samples)} ({len(coding_samples)/len(samples)*100:.1f}%)")
            logger.info(f"[STATS] Total code blocks: {total_code_blocks}")


async def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Cursor Parallel Tab Scraping')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--num-tabs', type=int, default=10, help='Number of tabs')
    parser.add_argument('--pages-per-tab', type=int, default=10, help='Pages per tab')
    parser.add_argument('--use-cursor-browser', action='store_true', default=True, help='Use Cursor browser')
    parser.add_argument('--remote-debugging-port', type=int, default=9222, help='Remote debugging port')
    parser.add_argument('--delay-per-action', type=float, default=1.5, help='Delay per action (seconds)')
    parser.add_argument('--timeout', type=int, default=30000, help='Timeout (milliseconds)')
    parser.add_argument('--start-urls', type=str, nargs='+', help='Start URLs')
    
    args = parser.parse_args()
    
    # デフォルトの開始URL（コーディング関連サイトを優先）
    if not args.start_urls:
        args.start_urls = [
            # コーディング関連サイト（優先）
            "https://github.com/trending",
            "https://github.com/explore",
            "https://stackoverflow.com/questions/tagged/python",
            "https://stackoverflow.com/questions/tagged/javascript",
            "https://stackoverflow.com/questions/tagged/typescript",
            "https://qiita.com/",
            "https://zenn.dev/",
            "https://dev.to/",
            "https://medium.com/tag/programming",
            "https://www.reddit.com/r/programming/",
            "https://news.ycombinator.com/",
            # 技術ドキュメントサイト
            "https://docs.python.org/",
            "https://developer.mozilla.org/",
            "https://react.dev/",
            "https://vuejs.org/",
            "https://angular.io/",
            # コーディング学習サイト
            "https://www.freecodecamp.org/",
            "https://www.codecademy.com/",
            "https://leetcode.com/",
            "https://www.codewars.com/",
            # 技術ブログ
            "https://techcrunch.com/",
            "https://www.infoq.com/",
            "https://www.oreilly.com/",
            # その他
            "https://ja.wikipedia.org/wiki/メインページ"
        ]
    
    scraper = CursorParallelTabScraper(
        output_dir=Path(args.output),
        num_tabs=args.num_tabs,
        pages_per_tab=args.pages_per_tab,
        use_cursor_browser=args.use_cursor_browser,
        use_mcp_chrome_devtools=args.use_mcp_chrome_devtools,
        remote_debugging_port=args.remote_debugging_port,
        delay_per_action=args.delay_per_action,
        timeout=args.timeout
    )
    
    # スクレイピング実行
    samples = await scraper.scrape_with_parallel_tabs(
        start_urls=args.start_urls,
        pages_per_tab=args.pages_per_tab,
        num_tabs=args.num_tabs
    )
    
    # サンプルを保存
    scraper.save_samples(samples)
    
    logger.info("="*80)
    logger.info("Parallel Tab Scraping Completed")
    logger.info("="*80)
    logger.info(f"Total samples: {len(samples)}")


if __name__ == '__main__':
    asyncio.run(main())

