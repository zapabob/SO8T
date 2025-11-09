#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人間を模倣したWebスクレイピングスクリプト

Cursorブラウザを使って、人間のような自然な動きでWebスクレイピングを実行します。
- ランダムなマウス移動
- 自然なスクロール動作
- ランダムな待機時間
- ページ要素へのホバー
- 自動ページ遷移

Usage:
    python scripts/data/human_like_web_scraping.py --urls https://ja.wikipedia.org/wiki/メインページ --output D:\webdataset\processed
"""

import sys
import json
import logging
import asyncio
import argparse
import random
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse, urljoin

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
        logging.FileHandler('logs/human_like_scraping.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HumanLikeScraper:
    """人間を模倣したWebスクレイピングクラス"""
    
    def __init__(
        self,
        output_dir: Path,
        use_cursor_browser: bool = True,
        remote_debugging_port: int = 9222,
        delay_per_request: float = 2.0,
        timeout: int = 30000,
        max_pages: int = 100,
        follow_links: bool = True,
        max_depth: int = 3
    ):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリ
            use_cursor_browser: Cursorのブラウザを使用するか
            remote_debugging_port: リモートデバッグポート
            delay_per_request: リクエスト間の遅延（秒）
            timeout: タイムアウト（ミリ秒）
            max_pages: 最大ページ数
            follow_links: リンクを追跡するか
            max_depth: 最大深度
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_cursor_browser = use_cursor_browser
        self.remote_debugging_port = remote_debugging_port
        self.delay_per_request = delay_per_request
        self.timeout = timeout
        self.max_pages = max_pages
        self.follow_links = follow_links
        self.max_depth = max_depth
        
        self.collected_samples: List[Dict] = []
        self.visited_urls: Set[str] = set()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info("="*80)
        logger.info("Human-Like Web Scraper Initialized")
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Use Cursor browser: {self.use_cursor_browser}")
        logger.info(f"Max pages: {self.max_pages}")
        logger.info(f"Follow links: {self.follow_links}")
        logger.info(f"Max depth: {self.max_depth}")
    
    async def connect_to_cursor_browser(self, playwright) -> Optional[Browser]:
        """Cursorのブラウザに接続"""
        if not self.use_cursor_browser:
            logger.info("[BROWSER] Launching new browser...")
            browser = await playwright.chromium.launch(headless=False)
            logger.info("[OK] Browser launched")
            return browser
        
        try:
            logger.info(f"[BROWSER] Connecting to Cursor browser on port {self.remote_debugging_port}...")
            cdp_endpoint = f"http://127.0.0.1:{self.remote_debugging_port}"
            browser = await playwright.chromium.connect_over_cdp(cdp_endpoint)
            
            contexts = browser.contexts
            if contexts:
                logger.info(f"[OK] Connected to Cursor browser (found {len(contexts)} contexts)")
            else:
                logger.info("[INFO] No existing contexts found, creating new context...")
                await browser.new_context()
                logger.info("[OK] New context created")
            
            return browser
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to connect to Cursor browser: {e}")
            logger.info("[INFO] Falling back to launching new browser...")
            browser = await playwright.chromium.launch(headless=False)
            logger.info("[OK] New browser launched")
            return browser
    
    async def human_like_mouse_move(self, page: Page):
        """人間のようなマウス移動"""
        try:
            # ランダムな位置にマウスを移動
            viewport_size = page.viewport_size
            if viewport_size:
                x = random.randint(100, viewport_size['width'] - 100)
                y = random.randint(100, viewport_size['height'] - 100)
                
                # 滑らかな移動（複数ステップ）
                steps = random.randint(3, 8)
                for i in range(steps):
                    await page.mouse.move(
                        x + random.randint(-50, 50),
                        y + random.randint(-50, 50),
                        steps=random.randint(5, 15)
                    )
                    await asyncio.sleep(random.uniform(0.05, 0.15))
                
                logger.debug("[MOUSE] Human-like mouse movement completed")
        except Exception as e:
            logger.debug(f"[MOUSE] Mouse movement failed: {e}")
    
    async def human_like_scroll(self, page: Page):
        """人間のようなスクロール動作"""
        try:
            # ページの高さを取得
            page_height = await page.evaluate("document.body.scrollHeight")
            viewport_height = page.viewport_size['height'] if page.viewport_size else 800
            
            # ランダムなスクロール位置
            scroll_positions = []
            current_pos = 0
            
            while current_pos < page_height - viewport_height:
                # ランダムなスクロール量（人間は一度に全部スクロールしない）
                scroll_amount = random.randint(200, 600)
                current_pos += scroll_amount
                
                if current_pos > page_height - viewport_height:
                    current_pos = page_height - viewport_height
                
                scroll_positions.append(current_pos)
            
            # スクロール実行（ランダムな速度で）
            for pos in scroll_positions:
                await page.evaluate(f"window.scrollTo({{top: {pos}, behavior: 'smooth'}})")
                await asyncio.sleep(random.uniform(0.5, 1.5))  # スクロール後の待機
            
            # 最後にトップに戻る（人間はよくやる）
            if random.random() > 0.5:
                await page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
                await asyncio.sleep(random.uniform(0.3, 0.8))
            
            logger.debug("[SCROLL] Human-like scrolling completed")
        except Exception as e:
            logger.debug(f"[SCROLL] Scrolling failed: {e}")
    
    async def human_like_wait(self, min_seconds: float = 1.0, max_seconds: float = 3.0):
        """人間のような待機（ランダムな時間）"""
        wait_time = random.uniform(min_seconds, max_seconds)
        await asyncio.sleep(wait_time)
    
    async def human_like_hover(self, page: Page):
        """ページ要素へのホバー（人間のような動作）"""
        try:
            # リンクやボタンなどのインタラクティブ要素を探す
            elements = await page.query_selector_all('a, button, [role="button"]')
            
            if elements:
                # ランダムに1-3個の要素にホバー
                num_hovers = random.randint(1, min(3, len(elements)))
                selected_elements = random.sample(elements, num_hovers)
                
                for element in selected_elements:
                    try:
                        await element.hover(timeout=2000)
                        await asyncio.sleep(random.uniform(0.3, 0.8))
                        logger.debug("[HOVER] Hovered over element")
                    except Exception:
                        pass  # ホバー失敗は無視
            
        except Exception as e:
            logger.debug(f"[HOVER] Hover failed: {e}")
    
    async def extract_links(self, page: Page, base_url: str) -> List[str]:
        """ページからリンクを抽出"""
        try:
            links = await page.evaluate("""
                () => {
                    const links = [];
                    document.querySelectorAll('a[href]').forEach(a => {
                        const href = a.getAttribute('href');
                        if (href && !href.startsWith('javascript:') && !href.startsWith('#')) {
                            links.push(href);
                        }
                    });
                    return links;
                }
            """)
            
            # 相対URLを絶対URLに変換
            absolute_links = []
            for link in links:
                try:
                    absolute_url = urljoin(base_url, link)
                    parsed = urlparse(absolute_url)
                    # 同じドメインのリンクのみ
                    base_parsed = urlparse(base_url)
                    if parsed.netloc == base_parsed.netloc or not parsed.netloc:
                        absolute_links.append(absolute_url)
                except Exception:
                    pass
            
            return absolute_links[:20]  # 最大20リンク
            
        except Exception as e:
            logger.debug(f"[LINKS] Link extraction failed: {e}")
            return []
    
    async def scrape_page(
        self,
        page: Page,
        url: str,
        depth: int = 0
    ) -> Optional[Dict]:
        """
        単一ページをスクレイピング（人間のような動作）
        
        Args:
            page: Playwright Pageオブジェクト
            url: スクレイピングするURL
            depth: 現在の深度
        
        Returns:
            サンプル辞書（成功時）またはNone
        """
        if url in self.visited_urls:
            return None
        
        if depth > self.max_depth:
            logger.debug(f"[SKIP] Max depth reached: {depth}")
            return None
        
        try:
            logger.info(f"[SCRAPE] [{depth}] Navigating to: {url}")
            
            # ページに移動
            await page.goto(url, timeout=self.timeout, wait_until="networkidle")
            
            # 人間のような待機（ページ読み込み後）
            await self.human_like_wait(1.0, 2.5)
            
            # 人間のようなマウス移動
            await self.human_like_mouse_move(page)
            await self.human_like_wait(0.5, 1.0)
            
            # 人間のようなスクロール
            await self.human_like_scroll(page)
            await self.human_like_wait(0.5, 1.0)
            
            # 人間のようなホバー
            await self.human_like_hover(page)
            await self.human_like_wait(0.5, 1.0)
            
            # タイトル取得
            title = await page.title()
            logger.info(f"[TITLE] {title}")
            
            # HTML取得
            html = await page.content()
            
            # テキスト抽出
            soup = BeautifulSoup(html, 'lxml')
            
            # スクリプト・スタイル・ナビゲーション要素を削除
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'noscript']):
                tag.decompose()
            
            # メインコンテンツを抽出
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=lambda x: x and ('content' in x.lower() or 'main' in x.lower() or 'article' in x.lower()))
            
            if main_content:
                text = main_content.get_text(separator='\n', strip=True)
            else:
                text = soup.get_text(separator='\n', strip=True)
            
            # テキストの正規化
            import re
            text = re.sub(r'\n\n+', '\n\n', text)
            text = re.sub(r'[ \t]+', ' ', text)
            text = text.strip()
            
            if len(text) < 200:
                logger.warning(f"[SKIP] Text too short: {len(text)} chars")
                return None
            
            # ドメイン名を取得
            parsed = urlparse(url)
            domain = parsed.netloc
            
            # サンプル作成
            sample = {
                "text": text,
                "url": url,
                "domain": domain,
                "title": title,
                "source": "human_like_scraper",
                "crawled_at": datetime.now().isoformat(),
                "text_length": len(text),
                "depth": depth
            }
            
            self.visited_urls.add(url)
            logger.info(f"[OK] Scraped {len(text)} characters from {url} (depth: {depth})")
            
            return sample
            
        except PlaywrightTimeoutError:
            logger.warning(f"[TIMEOUT] Page load timeout: {url}")
            return None
        except Exception as e:
            logger.error(f"[ERROR] Failed to scrape {url}: {e}")
            return None
    
    async def scrape_with_auto_navigation(
        self,
        start_urls: List[str],
        max_pages: int = None
    ) -> List[Dict]:
        """
        自動ページ遷移付きスクレイピング
        
        Args:
            start_urls: 開始URLリスト
            max_pages: 最大ページ数
        
        Returns:
            収集したサンプルリスト
        """
        if max_pages is None:
            max_pages = self.max_pages
        
        samples = []
        url_queue = [(url, 0) for url in start_urls]  # (url, depth)
        
        async with async_playwright() as playwright:
            browser = await self.connect_to_cursor_browser(playwright)
            
            try:
                contexts = browser.contexts
                if contexts:
                    context = contexts[0]
                else:
                    context = await browser.new_context()
                
                page = await context.new_page()
                
                # User-Agentを設定
                await page.set_extra_http_headers({
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                })
                
                logger.info(f"[SCRAPE] Starting human-like scraping (max {max_pages} pages)...")
                
                while url_queue and len(samples) < max_pages:
                    url, depth = url_queue.pop(0)
                    
                    if url in self.visited_urls:
                        continue
                    
                    # ページをスクレイピング
                    sample = await self.scrape_page(page, url, depth)
                    
                    if sample:
                        samples.append(sample)
                        self.collected_samples.append(sample)
                    
                    # リンクを追跡する場合
                    if self.follow_links and depth < self.max_depth and len(samples) < max_pages:
                        links = await self.extract_links(page, url)
                        
                        # 新しいリンクをキューに追加
                        for link in links:
                            if link not in self.visited_urls and (link, depth + 1) not in url_queue:
                                url_queue.append((link, depth + 1))
                        
                        logger.info(f"[LINKS] Found {len(links)} links, queue size: {len(url_queue)}")
                    
                    # リクエスト間の待機（人間のような動作）
                    await self.human_like_wait(self.delay_per_request, self.delay_per_request * 2)
                
                await page.close()
                
            finally:
                if not self.use_cursor_browser:
                    await browser.close()
                else:
                    logger.info("[BROWSER] Keeping Cursor browser connection open")
        
        logger.info(f"[OK] Scraped {len(samples)} pages successfully")
        return samples
    
    def save_samples(self, samples: List[Dict], filename: str = None) -> Path:
        """サンプルを保存"""
        if filename is None:
            filename = f"human_like_scraped_{self.session_id}.jsonl"
        
        output_file = self.output_dir / filename
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"[OK] Saved {len(samples)} samples to {output_file}")
        return output_file


async def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Human-Like Web Scraping with Cursor Browser")
    parser.add_argument(
        '--urls',
        nargs='+',
        required=True,
        help='Starting URLs to scrape'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('D:/webdataset/processed'),
        help='Output directory'
    )
    parser.add_argument(
        '--use-cursor-browser',
        action='store_true',
        default=True,
        help='Use Cursor browser'
    )
    parser.add_argument(
        '--remote-debugging-port',
        type=int,
        default=9222,
        help='Remote debugging port'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=2.0,
        help='Delay between requests (seconds)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=30000,
        help='Page load timeout (milliseconds)'
    )
    parser.add_argument(
        '--max-pages',
        type=int,
        default=100,
        help='Maximum pages to scrape'
    )
    parser.add_argument(
        '--follow-links',
        action='store_true',
        default=True,
        help='Follow links automatically'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=3,
        help='Maximum depth for link following'
    )
    
    args = parser.parse_args()
    
    # スクレイパー作成
    scraper = HumanLikeScraper(
        output_dir=args.output,
        use_cursor_browser=args.use_cursor_browser,
        remote_debugging_port=args.remote_debugging_port,
        delay_per_request=args.delay,
        timeout=args.timeout,
        max_pages=args.max_pages,
        follow_links=args.follow_links,
        max_depth=args.max_depth
    )
    
    # スクレイピング実行
    samples = await scraper.scrape_with_auto_navigation(
        urls=args.urls,
        max_pages=args.max_pages
    )
    
    # 保存
    output_file = scraper.save_samples(samples)
    
    logger.info(f"[SUCCESS] Scraping completed. Output: {output_file}")
    return output_file


if __name__ == "__main__":
    asyncio.run(main())





