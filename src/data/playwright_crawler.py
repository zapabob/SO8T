#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PlaywrightベースWebスクレイピングクローラー

Cursorのブラウザに接続してWebスクレイピングを実行します。
既存のrequestsベースのクローラーをPlaywrightに置き換えます。

Usage:
    python scripts/data/playwright_crawler.py --output D:\webdataset\processed --use-cursor-browser
"""

import sys
import json
import logging
import re
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

from playwright.async_api import async_playwright, Browser, Page, TimeoutError as PlaywrightTimeoutError
from bs4 import BeautifulSoup
from tqdm import tqdm

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "so8t-mmllm"))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/playwright_crawler.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# robots.txt キャッシュ
ROBOTS_CACHE: Dict[str, RobotFileParser] = {}


def check_robots_txt(url: str) -> bool:
    """robots.txtをチェック"""
    parsed = urlparse(url)
    domain = f"{parsed.scheme}://{parsed.netloc}"
    
    if domain not in ROBOTS_CACHE:
        rp = RobotFileParser()
        robots_url = urljoin(domain, '/robots.txt')
        try:
            rp.set_url(robots_url)
            rp.read()
            ROBOTS_CACHE[domain] = rp
        except Exception:
            ROBOTS_CACHE[domain] = None
    
    rp = ROBOTS_CACHE[domain]
    if rp is None:
        return True
    
    return rp.can_fetch('*', url)


def extract_text_from_html(html: str) -> str:
    """HTMLからテキストを抽出"""
    soup = BeautifulSoup(html, 'lxml')
    
    # スクリプト・スタイル・ナビゲーション要素を削除
    for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'noscript']):
        tag.decompose()
    
    # メインコンテンツを抽出
    main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile('content|main|article'))
    
    if main_content:
        text = main_content.get_text(separator='\n', strip=True)
    else:
        text = soup.get_text(separator='\n', strip=True)
    
    # テキストの正規化
    text = re.sub(r'\n\n+', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


class PlaywrightCrawler:
    """PlaywrightベースWebスクレイピングクローラー"""
    
    def __init__(
        self,
        output_dir: Path,
        use_cursor_browser: bool = True,
        remote_debugging_port: int = 9222,
        headless: bool = False,
        delay_per_request: float = 1.0,
        timeout: int = 30000,
        max_pages: int = 1000,
        min_text_length: int = 200,
        max_text_length: int = 10000,
    ):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリ
            use_cursor_browser: Cursorのブラウザを使用するか
            remote_debugging_port: リモートデバッグポート（Cursorブラウザ使用時）
            headless: ヘッドレスモード（Cursorブラウザ使用時はFalse）
            delay_per_request: リクエスト間の遅延（秒）
            timeout: タイムアウト（ミリ秒）
            max_pages: 最大ページ数
            min_text_length: 最小テキスト長
            max_text_length: 最大テキスト長
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_cursor_browser = use_cursor_browser
        self.remote_debugging_port = remote_debugging_port
        self.headless = headless
        self.delay_per_request = delay_per_request
        self.timeout = timeout
        self.max_pages = max_pages
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        
        self.visited_urls: Set[str] = set()
        self.collected_samples: List[Dict] = []
        
        self.checkpoint_dir = self.output_dir / "checkpoints" / "playwright_crawler"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # スクリーンショット保存ディレクトリ
        self.screenshots_dir = Path("D:/webdataset/screenshots")
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self.enable_screenshots = True  # スクリーンショット有効化
        
        logger.info("="*80)
        logger.info("Playwright Crawler Initialized")
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Use Cursor browser: {self.use_cursor_browser}")
        logger.info(f"Remote debugging port: {self.remote_debugging_port}")
        logger.info(f"Headless mode: {self.headless}")
        logger.info(f"Screenshots directory: {self.screenshots_dir}")
        logger.info(f"Screenshots enabled: {self.enable_screenshots}")
    
    async def connect_to_browser(self, playwright) -> Optional[Browser]:
        """
        ブラウザに接続
        
        Args:
            playwright: Playwrightインスタンス
        
        Returns:
            Browserインスタンス
        """
        if self.use_cursor_browser:
            try:
                # Cursorのブラウザに接続（CDP経由）
                logger.info(f"[BROWSER] Connecting to Cursor browser on port {self.remote_debugging_port}...")
                
                # CDPエンドポイントURL
                cdp_endpoint = f"http://127.0.0.1:{self.remote_debugging_port}"
                
                browser = await playwright.chromium.connect_over_cdp(cdp_endpoint)
                
                # 接続確認
                contexts = browser.contexts
                if contexts:
                    logger.info(f"[OK] Connected to Cursor browser (found {len(contexts)} contexts)")
                    return browser
                else:
                    # 新しいコンテキストを作成
                    logger.info("[INFO] No existing contexts found, creating new context...")
                    await browser.new_context()
                    logger.info("[OK] New context created")
                    return browser
                    
            except Exception as e:
                logger.warning(f"[WARNING] Failed to connect to Cursor browser: {e}")
                logger.info("[INFO] Falling back to launching new browser...")
                # フォールバック: 新しいブラウザを起動
                browser = await playwright.chromium.launch(headless=self.headless)
                logger.info("[OK] New browser launched")
                return browser
        else:
            # 新しいブラウザを起動
            logger.info("[BROWSER] Launching new browser...")
            browser = await playwright.chromium.launch(headless=self.headless)
            logger.info("[OK] Browser launched")
            return browser
    
    async def crawl_page(
        self,
        page: Page,
        url: str,
        domain: str = None
    ) -> Optional[Dict]:
        """
        単一ページをクロール
        
        Args:
            page: Playwright Pageオブジェクト
            url: クロールするURL
            domain: ドメイン名（オプション）
        
        Returns:
            サンプル辞書（成功時）またはNone
        """
        if url in self.visited_urls:
            return None
        
        if not check_robots_txt(url):
            logger.debug(f"[SKIP] robots.txt disallows: {url}")
            return None
        
        try:
            # レート制限
            await asyncio.sleep(self.delay_per_request)
            
            # ページに移動
            await page.goto(url, timeout=self.timeout, wait_until="networkidle")
            
            # スクリーンショットを撮影（有効な場合）
            if self.enable_screenshots:
                try:
                    screenshot_path = self.screenshots_dir / f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(url) % 10000}.png"
                    await page.screenshot(path=str(screenshot_path), full_page=False)
                    logger.debug(f"[SCREENSHOT] Saved: {screenshot_path}")
                except Exception as e:
                    logger.warning(f"[SCREENSHOT] Failed to capture screenshot: {e}")
            
            # タイトル取得
            title = await page.title()
            
            # HTML取得
            html = await page.content()
            
            # テキスト抽出
            text = extract_text_from_html(html)
            
            if len(text) < self.min_text_length:
                logger.debug(f"[SKIP] Text too short: {url} ({len(text)} chars)")
                return None
            
            if len(text) > self.max_text_length:
                text = text[:self.max_text_length]
            
            # ドメイン名を取得
            if domain is None:
                parsed = urlparse(url)
                domain = parsed.netloc
            
            # サンプル作成
            sample = {
                "text": text,
                "url": url,
                "domain": domain,
                "title": title,
                "source": "playwright_crawler",
                "crawled_at": datetime.now().isoformat(),
                "text_length": len(text)
            }
            
            self.visited_urls.add(url)
            return sample
            
        except PlaywrightTimeoutError:
            logger.warning(f"[TIMEOUT] Page load timeout: {url}")
            return None
        except Exception as e:
            logger.warning(f"[ERROR] Failed to crawl {url}: {e}")
            return None
    
    async def crawl_urls(
        self,
        urls: List[str],
        domain: str = None,
        max_pages: int = None
    ) -> List[Dict]:
        """
        URLリストをクロール
        
        Args:
            urls: クロールするURLリスト
            domain: ドメイン名（オプション）
            max_pages: 最大ページ数
        
        Returns:
            収集したサンプルリスト
        """
        if max_pages is None:
            max_pages = self.max_pages
        
        samples = []
        
        async with async_playwright() as playwright:
            browser = await self.connect_to_browser(playwright)
            
            try:
                # コンテキストを取得または作成
                contexts = browser.contexts
                if contexts:
                    context = contexts[0]
                    logger.info("[BROWSER] Using existing context")
                else:
                    context = await browser.new_context()
                    logger.info("[BROWSER] Created new context")
                
                # ページを作成
                page = await context.new_page()
                
                # User-Agentを設定（Chromeに偽装）
                await page.set_extra_http_headers({
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                })
                
                logger.info(f"[CRAWL] Starting to crawl {len(urls)} URLs (max {max_pages} pages)...")
                
                for i, url in enumerate(tqdm(urls[:max_pages], desc="Crawling pages")):
                    sample = await self.crawl_page(page, url, domain)
                    if sample:
                        samples.append(sample)
                        self.collected_samples.append(sample)
                    
                    # 定期的にスクリーンショットを撮影（10ページごと）
                    if self.enable_screenshots and (i + 1) % 10 == 0:
                        try:
                            screenshot_path = self.screenshots_dir / f"periodic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                            await page.screenshot(path=str(screenshot_path), full_page=False)
                            logger.info(f"[SCREENSHOT] Periodic screenshot saved: {screenshot_path}")
                        except Exception as e:
                            logger.warning(f"[SCREENSHOT] Failed to capture periodic screenshot: {e}")
                
                await page.close()
                
            finally:
                # Cursorブラウザに接続している場合は閉じない（既存のブラウザを閉じない）
                if not self.use_cursor_browser:
                    await browser.close()
                else:
                    logger.info("[BROWSER] Keeping Cursor browser connection open")
        
        logger.info(f"[OK] Crawled {len(samples)} pages successfully")
        return samples
    
    def save_samples(self, samples: List[Dict], filename: str = None) -> Path:
        """
        サンプルを保存
        
        Args:
            samples: サンプルリスト
            filename: ファイル名（オプション）
        
        Returns:
            保存されたファイルパス
        """
        if filename is None:
            filename = f"playwright_crawled_{self.session_id}.jsonl"
        
        output_file = self.output_dir / filename
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"[OK] Saved {len(samples)} samples to {output_file}")
        return output_file
    
    async def crawl_from_config(self, config: Dict) -> Path:
        """
        設定ファイルからクロール
        
        Args:
            config: 設定辞書
        
        Returns:
            保存されたファイルパス
        """
        data_sources = config.get('data_sources', {})
        urls = []
        
        # 各データソースからURLを収集
        if data_sources.get('enable_kanpou_4web', False):
            # 4web版官報のURLを追加
            urls.extend([
                "https://kanpou4web.nta.go.jp/",
                # 追加のURL...
            ])
        
        if data_sources.get('enable_egov', False):
            # e-GovのURLを追加
            urls.extend([
                "https://www.e-gov.go.jp/",
                # 追加のURL...
            ])
        
        if data_sources.get('enable_wikipedia_ja', False):
            # Wikipedia日本語版のURLを追加
            urls.extend([
                "https://ja.wikipedia.org/wiki/メインページ",
                # 追加のURL...
            ])
        
        # クロール実行
        samples = await self.crawl_urls(urls)
        
        # 保存
        output_file = self.save_samples(samples)
        
        return output_file


async def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Playwright Web Crawler")
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('D:/webdataset/processed'),
        help='Output directory'
    )
    parser.add_argument(
        '--use-cursor-browser',
        action='store_true',
        help='Use Cursor browser (connect via CDP)'
    )
    parser.add_argument(
        '--remote-debugging-port',
        type=int,
        default=9222,
        help='Remote debugging port (default: 9222)'
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run in headless mode'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
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
        default=1000,
        help='Maximum pages to crawl'
    )
    parser.add_argument(
        '--config',
        type=Path,
        help='Configuration file path'
    )
    
    args = parser.parse_args()
    
    # 設定ファイル読み込み（オプション）
    config = {}
    if args.config and args.config.exists():
        import yaml
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    
    # クローラー作成
    crawler = PlaywrightCrawler(
        output_dir=args.output,
        use_cursor_browser=args.use_cursor_browser,
        remote_debugging_port=args.remote_debugging_port,
        headless=args.headless,
        delay_per_request=args.delay,
        timeout=args.timeout,
        max_pages=args.max_pages
    )
    
    # クロール実行
    if config:
        output_file = await crawler.crawl_from_config(config)
    else:
        # デフォルトのURLリストでクロール
        urls = [
            "https://ja.wikipedia.org/wiki/メインページ",
            "https://www.e-gov.go.jp/",
        ]
        samples = await crawler.crawl_urls(urls)
        output_file = crawler.save_samples(samples)
    
    logger.info(f"[SUCCESS] Crawling completed. Output: {output_file}")
    return output_file


if __name__ == "__main__":
    asyncio.run(main())

