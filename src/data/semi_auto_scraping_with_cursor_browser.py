#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cursorブラウザを使った半自動Webスクレイピングスクリプト

Chrome DevTools MCPを使ってCursorのブラウザを操作し、
Playwrightと組み合わせて半自動スクレイピングを実行します。

Usage:
    python scripts/data/semi_auto_scraping_with_cursor_browser.py --urls https://ja.wikipedia.org/wiki/メインページ
"""

import sys
import json
import logging
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urlparse

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Playwrightインポート
try:
    from playwright.async_api import async_playwright, Browser, Page
except ImportError:
    print("[ERROR] Playwright not installed. Install with: pip install playwright")
    sys.exit(1)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/semi_auto_scraping.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SemiAutoScraper:
    """Cursorブラウザを使った半自動スクレイピングクラス"""
    
    def __init__(
        self,
        output_dir: Path,
        use_cursor_browser: bool = True,
        remote_debugging_port: int = 9222,
        delay_per_request: float = 2.0,
        timeout: int = 30000,
        interactive: bool = True
    ):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリ
            use_cursor_browser: Cursorのブラウザを使用するか
            remote_debugging_port: リモートデバッグポート
            delay_per_request: リクエスト間の遅延（秒）
            timeout: タイムアウト（ミリ秒）
            interactive: 対話モード（ユーザー確認を求める）
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_cursor_browser = use_cursor_browser
        self.remote_debugging_port = remote_debugging_port
        self.delay_per_request = delay_per_request
        self.timeout = timeout
        self.interactive = interactive
        
        self.collected_samples: List[Dict] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info("="*80)
        logger.info("Semi-Auto Scraper Initialized")
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Use Cursor browser: {self.use_cursor_browser}")
        logger.info(f"Remote debugging port: {self.remote_debugging_port}")
        logger.info(f"Interactive mode: {self.interactive}")
    
    async def connect_to_cursor_browser(self, playwright) -> Optional[Browser]:
        """
        Cursorのブラウザに接続
        
        Args:
            playwright: Playwrightインスタンス
        
        Returns:
            Browserインスタンス
        """
        if not self.use_cursor_browser:
            logger.info("[BROWSER] Launching new browser...")
            browser = await playwright.chromium.launch(headless=False)
            logger.info("[OK] Browser launched")
            return browser
        
        try:
            logger.info(f"[BROWSER] Connecting to Cursor browser on port {self.remote_debugging_port}...")
            
            # CDPエンドポイントURL
            cdp_endpoint = f"http://127.0.0.1:{self.remote_debugging_port}"
            
            browser = await playwright.chromium.connect_over_cdp(cdp_endpoint)
            
            # 接続確認
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
    
    async def scrape_page(
        self,
        page: Page,
        url: str,
        wait_for_user: bool = True
    ) -> Optional[Dict]:
        """
        単一ページをスクレイピング
        
        Args:
            page: Playwright Pageオブジェクト
            url: スクレイピングするURL
            wait_for_user: ユーザーの確認を待つか
        
        Returns:
            サンプル辞書（成功時）またはNone
        """
        try:
            logger.info(f"[SCRAPE] Navigating to: {url}")
            
            # ページに移動
            await page.goto(url, timeout=self.timeout, wait_until="networkidle")
            
            # 対話モードの場合、ユーザーに確認を求める
            if self.interactive and wait_for_user:
                logger.info("[INTERACTIVE] Page loaded. Please review the page in the browser.")
                logger.info("[INTERACTIVE] Press Enter to continue scraping, or type 'skip' to skip this page...")
                
                # ユーザー入力を待つ（非同期で実装）
                user_input = await asyncio.to_thread(input, "> ")
                
                if user_input.strip().lower() == 'skip':
                    logger.info("[SKIP] User skipped this page")
                    return None
            
            # タイトル取得
            title = await page.title()
            logger.info(f"[TITLE] {title}")
            
            # HTML取得
            html = await page.content()
            
            # テキスト抽出（簡易版）
            from bs4 import BeautifulSoup
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
            
            if len(text) < 100:
                logger.warning(f"[WARNING] Text too short: {len(text)} chars")
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
                "source": "semi_auto_scraper_cursor_browser",
                "crawled_at": datetime.now().isoformat(),
                "text_length": len(text)
            }
            
            logger.info(f"[OK] Scraped {len(text)} characters from {url}")
            return sample
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to scrape {url}: {e}")
            return None
    
    async def scrape_urls(
        self,
        urls: List[str],
        wait_for_each: bool = True
    ) -> List[Dict]:
        """
        URLリストをスクレイピング
        
        Args:
            urls: スクレイピングするURLリスト
            wait_for_each: 各ページでユーザー確認を待つか
        
        Returns:
            収集したサンプルリスト
        """
        samples = []
        
        async with async_playwright() as playwright:
            browser = await self.connect_to_cursor_browser(playwright)
            
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
                
                # User-Agentを設定
                await page.set_extra_http_headers({
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                })
                
                logger.info(f"[SCRAPE] Starting to scrape {len(urls)} URLs...")
                
                for i, url in enumerate(urls, 1):
                    logger.info(f"[PROGRESS] [{i}/{len(urls)}] Processing: {url}")
                    
                    sample = await self.scrape_page(page, url, wait_for_user=wait_for_each)
                    
                    if sample:
                        samples.append(sample)
                        self.collected_samples.append(sample)
                    
                    # リクエスト間の遅延
                    if i < len(urls):
                        await asyncio.sleep(self.delay_per_request)
                
                await page.close()
                
            finally:
                # Cursorブラウザに接続している場合は閉じない
                if not self.use_cursor_browser:
                    await browser.close()
                else:
                    logger.info("[BROWSER] Keeping Cursor browser connection open")
        
        logger.info(f"[OK] Scraped {len(samples)} pages successfully")
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
            filename = f"semi_auto_scraped_{self.session_id}.jsonl"
        
        output_file = self.output_dir / filename
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"[OK] Saved {len(samples)} samples to {output_file}")
        return output_file


async def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Semi-Auto Scraping with Cursor Browser")
    parser.add_argument(
        '--urls',
        nargs='+',
        required=True,
        help='URLs to scrape'
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
        help='Use Cursor browser (connect via CDP)'
    )
    parser.add_argument(
        '--remote-debugging-port',
        type=int,
        default=9222,
        help='Remote debugging port (default: 9222)'
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
        '--non-interactive',
        action='store_true',
        help='Run in non-interactive mode (no user prompts)'
    )
    
    args = parser.parse_args()
    
    # スクレイパー作成
    scraper = SemiAutoScraper(
        output_dir=args.output,
        use_cursor_browser=args.use_cursor_browser,
        remote_debugging_port=args.remote_debugging_port,
        delay_per_request=args.delay,
        timeout=args.timeout,
        interactive=not args.non_interactive
    )
    
    # スクレイピング実行
    samples = await scraper.scrape_urls(
        urls=args.urls,
        wait_for_each=not args.non_interactive
    )
    
    # 保存
    output_file = scraper.save_samples(samples)
    
    logger.info(f"[SUCCESS] Scraping completed. Output: {output_file}")
    return output_file


if __name__ == "__main__":
    asyncio.run(main())





