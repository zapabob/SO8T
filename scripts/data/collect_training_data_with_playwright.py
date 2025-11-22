#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Playwrightベース学習用データ収集スクリプト

日本語ドメイン別知識とコーディング能力向上を狙ったWebスクレイピングデータ収集

Usage:
    python scripts/data/collect_training_data_with_playwright.py \
        --output D:/webdataset/training_data_collected \
        --sources wikipedia_ja,github,stackoverflow \
        --target_samples 100000
"""

import sys
import json
import logging
import argparse
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse, urljoin
from collections import Counter
import re

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Playwrightインポート
try:
    from playwright.async_api import async_playwright, Browser, Page, TimeoutError as PlaywrightTimeoutError
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.error("[ERROR] Playwright not installed. Install with: pip install playwright && playwright install chromium")

# BeautifulSoupインポート
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logger.error("[ERROR] BeautifulSoup not installed. Install with: pip install beautifulsoup4")

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/collect_training_data_playwright.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrainingDataCollector:
    """Playwrightベース学習用データ収集クラス"""
    
    # データソース設定
    DATA_SOURCES = {
        "wikipedia_ja": {
            "base_url": "https://ja.wikipedia.org",
            "start_urls": [
                "https://ja.wikipedia.org/wiki/メインページ",
                "https://ja.wikipedia.org/wiki/Category:技術",
                "https://ja.wikipedia.org/wiki/Category:科学",
                "https://ja.wikipedia.org/wiki/Category:プログラミング",
            ],
            "selectors": {
                "title": "h1.firstHeading",
                "content": "#mw-content-text",
                "links": "#mw-content-text a[href^='/wiki/']"
            },
            "min_text_length": 500,
            "max_text_length": 10000
        },
        "github": {
            "base_url": "https://github.com",
            "start_urls": [
                "https://github.com/topics/python",
                "https://github.com/topics/javascript",
                "https://github.com/topics/typescript",
                "https://github.com/topics/rust",
            ],
            "selectors": {
                "title": "h1",
                "content": "article",
                "links": "a[href^='/']"
            },
            "min_text_length": 200,
            "max_text_length": 5000
        },
        "stackoverflow": {
            "base_url": "https://ja.stackoverflow.com",
            "start_urls": [
                "https://ja.stackoverflow.com/questions/tagged/python",
                "https://ja.stackoverflow.com/questions/tagged/javascript",
                "https://ja.stackoverflow.com/questions/tagged/typescript",
            ],
            "selectors": {
                "title": "h1 a.question-hyperlink",
                "content": ".question .post-text",
                "links": ".question a[href^='/questions/']"
            },
            "min_text_length": 300,
            "max_text_length": 8000
        },
        "qiita": {
            "base_url": "https://qiita.com",
            "start_urls": [
                "https://qiita.com/tags/python",
                "https://qiita.com/tags/javascript",
                "https://qiita.com/tags/typescript",
            ],
            "selectors": {
                "title": "h1",
                "content": ".it-MdContent",
                "links": "a[href^='/items/']"
            },
            "min_text_length": 400,
            "max_text_length": 10000
        }
    }
    
    def __init__(
        self,
        output_dir: Path,
        sources: List[str],
        target_samples: int = 100000,
        use_cursor_browser: bool = True,
        remote_debugging_port: int = 9222,
        headless: bool = False,
        delay_per_request: float = 2.0,
        timeout: int = 30000,
        max_depth: int = 3,
        max_pages_per_source: int = 1000
    ):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリ
            sources: データソースリスト（wikipedia_ja, github, stackoverflow, qiita）
            target_samples: 目標サンプル数
            use_cursor_browser: Cursorブラウザを使用するか
            remote_debugging_port: リモートデバッグポート
            headless: ヘッドレスモード
            delay_per_request: リクエスト間の遅延（秒）
            timeout: タイムアウト（ミリ秒）
            max_depth: 最大クロール深度
            max_pages_per_source: ソースごとの最大ページ数
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError("Playwright not available. Install with: pip install playwright && playwright install chromium")
        if not BS4_AVAILABLE:
            raise RuntimeError("BeautifulSoup not available. Install with: pip install beautifulsoup4")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sources = sources
        self.target_samples = target_samples
        self.use_cursor_browser = use_cursor_browser
        self.remote_debugging_port = remote_debugging_port
        self.headless = headless
        self.delay_per_request = delay_per_request
        self.timeout = timeout
        self.max_depth = max_depth
        self.max_pages_per_source = max_pages_per_source
        
        self.visited_urls: Set[str] = set()
        self.collected_samples: List[Dict] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # チェックポイント設定
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    async def connect_to_browser(self, playwright) -> Optional[Browser]:
        """ブラウザに接続"""
        if self.use_cursor_browser:
            try:
                # Cursorブラウザに接続（CDP経由）
                cdp_endpoint = f"http://127.0.0.1:{self.remote_debugging_port}"
                browser = await playwright.chromium.connect_over_cdp(cdp_endpoint)
                logger.info(f"[BROWSER] Connected to Cursor browser at {cdp_endpoint}")
                return browser
            except Exception as e:
                logger.warning(f"[BROWSER] Failed to connect to Cursor browser: {e}")
                logger.info("[BROWSER] Launching new browser...")
        
        # 新しいブラウザを起動
        browser = await playwright.chromium.launch(headless=self.headless)
        logger.info("[BROWSER] Browser launched")
        return browser
    
    def extract_text_from_html(self, html: str, selectors: Dict[str, str]) -> Dict[str, str]:
        """HTMLからテキストを抽出"""
        soup = BeautifulSoup(html, 'lxml')
        
        result = {}
        
        # タイトル抽出
        if "title" in selectors:
            title_elem = soup.select_one(selectors["title"])
            if title_elem:
                result["title"] = title_elem.get_text(strip=True)
        
        # コンテンツ抽出
        if "content" in selectors:
            content_elem = soup.select_one(selectors["content"])
            if content_elem:
                # スクリプト・スタイル・ナビゲーション要素を削除
                for elem in content_elem.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                    elem.decompose()
                
                text = content_elem.get_text(separator='\n', strip=True)
                # 余分な空白を削除
                text = re.sub(r'\n\s*\n', '\n\n', text)
                text = re.sub(r'[ \t]+', ' ', text)
                result["content"] = text.strip()
        
        # リンク抽出
        if "links" in selectors:
            links = []
            for link_elem in soup.select(selectors["links"]):
                href = link_elem.get('href', '')
                if href:
                    links.append(href)
            result["links"] = links
        
        return result
    
    async def scrape_page(
        self,
        page: Page,
        url: str,
        source_config: Dict,
        depth: int = 0
    ) -> Optional[Dict]:
        """
        単一ページをスクレイピング
        
        Args:
            page: Playwright Pageオブジェクト
            url: スクレイピングするURL
            source_config: ソース設定
            depth: 現在の深度
        
        Returns:
            サンプル辞書（成功時）またはNone
        """
        if url in self.visited_urls:
            return None
        
        if depth > self.max_depth:
            return None
        
        try:
            logger.info(f"[SCRAPE] [{depth}] {url}")
            
            # ページに移動
            await page.goto(url, timeout=self.timeout, wait_until="networkidle")
            
            # 待機（レート制限対策）
            await asyncio.sleep(self.delay_per_request)
            
            # HTML取得
            html = await page.content()
            
            # テキスト抽出
            extracted = self.extract_text_from_html(html, source_config["selectors"])
            
            if not extracted.get("content"):
                logger.debug(f"[SKIP] No content found: {url}")
                return None
            
            content = extracted["content"]
            min_length = source_config.get("min_text_length", 200)
            max_length = source_config.get("max_text_length", 10000)
            
            if len(content) < min_length:
                logger.debug(f"[SKIP] Text too short: {url} ({len(content)} chars)")
                return None
            
            if len(content) > max_length:
                content = content[:max_length]
            
            # サンプル作成
            sample = {
                "instruction": f"以下の{extracted.get('title', 'コンテンツ')}について説明してください。",
                "output": content,
                "title": extracted.get("title", ""),
                "url": url,
                "source": source_config.get("name", "unknown"),
                "depth": depth,
                "timestamp": datetime.now().isoformat()
            }
            
            self.visited_urls.add(url)
            return sample
            
        except PlaywrightTimeoutError:
            logger.warning(f"[TIMEOUT] {url}")
            return None
        except Exception as e:
            logger.error(f"[ERROR] Failed to scrape {url}: {e}")
            return None
    
    async def crawl_source(
        self,
        browser: Browser,
        source_name: str,
        source_config: Dict
    ) -> List[Dict]:
        """
        単一ソースをクロール
        
        Args:
            browser: ブラウザオブジェクト
            source_name: ソース名
            source_config: ソース設定
        
        Returns:
            収集されたサンプルリスト
        """
        logger.info(f"="*80)
        logger.info(f"Crawling source: {source_name}")
        logger.info(f"="*80)
        
        source_config["name"] = source_name
        samples = []
        urls_to_visit = list(source_config["start_urls"])
        visited_count = 0
        
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            while urls_to_visit and visited_count < self.max_pages_per_source:
                if len(samples) >= self.target_samples // len(self.sources):
                    logger.info(f"[COMPLETE] Target samples reached for {source_name}")
                    break
                
                url = urls_to_visit.pop(0)
                
                # 絶対URLに変換
                if not url.startswith("http"):
                    url = urljoin(source_config["base_url"], url)
                
                # ページスクレイピング
                sample = await self.scrape_page(page, url, source_config, depth=0)
                
                if sample:
                    samples.append(sample)
                    visited_count += 1
                    
                    # リンクを追加（深度制限内）
                    if "links" in sample and sample.get("depth", 0) < self.max_depth:
                        for link in sample.get("links", [])[:10]:  # 最大10リンク
                            if link.startswith("/"):
                                full_url = urljoin(source_config["base_url"], link)
                                if full_url not in self.visited_urls:
                                    urls_to_visit.append(full_url)
                
                # 進捗表示
                if visited_count % 10 == 0:
                    logger.info(f"[PROGRESS] {source_name}: {visited_count} pages, {len(samples)} samples")
        
        finally:
            await context.close()
        
        logger.info(f"[COMPLETE] {source_name}: {len(samples)} samples collected")
        return samples
    
    async def collect_all(self) -> List[Dict]:
        """すべてのソースからデータを収集"""
        logger.info("="*80)
        logger.info("Starting Playwright-based training data collection")
        logger.info("="*80)
        logger.info(f"Sources: {', '.join(self.sources)}")
        logger.info(f"Target samples: {self.target_samples:,}")
        logger.info("")
        
        all_samples = []
        
        async with async_playwright() as playwright:
            browser = await self.connect_to_browser(playwright)
            
            try:
                for source_name in self.sources:
                    if source_name not in self.DATA_SOURCES:
                        logger.warning(f"[SKIP] Unknown source: {source_name}")
                        continue
                    
                    source_config = self.DATA_SOURCES[source_name].copy()
                    samples = await self.crawl_source(browser, source_name, source_config)
                    all_samples.extend(samples)
                    
                    logger.info(f"[TOTAL] {len(all_samples)} samples collected so far")
                    
                    if len(all_samples) >= self.target_samples:
                        logger.info(f"[COMPLETE] Target samples reached: {len(all_samples)}")
                        break
                
            finally:
                await browser.close()
        
        return all_samples
    
    def save_samples(self, samples: List[Dict]):
        """サンプルを保存"""
        # JSONL形式で保存
        output_file = self.output_dir / f"training_data_{self.session_id}.jsonl"
        
        logger.info(f"Saving {len(samples):,} samples to {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"[OK] Saved to {output_file}")
        
        # 統計情報を保存
        stats_file = self.output_dir / f"stats_{self.session_id}.json"
        stats = {
            "total_samples": len(samples),
            "sources": Counter(s["source"] for s in samples),
            "avg_content_length": sum(len(s.get("output", "")) for s in samples) / len(samples) if samples else 0,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[STATS] Statistics saved to {stats_file}")
        logger.info(f"  Total samples: {stats['total_samples']:,}")
        logger.info(f"  Sources: {dict(stats['sources'])}")
        logger.info(f"  Avg content length: {stats['avg_content_length']:.0f} chars")


async def main():
    parser = argparse.ArgumentParser(description="Playwright-based training data collection")
    parser.add_argument("--output", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--sources", type=str, default="wikipedia_ja,github,stackoverflow",
                       help="Comma-separated list of sources (wikipedia_ja,github,stackoverflow,qiita)")
    parser.add_argument("--target_samples", type=int, default=100000,
                       help="Target number of samples")
    parser.add_argument("--use_cursor_browser", action="store_true",
                       help="Use Cursor browser (CDP connection)")
    parser.add_argument("--remote_debugging_port", type=int, default=9222,
                       help="Remote debugging port (for Cursor browser)")
    parser.add_argument("--headless", action="store_true",
                       help="Run browser in headless mode")
    parser.add_argument("--delay", type=float, default=2.0,
                       help="Delay between requests (seconds)")
    parser.add_argument("--timeout", type=int, default=30000,
                       help="Page load timeout (milliseconds)")
    parser.add_argument("--max_depth", type=int, default=3,
                       help="Maximum crawl depth")
    parser.add_argument("--max_pages_per_source", type=int, default=1000,
                       help="Maximum pages per source")
    
    args = parser.parse_args()
    
    if not PLAYWRIGHT_AVAILABLE:
        logger.error("[ERROR] Playwright not installed. Install with: pip install playwright && playwright install chromium")
        return 1
    
    sources = [s.strip() for s in args.sources.split(",")]
    
    collector = TrainingDataCollector(
        output_dir=Path(args.output),
        sources=sources,
        target_samples=args.target_samples,
        use_cursor_browser=args.use_cursor_browser,
        remote_debugging_port=args.remote_debugging_port,
        headless=args.headless,
        delay_per_request=args.delay,
        timeout=args.timeout,
        max_depth=args.max_depth,
        max_pages_per_source=args.max_pages_per_source
    )
    
    # データ収集実行
    samples = await collector.collect_all()
    
    # サンプル保存
    collector.save_samples(samples)
    
    logger.info("="*80)
    logger.info("Data collection completed!")
    logger.info("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))






















































































































