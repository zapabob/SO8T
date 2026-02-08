"""
安全なデータソースからのクロールスクリプト

Playwright + Chromiumを使用し、公開情報のみを収集。
利用規約遵守、robots.txt確認、NSFW/危険コンテンツのフィルタリングを実装。
"""

import asyncio
import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser

import aiohttp
from playwright.async_api import async_playwright, Page, Browser
from bs4 import BeautifulSoup
import tqdm


# 安全なソースURL（例）
SAFE_SOURCE_URLS = {
    "wikipedia_ja": [
        "https://ja.wikipedia.org/wiki/防衛省",
        "https://ja.wikipedia.org/wiki/航空宇宙",
        "https://ja.wikipedia.org/wiki/医療",
    ],
    "wikipedia_en": [
        "https://en.wikipedia.org/wiki/Defense",
        "https://en.wikipedia.org/wiki/Aerospace",
        "https://en.wikipedia.org/wiki/Medicine",
    ],
    "government_ja": [
        "https://www.mod.go.jp/",
        "https://www.jaxa.jp/",
        "https://www.mhlw.go.jp/",
    ],
    "government_us": [
        "https://www.defense.gov/",
        "https://www.nasa.gov/",
        "https://www.fda.gov/",
    ],
    "tech_blogs": [
        "https://qiita.com/",
        "https://zenn.dev/",
        "https://note.com/tech",
    ],
}


class SafeCrawler:
    """安全なデータクローラー"""
    
    def __init__(
        self,
        output_dir: Path,
        max_pages_per_domain: int = 100,
        respect_robots_txt: bool = True,
        delay_seconds: float = 1.0,
    ):
        """
        Args:
            output_dir: 出力ディレクトリ
            max_pages_per_domain: ドメインあたりの最大ページ数
            respect_robots_txt: robots.txtを尊重するか
            delay_seconds: リクエスト間の遅延（秒）
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_pages_per_domain = max_pages_per_domain
        self.respect_robots_txt = respect_robots_txt
        self.delay_seconds = delay_seconds
        self.robots_cache: Dict[str, Optional[RobotFileParser]] = {}
        self.crawled_urls = set()
        self.results: List[Dict[str, Any]] = []
    
    def check_robots_txt(self, url: str) -> bool:
        """
        robots.txtをチェック
        
        Args:
            url: チェック対象のURL
        
        Returns:
            クロール可能かどうか
        """
        if not self.respect_robots_txt:
            return True
        
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        
        if domain not in self.robots_cache:
            robots_url = urljoin(domain, "/robots.txt")
            rp = RobotFileParser()
            try:
                rp.set_url(robots_url)
                rp.read()
                self.robots_cache[domain] = rp
            except Exception:
                self.robots_cache[domain] = None
        
        rp = self.robots_cache.get(domain)
        if rp is None:
            return True  # robots.txtが読めない場合は許可
        
        return rp.can_fetch("*", url)
    
    def extract_main_content(self, html: str, url: str) -> Optional[str]:
        """
        HTMLから本文を抽出
        
        Args:
            html: HTMLコンテンツ
            url: URL
        
        Returns:
            抽出されたテキスト
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # ナビゲーション、広告、フッターを除去
        for tag in soup.find_all(['nav', 'header', 'footer', 'aside', 'script', 'style']):
            tag.decompose()
        
        # 本文を抽出（サイトごとに調整が必要）
        main_content = None
        
        # Wikipedia
        if 'wikipedia.org' in url:
            content_div = soup.find('div', {'id': 'mw-content-text'})
            if content_div:
                main_content = content_div.get_text(separator='\n', strip=True)
        
        # 一般的な記事ページ
        if not main_content:
            article = soup.find('article') or soup.find('main') or soup.find('div', class_=re.compile(r'content|main|article'))
            if article:
                main_content = article.get_text(separator='\n', strip=True)
        
        # フォールバック: body全体
        if not main_content:
            body = soup.find('body')
            if body:
                main_content = body.get_text(separator='\n', strip=True)
        
        if main_content:
            # 余分な空白を整理
            main_content = re.sub(r'\n\s*\n', '\n\n', main_content)
            main_content = main_content.strip()
        
        return main_content
    
    def filter_dangerous_content(self, text: str) -> bool:
        """
        危険なコンテンツをフィルタ
        
        Args:
            text: チェック対象のテキスト
        
        Returns:
            安全かどうか（True=安全、False=危険）
        """
        if not text:
            return False
        
        # 危険なキーワードパターン（具体的手順を含むもの）
        dangerous_patterns = [
            r'爆発物.*作り方',
            r'兵器.*製造.*手順',
            r'サイバー攻撃.*実行',
            r'ハッキング.*方法',
            r'違法.*薬物.*合成',
        ]
        
        text_lower = text.lower()
        for pattern in dangerous_patterns:
            if re.search(pattern, text_lower):
                return False
        
        # 個人情報パターン（電話番号、メールアドレス等）
        personal_info_patterns = [
            r'\d{3}-\d{4}-\d{4}',  # 電話番号
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # メール
        ]
        
        for pattern in personal_info_patterns:
            if re.search(pattern, text):
                return False
        
        return True
    
    async def fetch_page(self, page: Page, url: str) -> Optional[Dict[str, Any]]:
        """
        ページを取得
        
        Args:
            page: Playwright Page
            url: URL
        
        Returns:
            取得結果の辞書
        """
        if url in self.crawled_urls:
            return None
        
        if not self.check_robots_txt(url):
            return None
        
        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
            html = await page.content()
            
            content = self.extract_main_content(html, url)
            if not content:
                return None
            
            # 危険なコンテンツをフィルタ
            if not self.filter_dangerous_content(content):
                return None
            
            self.crawled_urls.add(url)
            
            return {
                "url": url,
                "content": content,
                "title": await page.title(),
                "timestamp": datetime.utcnow().isoformat(),
                "domain": urlparse(url).netloc,
            }
        
        except Exception as e:
            print(f"[WARNING] Failed to fetch {url}: {e}")
            return None
    
    async def crawl_domain(self, browser: Browser, urls: List[str], domain_name: str):
        """
        ドメインをクロール
        
        Args:
            browser: Playwright Browser
            urls: URLリスト
            domain_name: ドメイン名
        """
        page = await browser.new_page()
        
        domain_results = []
        for url in urls[:self.max_pages_per_domain]:
            result = await self.fetch_page(page, url)
            if result:
                domain_results.append(result)
                self.results.append(result)
            
            await asyncio.sleep(self.delay_seconds)
        
        await page.close()
        
        # ドメインごとに保存
        if domain_results:
            output_file = self.output_dir / f"{domain_name}_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in domain_results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            print(f"[INFO] Saved {len(domain_results)} pages from {domain_name}")
    
    async def crawl_all(self, source_urls: Dict[str, List[str]]):
        """
        すべてのソースをクロール
        
        Args:
            source_urls: ソースURLの辞書
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            
            for domain_name, urls in source_urls.items():
                print(f"[INFO] Crawling {domain_name}...")
                await self.crawl_domain(browser, urls, domain_name)
            
            await browser.close()
        
        # 全体の結果を保存
        if self.results:
            output_file = self.output_dir / f"all_crawled_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in self.results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            print(f"[INFO] Total: {len(self.results)} pages crawled")


async def main():
    parser = argparse.ArgumentParser(description="Crawl safe data sources")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/crawled"),
        help="Output directory",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=100,
        help="Max pages per domain",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between requests (seconds)",
    )
    parser.add_argument(
        "--no-robots",
        action="store_true",
        help="Don't respect robots.txt",
    )
    
    args = parser.parse_args()
    
    crawler = SafeCrawler(
        output_dir=args.output_dir,
        max_pages_per_domain=args.max_pages,
        respect_robots_txt=not args.no_robots,
        delay_seconds=args.delay,
    )
    
    await crawler.crawl_all(SAFE_SOURCE_URLS)
    
    print(f"[SUCCESS] Crawling completed. Total: {len(crawler.results)} pages")


if __name__ == "__main__":
    asyncio.run(main())

