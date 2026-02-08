"""
公式ソースからの安全なデータ収集スクリプト

防衛白書PDF、NASA技術文書、PMDA添付文書、e-Gov法令、Wikipedia等の
公開情報を収集し、四重Thinking形式のJSONLに変換する。
"""

import asyncio
import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from urllib.parse import urlparse, urljoin
import sys

import aiohttp
from playwright.async_api import async_playwright, Page, Browser
from bs4 import BeautifulSoup
import io

# PDF処理ライブラリ（pypdfまたはPyPDF2）
try:
    import pypdf
    PDF_READER = pypdf.PdfReader
except ImportError:
    try:
        import PyPDF2
        PDF_READER = PyPDF2.PdfReader
    except ImportError:
        PDF_READER = None
        print("[WARNING] PDF library not found. PDF extraction will be disabled.")

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "so8t-mmllm" / "src"))


# 公式ソースURL定義
OFFICIAL_SOURCE_URLS = {
    "defense_ja": [
        "https://www.mod.go.jp/j/press/wp/index.html",  # 防衛白書
        "https://www.mod.go.jp/j/policy/agenda/guideline/index.html",  # 防衛3文書
    ],
    "defense_en": [
        "https://www.mod.go.jp/en/publ/w_paper/index.html",  # Defense White Paper (English)
    ],
    "aerospace": [
        "https://ntrs.nasa.gov/",  # NASA Technical Reports Server
        "https://www.jaxa.jp/",  # JAXA
    ],
    "medical": [
        "https://www.pmda.go.jp/",  # PMDA
    ],
    "law": [
        "https://elaws.e-gov.go.jp/",  # e-Gov法令データ
    ],
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
}


class OfficialSourceCrawler:
    """公式ソースクローラー"""
    
    def __init__(
        self,
        output_dir: Path,
        max_pages_per_domain: int = 50,
        delay_seconds: float = 2.0,
    ):
        """
        Args:
            output_dir: 出力ディレクトリ
            max_pages_per_domain: ドメインあたりの最大ページ数
            delay_seconds: リクエスト間の遅延（秒）
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_pages_per_domain = max_pages_per_domain
        self.delay_seconds = delay_seconds
        self.crawled_urls = set()
        self.results: List[Dict[str, Any]] = []
    
    def extract_text_from_pdf(self, pdf_bytes: bytes) -> Optional[str]:
        """
        PDFからテキストを抽出
        
        Args:
            pdf_bytes: PDFファイルのバイト列
        
        Returns:
            抽出されたテキスト
        """
        if PDF_READER is None:
            print("[WARNING] PDF library not available. Cannot extract PDF text.")
            return None
        
        try:
            pdf_file = io.BytesIO(pdf_bytes)
            pdf_reader = PDF_READER(pdf_file)
            text_parts = []
            
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            return "\n\n".join(text_parts) if text_parts else None
        except Exception as e:
            print(f"[WARNING] Failed to extract PDF text: {e}")
            return None
    
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
        
        main_content = None
        
        # 防衛省サイト
        if 'mod.go.jp' in url:
            content_div = soup.find('div', class_=re.compile(r'content|main|article|wp-content'))
            if content_div:
                main_content = content_div.get_text(separator='\n', strip=True)
        
        # Wikipedia
        elif 'wikipedia.org' in url:
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
    
    def determine_domain_label(self, url: str, content: str) -> str:
        """
        ドメインラベルを決定
        
        Args:
            url: URL
            content: コンテンツ
        
        Returns:
            ドメインラベル
        """
        url_lower = url.lower()
        content_lower = content.lower()[:500]  # 最初の500文字で判定
        
        if 'mod.go.jp' in url_lower or '防衛' in content_lower or 'defense' in content_lower:
            return "defense_public"
        elif 'nasa.gov' in url_lower or 'jaxa.jp' in url_lower or 'aerospace' in content_lower:
            return "aerospace"
        elif 'pmda.go.jp' in url_lower or '医薬品' in content_lower or 'medical' in content_lower:
            return "medical_reg"
        elif 'e-gov.go.jp' in url_lower or '法令' in content_lower or 'law' in content_lower:
            return "law_policy"
        elif 'wikipedia.org' in url_lower:
            return "wikipedia_ja_en"
        else:
            return "general"
    
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
        
        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
            
            # PDFの場合は特別処理
            if url.endswith('.pdf') or 'pdf' in url.lower():
                # PDFダウンロード
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            pdf_bytes = await response.read()
                            content = self.extract_text_from_pdf(pdf_bytes)
                        else:
                            return None
            else:
                html = await page.content()
                content = self.extract_main_content(html, url)
            
            if not content or len(content) < 100:
                return None
            
            domain_label = self.determine_domain_label(url, content)
            
            self.crawled_urls.add(url)
            
            return {
                "url": url,
                "content": content,
                "title": await page.title() if not url.endswith('.pdf') else url.split('/')[-1],
                "timestamp": datetime.utcnow().isoformat(),
                "domain": urlparse(url).netloc,
                "domain_label": domain_label,
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
            output_file = self.output_dir / f"all_official_sources_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in self.results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            print(f"[INFO] Total: {len(self.results)} pages crawled")


async def main():
    parser = argparse.ArgumentParser(description="Crawl official data sources")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/official_sources"),
        help="Output directory",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=50,
        help="Max pages per domain",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between requests (seconds)",
    )
    
    args = parser.parse_args()
    
    crawler = OfficialSourceCrawler(
        output_dir=args.output_dir,
        max_pages_per_domain=args.max_pages,
        delay_seconds=args.delay,
    )
    
    await crawler.crawl_all(OFFICIAL_SOURCE_URLS)
    
    print(f"[SUCCESS] Crawling completed. Total: {len(crawler.results)} pages")


if __name__ == "__main__":
    asyncio.run(main())

