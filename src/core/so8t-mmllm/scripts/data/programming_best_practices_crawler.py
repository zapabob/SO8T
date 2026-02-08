#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
プログラミング言語ベストプラクティス収集クローラー

主要プログラミング言語のベストプラクティス、コード例、ドキュメントを収集

対象サイト:
- GitHub (公開リポジトリのREADME、コード例、ドキュメント)
- Stack Overflow (ベストプラクティス関連Q&A)
- MDN Web Docs (JavaScript, HTML, CSS)
- Python.org (公式ドキュメント)
- Rust Book (Rust公式ドキュメント)
- Go Documentation (Go公式ドキュメント)
- Microsoft Learn (C#, .NET)
- Oracle Java Documentation (Java)
- Node.js Documentation (Node.js)
- React Documentation (React)
"""

import sys
import json
import time
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

# Chromeヘッダー生成ユーティリティ
try:
    from .chrome_headers import get_chrome_headers, get_chrome_user_agent
except ImportError:
    # フォールバック: 直接定義
    def get_chrome_headers(referer: Optional[str] = None) -> Dict[str, str]:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Windows"',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Upgrade-Insecure-Requests': '1',
        }
        if referer:
            headers['Referer'] = referer
        return headers
    
    def get_chrome_user_agent() -> str:
        return 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class ProgrammingBestPracticesCrawler:
    """プログラミング言語ベストプラクティス収集クローラー"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: クローラー設定
        """
        self.config = config
        self.languages = config.get('languages', ['python', 'javascript', 'java', 'rust', 'go', 'csharp'])
        self.max_samples_per_language = config.get('max_samples_per_language', 10000)
        self.delay = config.get('delay', 1.0)
        self.timeout = config.get('timeout', 15)
        # Chromeに偽装
        self.user_agent = config.get('user_agent', get_chrome_user_agent())
        self.chrome_headers = get_chrome_headers()
        
        self.visited_urls: Set[str] = set()
        self.samples: List[Dict] = []
        
        # 言語別ソース定義
        self.language_sources = {
            'python': [
                'https://docs.python.org/3/',
                'https://www.python.org/dev/peps/',
                'https://realpython.com/',
            ],
            'javascript': [
                'https://developer.mozilla.org/en-US/docs/Web/JavaScript',
                'https://javascript.info/',
                'https://nodejs.org/en/docs/',
            ],
            'java': [
                'https://docs.oracle.com/javase/tutorial/',
                'https://docs.oracle.com/en/java/',
            ],
            'rust': [
                'https://doc.rust-lang.org/book/',
                'https://doc.rust-lang.org/rust-by-example/',
            ],
            'go': [
                'https://go.dev/doc/',
                'https://golang.org/doc/',
            ],
            'csharp': [
                'https://learn.microsoft.com/en-us/dotnet/csharp/',
                'https://docs.microsoft.com/en-us/dotnet/',
            ],
        }
    
    def _check_robots_txt(self, url: str) -> bool:
        """robots.txtをチェック"""
        try:
            parsed = urlparse(url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()
            return rp.can_fetch(self.user_agent, url)
        except Exception:
            return True  # robots.txtが読めない場合は許可
    
    def _fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """ページを取得"""
        if url in self.visited_urls:
            return None
        
        if not self._check_robots_txt(url):
            logger.debug(f"[ROBOTS] Disallowed: {url}")
            return None
        
        try:
            time.sleep(self.delay)
            # Chromeヘッダーを使用
            headers = self.chrome_headers.copy()
            response = requests.get(
                url,
                timeout=self.timeout,
                headers=headers,
                allow_redirects=True
            )
            response.raise_for_status()
            
            self.visited_urls.add(url)
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            logger.debug(f"Failed to fetch {url}: {e}")
            return None
    
    def _extract_text(self, soup: BeautifulSoup) -> str:
        """テキストを抽出"""
        # スクリプト、スタイル、ナビゲーションを削除
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'noscript']):
            tag.decompose()
        
        text = soup.get_text(separator='\n', strip=True)
        text = re.sub(r'\n\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()
    
    def _extract_code_blocks(self, soup: BeautifulSoup) -> List[str]:
        """コードブロックを抽出"""
        code_blocks = []
        for code_tag in soup.find_all(['code', 'pre']):
            code_text = code_tag.get_text(strip=True)
            if len(code_text) > 20:  # 最小長さ
                code_blocks.append(code_text)
        return code_blocks
    
    def _crawl_github_repos(self, language: str) -> List[Dict]:
        """GitHubから言語別のリポジトリを収集"""
        samples = []
        
        try:
            # GitHub Search APIを使用（認証なしでも利用可能、レート制限あり）
            search_query = f"language:{language} stars:>100"
            api_url = "https://api.github.com/search/repositories"
            params = {
                'q': search_query,
                'sort': 'stars',
                'order': 'desc',
                'per_page': 30  # レート制限を考慮
            }
            
            # GitHub API用のChromeヘッダー
            api_headers = {
                'User-Agent': self.user_agent,
                'Accept': 'application/vnd.github.v3+json'
            }
            response = requests.get(
                api_url,
                params=params,
                headers=api_headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                repos = data.get('items', [])[:20]  # 最大20リポジトリ
                
                for repo in repos:
                    repo_url = repo.get('html_url', '')
                    readme_url = f"https://api.github.com/repos/{repo.get('full_name', '')}/readme"
                    
                    # README取得
                    readme_headers = {
                        'User-Agent': self.user_agent,
                        'Accept': 'application/vnd.github.v3.raw'
                    }
                    readme_response = requests.get(
                        readme_url,
                        headers=readme_headers,
                        timeout=self.timeout
                    )
                    
                    if readme_response.status_code == 200:
                        readme_content = readme_response.text
                        if len(readme_content) > 200:
                            samples.append({
                                'url': repo_url,
                                'content': readme_content,
                                'title': repo.get('name', ''),
                                'language': language,
                                'domain': 'programming',
                                'subdomain': language,
                                'timestamp': datetime.now().isoformat(),
                                'type': 'github_readme',
                                'stars': repo.get('stargazers_count', 0)
                            })
                    
                    time.sleep(self.delay)
                    
                    if len(samples) >= self.max_samples_per_language // 2:
                        break
            
        except Exception as e:
            logger.debug(f"GitHub API failed for {language}: {e}")
        
        return samples
    
    def _crawl_documentation_site(self, base_url: str, language: str) -> List[Dict]:
        """ドキュメントサイトを再帰的にクロール"""
        samples = []
        urls_to_visit = [base_url]
        max_pages = 50  # サイトあたり最大50ページ
        
        while urls_to_visit and len(samples) < max_pages:
            current_url = urls_to_visit.pop(0)
            
            soup = self._fetch_page(current_url)
            if not soup:
                continue
            
            # メインテキスト抽出
            text = self._extract_text(soup)
            if len(text) > 200:
                samples.append({
                    'url': current_url,
                    'content': text,
                    'title': soup.title.string if soup.title else current_url,
                    'language': language,
                    'domain': 'programming',
                    'subdomain': language,
                    'timestamp': datetime.now().isoformat(),
                    'type': 'documentation'
                })
            
            # コードブロック抽出
            code_blocks = self._extract_code_blocks(soup)
            for code_block in code_blocks[:5]:  # ページあたり最大5個
                if len(code_block) > 50:
                    samples.append({
                        'url': current_url,
                        'content': code_block,
                        'title': f"Code example from {current_url}",
                        'language': language,
                        'domain': 'programming',
                        'subdomain': language,
                        'timestamp': datetime.now().isoformat(),
                        'type': 'code_example'
                    })
            
            # リンクを収集（同じドメイン内のみ）
            if len(urls_to_visit) < 20:  # キューサイズ制限
                parsed_base = urlparse(base_url)
                for link in soup.find_all('a', href=True):
                    href = link.get('href', '')
                    full_url = urljoin(base_url, href)
                    parsed_url = urlparse(full_url)
                    
                    if parsed_url.netloc == parsed_base.netloc and \
                       full_url not in self.visited_urls and \
                       full_url not in urls_to_visit and \
                       any(ext in full_url for ext in ['/docs/', '/documentation/', '/guide/', '/tutorial/']):
                        urls_to_visit.append(full_url)
            
            time.sleep(self.delay)
        
        return samples
    
    def crawl_language(self, language: str) -> List[Dict]:
        """特定言語のベストプラクティスを収集"""
        logger.info(f"[{language.upper()}] Starting crawl...")
        
        sources = self.language_sources.get(language, [])
        if not sources:
            logger.warning(f"[{language.upper()}] No sources defined")
            return []
        
        samples = []
        
        # GitHubからリポジトリを収集
        logger.info(f"[{language.upper()}] Crawling GitHub repositories...")
        github_samples = self._crawl_github_repos(language)
        samples.extend(github_samples)
        logger.info(f"[{language.upper()}] Collected {len(github_samples)} samples from GitHub")
        
        # ドキュメントサイトをクロール
        for source_url in sources:
            if len(samples) >= self.max_samples_per_language:
                break
            
            logger.info(f"[{language.upper()}] Crawling {source_url}...")
            doc_samples = self._crawl_documentation_site(source_url, language)
            samples.extend(doc_samples)
            logger.info(f"[{language.upper()}] Collected {len(doc_samples)} samples from {source_url}")
        
        logger.info(f"[{language.upper()}] Collected {len(samples)} total samples")
        return samples
    
    def crawl(self) -> List[Dict]:
        """全言語のベストプラクティスを収集"""
        logger.info("="*80)
        logger.info("Programming Best Practices Crawler")
        logger.info("="*80)
        
        all_samples = []
        for language in self.languages:
            samples = self.crawl_language(language)
            all_samples.extend(samples)
        
        self.samples = all_samples
        logger.info(f"[TOTAL] Collected {len(all_samples)} samples")
        return all_samples
    
    def save(self, output_path: Path):
        """結果を保存"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in self.samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"[SAVE] Saved {len(self.samples)} samples to {output_path}")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Programming Best Practices Crawler')
    parser.add_argument('--output', type=str, default='D:/webdataset/processed/programming_best_practices.jsonl')
    parser.add_argument('--languages', nargs='+', default=['python', 'javascript', 'java', 'rust', 'go', 'csharp'])
    parser.add_argument('--max-samples', type=int, default=10000)
    parser.add_argument('--delay', type=float, default=1.0)
    
    args = parser.parse_args()
    
    config = {
        'languages': args.languages,
        'max_samples_per_language': args.max_samples,
        'delay': args.delay,
        'timeout': 15,
        'user_agent': 'SO8T-DataCollector/1.0 (Research Purpose)'
    }
    
    crawler = ProgrammingBestPracticesCrawler(config)
    crawler.crawl()
    crawler.save(Path(args.output))


if __name__ == '__main__':
    main()

