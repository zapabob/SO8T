#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ドキュメンテーション収集スクリプト

APIドキュメント（OpenAPI/Swagger、REST API）、技術ドキュメント（README、Wiki、技術ブログ）、
コードコメントとdocstringを収集します。

Usage:
    python scripts/data/documentation_scraper.py --output D:\webdataset\processed\documentation
"""

import sys
import json
import logging
import argparse
import re
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import ast
import yaml

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/documentation_scraper.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DocumentationScraper:
    """ドキュメンテーション収集クラス"""
    
    def __init__(
        self,
        output_dir: Path,
        delay_per_request: float = 2.0,
        max_docs_per_source: int = 100
    ):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリ
            delay_per_request: リクエスト間の遅延（秒）
            max_docs_per_source: ソースあたりの最大ドキュメント数
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.delay_per_request = delay_per_request
        self.max_docs_per_source = max_docs_per_source
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.all_samples: List[Dict] = []
        
        logger.info("="*80)
        logger.info("Documentation Scraper Initialized")
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Max docs per source: {self.max_docs_per_source}")
    
    def extract_openapi_spec(self, url: str) -> Optional[Dict]:
        """OpenAPI/Swagger仕様を抽出"""
        try:
            response = self.session.get(url, timeout=30)
            if response.status_code != 200:
                return None
            
            content_type = response.headers.get('Content-Type', '').lower()
            
            # JSON形式のOpenAPI仕様
            if 'json' in content_type or url.endswith('.json'):
                try:
                    spec = response.json()
                    if 'openapi' in spec or 'swagger' in spec:
                        return {
                            'type': 'openapi',
                            'spec': spec,
                            'url': url
                        }
                except json.JSONDecodeError:
                    pass
            
            # YAML形式のOpenAPI仕様
            if 'yaml' in content_type or 'yml' in content_type or url.endswith(('.yaml', '.yml')):
                try:
                    spec = yaml.safe_load(response.text)
                    if isinstance(spec, dict) and ('openapi' in spec or 'swagger' in spec):
                        return {
                            'type': 'openapi',
                            'spec': spec,
                            'url': url
                        }
                except yaml.YAMLError:
                    pass
            
            # HTMLページ内のOpenAPI仕様へのリンクを検索
            if 'html' in content_type:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # JSON/YAMLファイルへのリンクを検索
                for link in soup.find_all('a', href=True):
                    href = link.get('href', '')
                    if any(ext in href.lower() for ext in ['.json', '.yaml', '.yml', 'openapi', 'swagger']):
                        full_url = urljoin(url, href)
                        return self.extract_openapi_spec(full_url)
            
            return None
            
        except Exception as e:
            logger.warning(f"[OPENAPI] Failed to extract from {url}: {e}")
            return None
    
    def extract_readme(self, url: str) -> Optional[Dict]:
        """READMEファイルを抽出"""
        readme_patterns = [
            r'README\.md',
            r'README\.txt',
            r'readme\.md',
            r'readme\.txt',
            r'/README',
            r'/readme'
        ]
        
        # GitHubリポジトリのREADME
        if 'github.com' in url:
            # GitHub APIを使用してREADMEを取得
            repo_match = re.search(r'github\.com/([^/]+)/([^/]+)', url)
            if repo_match:
                owner, repo = repo_match.groups()
                api_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
                
                try:
                    response = self.session.get(api_url, timeout=30)
                    if response.status_code == 200:
                        data = response.json()
                        import base64
                        content = base64.b64decode(data.get('content', '')).decode('utf-8')
                        
                        return {
                            'type': 'readme',
                            'content': content,
                            'url': url,
                            'source': 'github'
                        }
                except Exception as e:
                    logger.debug(f"[README] GitHub API failed for {url}: {e}")
        
        # 直接READMEファイルにアクセス
        for pattern in readme_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                try:
                    response = self.session.get(url, timeout=30)
                    if response.status_code == 200:
                        return {
                            'type': 'readme',
                            'content': response.text,
                            'url': url,
                            'source': 'direct'
                        }
                except Exception as e:
                    logger.debug(f"[README] Failed to fetch {url}: {e}")
        
        return None
    
    def extract_wiki_page(self, url: str) -> Optional[Dict]:
        """Wikiページを抽出"""
        try:
            response = self.session.get(url, timeout=30)
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Wikiコンテンツを抽出
            content_selectors = [
                'div.wiki-content',
                'div.markdown-body',
                'div.content',
                'article',
                'main'
            ]
            
            content = None
            for selector in content_selectors:
                element = soup.select_one(selector)
                if element:
                    content = element.get_text(separator='\n', strip=True)
                    break
            
            if not content:
                # フォールバック: body全体から抽出
                body = soup.find('body')
                if body:
                    content = body.get_text(separator='\n', strip=True)
            
            if content and len(content) > 100:  # 最小長チェック
                return {
                    'type': 'wiki',
                    'content': content,
                    'url': url,
                    'title': soup.title.string if soup.title else ''
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"[WIKI] Failed to extract from {url}: {e}")
            return None
    
    def extract_docstring_from_code(self, code: str, language: str = 'python') -> List[Dict]:
        """コードからdocstringとコメントを抽出"""
        docstrings = []
        
        if language.lower() == 'python':
            try:
                tree = ast.parse(code)
                
                for node in ast.walk(tree):
                    # 関数/クラスのdocstringを抽出
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                        docstring = ast.get_docstring(node)
                        if docstring:
                            docstrings.append({
                                'type': 'docstring',
                                'language': 'python',
                                'node_type': type(node).__name__,
                                'name': node.name if hasattr(node, 'name') else 'module',
                                'content': docstring
                            })
                
                # コメントを抽出
                for line_num, line in enumerate(code.split('\n'), 1):
                    stripped = line.strip()
                    if stripped.startswith('#'):
                        comment = stripped[1:].strip()
                        if len(comment) > 10:  # 最小長チェック
                            docstrings.append({
                                'type': 'comment',
                                'language': 'python',
                                'line': line_num,
                                'content': comment
                            })
            
            except SyntaxError:
                # 構文エラーの場合はスキップ
                pass
            except Exception as e:
                logger.debug(f"[DOCSTRING] Failed to parse Python code: {e}")
        
        elif language.lower() in ['javascript', 'typescript']:
            # JavaScript/TypeScriptのコメントを抽出
            for line_num, line in enumerate(code.split('\n'), 1):
                stripped = line.strip()
                # 単行コメント
                if stripped.startswith('//'):
                    comment = stripped[2:].strip()
                    if len(comment) > 10:
                        docstrings.append({
                            'type': 'comment',
                            'language': language,
                            'line': line_num,
                            'content': comment
                        })
                # JSDocスタイルのコメント
                elif stripped.startswith('/**') or stripped.startswith('*'):
                    comment = stripped.lstrip('*/').strip()
                    if len(comment) > 10:
                        docstrings.append({
                            'type': 'jsdoc',
                            'language': language,
                            'line': line_num,
                            'content': comment
                        })
        
        elif language.lower() in ['java', 'c', 'cpp', 'rust', 'go']:
            # Cスタイルのコメントを抽出
            for line_num, line in enumerate(code.split('\n'), 1):
                stripped = line.strip()
                # 単行コメント
                if stripped.startswith('//'):
                    comment = stripped[2:].strip()
                    if len(comment) > 10:
                        docstrings.append({
                            'type': 'comment',
                            'language': language,
                            'line': line_num,
                            'content': comment
                        })
                # ブロックコメント
                elif '/*' in stripped:
                    comment_match = re.search(r'/\*(.*?)\*/', stripped, re.DOTALL)
                    if comment_match:
                        comment = comment_match.group(1).strip()
                        if len(comment) > 10:
                            docstrings.append({
                                'type': 'comment',
                                'language': language,
                                'line': line_num,
                                'content': comment
                            })
        
        return docstrings
    
    def scrape_github_repository_docs(self, repo_url: str) -> List[Dict]:
        """GitHubリポジトリからドキュメンテーションを収集"""
        samples = []
        
        try:
            # READMEを抽出
            readme = self.extract_readme(repo_url)
            if readme:
                samples.append({
                    'type': 'readme',
                    'source': 'github',
                    'url': repo_url,
                    'content': readme.get('content', ''),
                    'metadata': {
                        'extracted_at': datetime.now().isoformat(),
                        'source_type': 'github_repository'
                    }
                })
            
            # Wikiページを検索
            wiki_url = repo_url.rstrip('/') + '/wiki'
            wiki = self.extract_wiki_page(wiki_url)
            if wiki:
                samples.append({
                    'type': 'wiki',
                    'source': 'github',
                    'url': wiki_url,
                    'content': wiki.get('content', ''),
                    'title': wiki.get('title', ''),
                    'metadata': {
                        'extracted_at': datetime.now().isoformat(),
                        'source_type': 'github_wiki'
                    }
                })
            
            time.sleep(self.delay_per_request)
            
        except Exception as e:
            logger.warning(f"[GITHUB_DOCS] Failed to scrape {repo_url}: {e}")
        
        return samples
    
    def scrape_api_documentation(self, base_url: str) -> List[Dict]:
        """APIドキュメンテーションを収集"""
        samples = []
        
        # OpenAPI/Swagger仕様の一般的なパス
        openapi_paths = [
            '/openapi.json',
            '/swagger.json',
            '/api/openapi.json',
            '/api/swagger.json',
            '/docs/openapi.json',
            '/docs/swagger.json',
            '/v1/openapi.json',
            '/v2/openapi.json',
            '/openapi.yaml',
            '/swagger.yaml'
        ]
        
        for path in openapi_paths:
            url = urljoin(base_url, path)
            spec = self.extract_openapi_spec(url)
            
            if spec:
                samples.append({
                    'type': 'openapi',
                    'source': 'api',
                    'url': url,
                    'spec': spec.get('spec', {}),
                    'metadata': {
                        'extracted_at': datetime.now().isoformat(),
                        'source_type': 'api_documentation'
                    }
                })
                logger.info(f"[API_DOCS] Found OpenAPI spec at {url}")
                break  # 1つ見つかれば十分
            
            time.sleep(self.delay_per_request)
        
        return samples
    
    def scrape_technical_blog(self, blog_url: str) -> List[Dict]:
        """技術ブログからドキュメンテーションを収集"""
        samples = []
        
        try:
            response = self.session.get(blog_url, timeout=30)
            if response.status_code != 200:
                return samples
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 記事コンテンツを抽出
            content_selectors = [
                'article',
                'div.article-content',
                'div.post-content',
                'div.content',
                'main'
            ]
            
            content = None
            for selector in content_selectors:
                element = soup.select_one(selector)
                if element:
                    content = element.get_text(separator='\n', strip=True)
                    break
            
            if not content:
                body = soup.find('body')
                if body:
                    content = body.get_text(separator='\n', strip=True)
            
            if content and len(content) > 200:  # 最小長チェック
                samples.append({
                    'type': 'technical_blog',
                    'source': 'blog',
                    'url': blog_url,
                    'content': content,
                    'title': soup.title.string if soup.title else '',
                    'metadata': {
                        'extracted_at': datetime.now().isoformat(),
                        'source_type': 'technical_blog'
                    }
                })
            
            time.sleep(self.delay_per_request)
            
        except Exception as e:
            logger.warning(f"[BLOG] Failed to scrape {blog_url}: {e}")
        
        return samples
    
    def save_samples(self, samples: List[Dict], doc_type: str):
        """サンプルを保存"""
        if not samples:
            return
        
        output_file = self.output_dir / f"documentation_{doc_type}_{self.session_id}.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"[SAVE] Saved {len(samples)} {doc_type} samples to {output_file}")
    
    def run_scraping(
        self,
        github_repos: Optional[List[str]] = None,
        api_urls: Optional[List[str]] = None,
        blog_urls: Optional[List[str]] = None
    ):
        """スクレイピングを実行"""
        logger.info("[SCRAPE] Starting documentation scraping...")
        
        all_samples = []
        
        # GitHubリポジトリからドキュメンテーションを収集
        if github_repos:
            logger.info(f"[SCRAPE] Scraping {len(github_repos)} GitHub repositories...")
            for repo_url in github_repos[:self.max_docs_per_source]:
                samples = self.scrape_github_repository_docs(repo_url)
                all_samples.extend(samples)
                time.sleep(self.delay_per_request)
        
        # APIドキュメンテーションを収集
        if api_urls:
            logger.info(f"[SCRAPE] Scraping {len(api_urls)} API documentation sites...")
            for api_url in api_urls[:self.max_docs_per_source]:
                samples = self.scrape_api_documentation(api_url)
                all_samples.extend(samples)
                time.sleep(self.delay_per_request)
        
        # 技術ブログを収集
        if blog_urls:
            logger.info(f"[SCRAPE] Scraping {len(blog_urls)} technical blogs...")
            for blog_url in blog_urls[:self.max_docs_per_source]:
                samples = self.scrape_technical_blog(blog_url)
                all_samples.extend(samples)
                time.sleep(self.delay_per_request)
        
        # タイプ別に保存
        type_groups = {}
        for sample in all_samples:
            doc_type = sample.get('type', 'unknown')
            if doc_type not in type_groups:
                type_groups[doc_type] = []
            type_groups[doc_type].append(sample)
        
        for doc_type, samples in type_groups.items():
            self.save_samples(samples, doc_type)
        
        logger.info(f"[COMPLETE] Scraped {len(all_samples)} documentation samples")
        return all_samples


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Documentation Scraper')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--github-repos', type=str, nargs='+', help='GitHub repository URLs')
    parser.add_argument('--api-urls', type=str, nargs='+', help='API documentation URLs')
    parser.add_argument('--blog-urls', type=str, nargs='+', help='Technical blog URLs')
    parser.add_argument('--delay', type=float, default=2.0, help='Delay between requests (seconds)')
    parser.add_argument('--max-docs', type=int, default=100, help='Max documents per source')
    
    args = parser.parse_args()
    
    scraper = DocumentationScraper(
        output_dir=args.output,
        delay_per_request=args.delay,
        max_docs_per_source=args.max_docs
    )
    
    scraper.run_scraping(
        github_repos=args.github_repos,
        api_urls=args.api_urls,
        blog_urls=args.blog_urls
    )


if __name__ == '__main__':
    main()

