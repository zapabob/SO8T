#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHubリポジトリ検索スクレイピングスクリプト

GitHub APIを使用してスター数の多いリポジトリを検索し、
README、ドキュメンテーション、サンプルコードを抽出します。

Usage:
    python scripts/data/github_repository_scraper.py --output D:\webdataset\processed
"""

import sys
import json
import logging
import argparse
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urlparse

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/github_repository_scraper.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GitHubRepositoryScraper:
    """GitHubリポジトリ検索スクレイパー"""
    
    def __init__(
        self,
        output_dir: Path,
        github_token: Optional[str] = None,
        max_repos_per_query: int = 100,
        min_stars: int = 100,
        rate_limit_delay: float = 1.0
    ):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリ
            github_token: GitHub APIトークン（Noneの場合は環境変数から読み込み、それもない場合は認証なし）
            max_repos_per_query: クエリあたりの最大リポジトリ数
            min_stars: 最小スター数
            rate_limit_delay: レート制限対策の遅延（秒）
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # GitHub APIトークンを環境変数から読み込む（引数で指定されていない場合）
        if github_token is None:
            try:
                from scripts.utils.env_loader import get_env
                github_token = get_env('GITHUB_API_TOKEN')
            except ImportError:
                logger.warning("[ENV] Failed to import env_loader, using provided token or None")
        
        self.github_token = github_token
        self.max_repos_per_query = max_repos_per_query
        self.min_stars = min_stars
        self.rate_limit_delay = rate_limit_delay
        
        # APIヘッダー
        self.headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'SO8T-GitHub-Scraper'
        }
        if self.github_token:
            self.headers['Authorization'] = f'token {self.github_token}'
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.all_samples: List[Dict] = []
        
        logger.info("="*80)
        logger.info("GitHub Repository Scraper Initialized")
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Min stars: {self.min_stars}")
        logger.info(f"Max repos per query: {self.max_repos_per_query}")
    
    def search_repositories(
        self,
        query: str,
        language: Optional[str] = None,
        sort: str = 'stars',
        order: str = 'desc'
    ) -> List[Dict]:
        """
        GitHubリポジトリを検索
        
        Args:
            query: 検索クエリ
            language: プログラミング言語（オプション）
            sort: ソート方法（stars, forks, updated, etc.）
            order: ソート順（desc, asc）
        
        Returns:
            リポジトリ情報のリスト
        """
        url = "https://api.github.com/search/repositories"
        
        # クエリを構築
        search_query = query
        if language:
            search_query = f"{query} language:{language}"
        
        params = {
            'q': search_query,
            'sort': sort,
            'order': order,
            'per_page': min(self.max_repos_per_query, 100)  # GitHub APIの最大値は100
        }
        
        try:
            logger.info(f"[SEARCH] Searching repositories: {search_query}")
            response = self.session.get(url, params=params, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                repos = data.get('items', [])
                
                # 最小スター数でフィルタリング
                filtered_repos = [
                    repo for repo in repos 
                    if repo.get('stargazers_count', 0) >= self.min_stars
                ]
                
                logger.info(f"[OK] Found {len(filtered_repos)} repositories (filtered from {len(repos)})")
                return filtered_repos
            elif response.status_code == 403:
                logger.warning(f"[RATE_LIMIT] Rate limit exceeded. Waiting {self.rate_limit_delay * 60} seconds...")
                time.sleep(self.rate_limit_delay * 60)
                return []
            else:
                logger.error(f"[ERROR] GitHub API error: {response.status_code} - {response.text}")
                return []
        
        except Exception as e:
            logger.error(f"[ERROR] Failed to search repositories: {e}")
            return []
    
    def get_repository_content(self, repo: Dict) -> Dict:
        """
        リポジトリのコンテンツを取得（README、ドキュメンテーション、サンプルコード）
        
        Args:
            repo: リポジトリ情報
        
        Returns:
            コンテンツ情報
        """
        repo_name = repo.get('full_name', '')
        repo_url = repo.get('html_url', '')
        default_branch = repo.get('default_branch', 'main')
        
        content = {
            'repo_name': repo_name,
            'repo_url': repo_url,
            'description': repo.get('description', ''),
            'language': repo.get('language', ''),
            'stars': repo.get('stargazers_count', 0),
            'forks': repo.get('forks_count', 0),
            'readme': '',
            'documentation': [],
            'sample_code': []
        }
        
        try:
            # READMEを取得
            readme_url = f"https://api.github.com/repos/{repo_name}/readme"
            response = self.session.get(readme_url, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                readme_data = response.json()
                import base64
                readme_content = base64.b64decode(readme_data.get('content', '')).decode('utf-8', errors='ignore')
                content['readme'] = readme_content
                logger.debug(f"[OK] Retrieved README for {repo_name}")
            
            # ドキュメンテーションディレクトリを検索
            docs_dirs = ['docs', 'documentation', 'doc', 'wiki']
            for docs_dir in docs_dirs:
                docs_url = f"https://api.github.com/repos/{repo_name}/contents/{docs_dir}"
                response = self.session.get(docs_url, headers=self.headers, timeout=30)
                
                if response.status_code == 200:
                    docs_files = response.json()
                    for doc_file in docs_files[:10]:  # 最大10ファイル
                        if doc_file.get('type') == 'file' and doc_file.get('name', '').endswith(('.md', '.txt', '.rst')):
                            file_url = doc_file.get('download_url', '')
                            if file_url:
                                try:
                                    file_response = self.session.get(file_url, timeout=30)
                                    if file_response.status_code == 200:
                                        doc_content = file_response.text
                                        content['documentation'].append({
                                            'file': doc_file.get('name', ''),
                                            'path': doc_file.get('path', ''),
                                            'content': doc_content[:10000]  # 最大10000文字
                                        })
                                except Exception as e:
                                    logger.debug(f"[DEBUG] Failed to fetch doc file {doc_file.get('name')}: {e}")
            
            # サンプルコードを検索（examples, samples, demoディレクトリ）
            example_dirs = ['examples', 'samples', 'demo', 'demos', 'test', 'tests']
            for example_dir in example_dirs:
                example_url = f"https://api.github.com/repos/{repo_name}/contents/{example_dir}"
                response = self.session.get(example_url, headers=self.headers, timeout=30)
                
                if response.status_code == 200:
                    example_files = response.json()
                    for example_file in example_files[:20]:  # 最大20ファイル
                        if example_file.get('type') == 'file':
                            file_name = example_file.get('name', '')
                            # コードファイルの拡張子をチェック
                            code_extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.rs', '.go', '.rb', '.php', '.sql']
                            if any(file_name.endswith(ext) for ext in code_extensions):
                                file_url = example_file.get('download_url', '')
                                if file_url:
                                    try:
                                        file_response = self.session.get(file_url, timeout=30)
                                        if file_response.status_code == 200:
                                            code_content = file_response.text
                                            content['sample_code'].append({
                                                'file': file_name,
                                                'path': example_file.get('path', ''),
                                                'content': code_content[:5000]  # 最大5000文字
                                            })
                                    except Exception as e:
                                        logger.debug(f"[DEBUG] Failed to fetch code file {file_name}: {e}")
        
        except Exception as e:
            logger.warning(f"[WARNING] Failed to get repository content for {repo_name}: {e}")
        
        return content
    
    def scrape_repositories(
        self,
        queries: List[str],
        languages: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        複数のクエリでリポジトリをスクレイピング
        
        Args:
            queries: 検索クエリのリスト
            languages: プログラミング言語のリスト（オプション）
        
        Returns:
            サンプルのリスト
        """
        self.session = requests.Session()
        samples = []
        
        for query in queries:
            logger.info(f"[QUERY] Processing query: {query}")
            
            if languages:
                for language in languages:
                    repos = self.search_repositories(query, language=language)
                    for repo in repos:
                        content = self.get_repository_content(repo)
                        
                        # サンプルを作成
                        sample = {
                            'text': content.get('readme', '') + '\n\n' + content.get('description', ''),
                            'url': content.get('repo_url', ''),
                            'domain': 'github.com',
                            'category': 'programming',
                            'language': content.get('language', ''),
                            'metadata': {
                                'repo_name': content.get('repo_name', ''),
                                'stars': content.get('stars', 0),
                                'forks': content.get('forks', 0),
                                'documentation_count': len(content.get('documentation', [])),
                                'sample_code_count': len(content.get('sample_code', []))
                            },
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        samples.append(sample)
                        
                        # レート制限対策
                        time.sleep(self.rate_limit_delay)
            else:
                repos = self.search_repositories(query)
                for repo in repos:
                    content = self.get_repository_content(repo)
                    
                    # サンプルを作成
                    readme = content.get('readme') or ''
                    description = content.get('description') or ''
                    sample = {
                        'text': readme + '\n\n' + description,
                        'url': content.get('repo_url', ''),
                        'domain': 'github.com',
                        'category': 'programming',
                        'language': content.get('language', ''),
                        'metadata': {
                            'repo_name': content.get('repo_name', ''),
                            'stars': content.get('stars', 0),
                            'forks': content.get('forks', 0),
                            'documentation_count': len(content.get('documentation', [])),
                            'sample_code_count': len(content.get('sample_code', []))
                        },
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    samples.append(sample)
                    
                    # レート制限対策
                    time.sleep(self.rate_limit_delay)
        
        self.session.close()
        return samples
    
    def save_samples(self, samples: List[Dict]):
        """サンプルを保存"""
        output_file = self.output_dir / f"github_repositories_{self.session_id}.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"[SAVE] Saved {len(samples)} samples to {output_file}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='GitHub Repository Scraper')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--github-token', type=str, default=None, help='GitHub API token')
    parser.add_argument('--max-repos', type=int, default=100, help='Max repositories per query')
    parser.add_argument('--min-stars', type=int, default=100, help='Minimum stars')
    parser.add_argument('--queries', type=str, nargs='+', default=['best practices', 'tutorial', 'example'], help='Search queries')
    parser.add_argument('--languages', type=str, nargs='+', default=None, help='Programming languages')
    
    args = parser.parse_args()
    
    scraper = GitHubRepositoryScraper(
        output_dir=args.output,
        github_token=args.github_token,
        max_repos_per_query=args.max_repos,
        min_stars=args.min_stars
    )
    
    samples = scraper.scrape_repositories(
        queries=args.queries,
        languages=args.languages
    )
    
    scraper.save_samples(samples)
    
    logger.info(f"[COMPLETE] Scraped {len(samples)} samples from GitHub")


if __name__ == '__main__':
    main()

