#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
オープンアクセス科学論文収集クローラー

複数のオープンアクセスリポジトリから科学論文を収集

対象リポジトリ:
- PubMed Central (PMC) - 生物医学・生命科学
- DOAJ (Directory of Open Access Journals)
- Europe PMC - 欧州のオープンアクセス論文
- CORE - オープンアクセス論文アグリゲーター
- Zenodo - 研究データ・論文リポジトリ
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse
import re
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Chromeヘッダー生成ユーティリティ
try:
    from .chrome_headers import get_chrome_headers, get_chrome_api_headers, get_chrome_user_agent
except ImportError:
    # フォールバック: 直接定義
    def get_chrome_headers(referer: Optional[str] = None) -> Dict[str, str]:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
        }
        if referer:
            headers['Referer'] = referer
        return headers
    
    def get_chrome_api_headers() -> Dict[str, str]:
        return {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
        }
    
    def get_chrome_user_agent() -> str:
        return 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class OpenAccessPapersCrawler:
    """オープンアクセス科学論文収集クローラー"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: クローラー設定
        """
        self.config = config
        self.repositories = config.get('repositories', ['pmc', 'doaj', 'europe_pmc', 'core', 'zenodo'])
        self.max_papers = config.get('max_papers', 100000)
        self.delay = config.get('delay', 1.0)
        self.timeout = config.get('timeout', 15)
        # Chromeに偽装
        self.user_agent = config.get('user_agent', get_chrome_user_agent())
        self.chrome_headers = get_chrome_headers()
        self.chrome_api_headers = get_chrome_api_headers()
        
        self.samples: List[Dict] = []
        
        # リポジトリ別設定
        self.repo_configs = {
            'pmc': {
                'base_url': 'https://www.ncbi.nlm.nih.gov/pmc/',
                'api_url': 'https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi',
                'domain': 'biomedical'
            },
            'doaj': {
                'base_url': 'https://doaj.org/',
                'api_url': 'https://doaj.org/api/v2/',
                'domain': 'general'
            },
            'europe_pmc': {
                'base_url': 'https://europepmc.org/',
                'api_url': 'https://www.ebi.ac.uk/europepmc/webservices/rest/',
                'domain': 'biomedical'
            },
            'core': {
                'base_url': 'https://core.ac.uk/',
                'api_url': 'https://api.core.ac.uk/v3/',
                'domain': 'general'
            },
            'zenodo': {
                'base_url': 'https://zenodo.org/',
                'api_url': 'https://zenodo.org/api/',
                'domain': 'general'
            }
        }
    
    def _fetch_api(self, url: str, params: Dict = None) -> Optional[Dict]:
        """APIを呼び出し"""
        try:
            time.sleep(self.delay)
            # Chrome APIヘッダーを使用
            response = requests.get(
                url,
                params=params,
                timeout=self.timeout,
                headers=self.chrome_api_headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.debug(f"API call failed: {url}, {e}")
            return None
    
    def crawl_pmc(self) -> List[Dict]:
        """PubMed Centralから論文を収集"""
        logger.info("[PMC] Crawling PubMed Central...")
        samples = []
        
        try:
            # PMC OAI APIを使用（XMLパーサー実装）
            try:
                import xml.etree.ElementTree as ET
            except ImportError:
                logger.error("[PMC] XML parser not available")
                return samples
            
            api_url = self.repo_configs['pmc']['api_url']
            params = {
                'verb': 'ListRecords',
                'metadataPrefix': 'pmc',
                'set': 'open',
                'from': '2020-01-01'
            }
            
            # OAI API呼び出し（Chromeヘッダー使用）
            response = requests.get(
                api_url,
                params=params,
                headers=self.chrome_api_headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                # XML解析
                root = ET.fromstring(response.content)
                
                # OAI名前空間
                ns = {'oai': 'http://www.openarchives.org/OAI/2.0/',
                      'pmc': 'http://www.ncbi.nlm.nih.gov/pmc/oai/pmc/'}
                
                records = root.findall('.//oai:record', ns)
                
                for record in records[:500]:  # 最大500件
                    try:
                        header = record.find('oai:header', ns)
                        metadata = record.find('oai:metadata', ns)
                        
                        if header is None or metadata is None:
                            continue
                        
                        identifier = header.find('oai:identifier', ns)
                        if identifier is None or identifier.text is None:
                            continue
                        
                        # メタデータ抽出
                        article = metadata.find('.//pmc:article', ns)
                        if article is None:
                            continue
                        
                        title_elem = article.find('.//pmc:article-title', ns)
                        abstract_elem = article.find('.//pmc:abstract', ns)
                        
                        title = title_elem.text if title_elem is not None and title_elem.text else ''
                        abstract = ''
                        
                        if abstract_elem is not None:
                            abstract_parts = []
                            for p in abstract_elem.findall('.//pmc:p', ns):
                                if p.text:
                                    abstract_parts.append(p.text)
                            abstract = ' '.join(abstract_parts)
                        
                        if not title:
                            continue
                        
                        content = title
                        if abstract:
                            content += f"\n\n{abstract}"
                        
                        samples.append({
                            'url': identifier.text,
                            'content': content,
                            'title': title,
                            'domain': 'scientific_paper',
                            'subdomain': 'pmc',
                            'language': 'en',
                            'timestamp': datetime.now().isoformat(),
                            'type': 'open_access_paper',
                            'repository': 'pmc'
                        })
                        
                        time.sleep(self.delay)
                        
                        if len(samples) >= 1000:  # 最大1000件
                            break
                    
                    except Exception as e:
                        logger.debug(f"Failed to process PMC record: {e}")
                        continue
            
        except Exception as e:
            logger.error(f"[PMC] Failed: {e}")
        
        logger.info(f"[PMC] Collected {len(samples)} papers")
        return samples
    
    def crawl_doaj(self) -> List[Dict]:
        """DOAJから論文を収集"""
        logger.info("[DOAJ] Crawling DOAJ...")
        samples = []
        
        try:
            api_url = self.repo_configs['doaj']['api_url']
            params = {
                'pageSize': 100,
                'page': 1
            }
            
            # DOAJ API呼び出し
            data = self._fetch_api(f"{api_url}search/articles", params)
            if data and 'results' in data:
                for article in data['results'][:1000]:  # 最大1000件
                    try:
                        title = article.get('bibjson', {}).get('title', '')
                        abstract = article.get('bibjson', {}).get('abstract', '')
                        
                        if not title or not abstract:
                            continue
                        
                        samples.append({
                            'url': article.get('id', ''),
                            'content': f"{title}\n\n{abstract}",
                            'title': title,
                            'domain': 'scientific_paper',
                            'subdomain': 'doaj',
                            'language': 'en',
                            'timestamp': datetime.now().isoformat(),
                            'type': 'open_access_paper',
                            'repository': 'doaj'
                        })
                    except Exception as e:
                        logger.debug(f"Failed to process DOAJ article: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"[DOAJ] Failed: {e}")
        
        logger.info(f"[DOAJ] Collected {len(samples)} papers")
        return samples
    
    def crawl_europe_pmc(self) -> List[Dict]:
        """Europe PMCから論文を収集"""
        logger.info("[EUROPE PMC] Crawling Europe PMC...")
        samples = []
        
        try:
            api_url = self.repo_configs['europe_pmc']['api_url']
            params = {
                'query': 'OPEN_ACCESS:Y',
                'format': 'json',
                'pageSize': 100,
                'page': 1
            }
            
            # Europe PMC API呼び出し
            data = self._fetch_api(f"{api_url}search", params)
            if data and 'resultList' in data:
                for result in data['resultList']['result'][:1000]:  # 最大1000件
                    try:
                        title = result.get('title', '')
                        abstract = result.get('abstractText', '')
                        
                        if not title or not abstract:
                            continue
                        
                        samples.append({
                            'url': result.get('id', ''),
                            'content': f"{title}\n\n{abstract}",
                            'title': title,
                            'domain': 'scientific_paper',
                            'subdomain': 'europe_pmc',
                            'language': 'en',
                            'timestamp': datetime.now().isoformat(),
                            'type': 'open_access_paper',
                            'repository': 'europe_pmc'
                        })
                    except Exception as e:
                        logger.debug(f"Failed to process Europe PMC article: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"[EUROPE PMC] Failed: {e}")
        
        logger.info(f"[EUROPE PMC] Collected {len(samples)} papers")
        return samples
    
    def crawl_core(self) -> List[Dict]:
        """COREから論文を収集"""
        logger.info("[CORE] Crawling CORE...")
        samples = []
        
        try:
            api_url = self.repo_configs['core']['api_url']
            
            # CORE APIは認証が必要（APIキーを環境変数から取得）
            api_key = os.environ.get('CORE_API_KEY')
            if not api_key:
                logger.warning("[CORE] CORE_API_KEY not set. Using public search endpoint.")
                # 公開検索エンドポイントを使用
                search_url = f"{api_url}search"
                params = {
                    'q': 'open access',
                    'page': 1,
                    'pageSize': 100
                }
            else:
                # 認証付きエンドポイント
                search_url = f"{api_url}search"
                params = {
                    'q': 'open access',
                    'page': 1,
                    'pageSize': 100
                }
                headers = {'Authorization': f'Bearer {api_key}'}
            
            # API呼び出し
            for page in range(1, 11):  # 最大10ページ
                params['page'] = page
                
                # CORE API呼び出し（Chromeヘッダー使用）
                if not api_key:
                    headers = self.chrome_api_headers.copy()
                response = requests.get(
                    search_url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout
                )
                
                if response.status_code != 200:
                    if response.status_code == 401:
                        logger.warning("[CORE] Authentication failed. Set CORE_API_KEY environment variable.")
                    break
                
                data = response.json()
                results = data.get('results', [])
                
                if not results:
                    break
                
                for result in results:
                    try:
                        title = result.get('title', '')
                        abstract = result.get('abstract', '')
                        
                        if not title:
                            continue
                        
                        content = title
                        if abstract:
                            content += f"\n\n{abstract}"
                        
                        samples.append({
                            'url': result.get('downloadUrl', result.get('url', '')),
                            'content': content,
                            'title': title,
                            'domain': 'scientific_paper',
                            'subdomain': 'core',
                            'language': 'en',
                            'timestamp': datetime.now().isoformat(),
                            'type': 'open_access_paper',
                            'repository': 'core'
                        })
                        
                        if len(samples) >= 1000:  # 最大1000件
                            break
                    
                    except Exception as e:
                        logger.debug(f"Failed to process CORE result: {e}")
                        continue
                
                if len(samples) >= 1000:
                    break
                
                time.sleep(self.delay)
        
        except Exception as e:
            logger.error(f"[CORE] Failed: {e}")
        
        logger.info(f"[CORE] Collected {len(samples)} papers")
        return samples
    
    def crawl_zenodo(self) -> List[Dict]:
        """Zenodoから論文を収集"""
        logger.info("[ZENODO] Crawling Zenodo...")
        samples = []
        
        try:
            api_url = self.repo_configs['zenodo']['api_url']
            params = {
                'q': 'open access',
                'type': 'publication',
                'size': 100,
                'page': 1
            }
            
            # Zenodo API呼び出し
            data = self._fetch_api(f"{api_url}records", params)
            if data and 'hits' in data and 'hits' in data['hits']:
                for hit in data['hits']['hits'][:1000]:  # 最大1000件
                    try:
                        metadata = hit.get('metadata', {})
                        title = metadata.get('title', '')
                        description = metadata.get('description', '')
                        
                        if not title:
                            continue
                        
                        content = title
                        if description:
                            content += f"\n\n{description}"
                        
                        samples.append({
                            'url': hit.get('links', {}).get('html', ''),
                            'content': content,
                            'title': title,
                            'domain': 'scientific_paper',
                            'subdomain': 'zenodo',
                            'language': 'en',
                            'timestamp': datetime.now().isoformat(),
                            'type': 'open_access_paper',
                            'repository': 'zenodo'
                        })
                    except Exception as e:
                        logger.debug(f"Failed to process Zenodo record: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"[ZENODO] Failed: {e}")
        
        logger.info(f"[ZENODO] Collected {len(samples)} papers")
        return samples
    
    def crawl(self) -> List[Dict]:
        """全リポジトリから論文を収集"""
        logger.info("="*80)
        logger.info("Open Access Papers Crawler")
        logger.info("="*80)
        
        all_samples = []
        papers_per_repo = self.max_papers // len(self.repositories)
        
        for repo in self.repositories:
            if repo == 'pmc':
                samples = self.crawl_pmc()
            elif repo == 'doaj':
                samples = self.crawl_doaj()
            elif repo == 'europe_pmc':
                samples = self.crawl_europe_pmc()
            elif repo == 'core':
                samples = self.crawl_core()
            elif repo == 'zenodo':
                samples = self.crawl_zenodo()
            else:
                logger.warning(f"Unknown repository: {repo}")
                continue
            
            all_samples.extend(samples)
            
            if len(all_samples) >= self.max_papers:
                break
        
        self.samples = all_samples[:self.max_papers]
        logger.info(f"[TOTAL] Collected {len(self.samples)} papers")
        return self.samples
    
    def save(self, output_path: Path):
        """結果を保存"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in self.samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"[SAVE] Saved {len(self.samples)} papers to {output_path}")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Open Access Papers Crawler')
    parser.add_argument('--output', type=str, default='D:/webdataset/processed/open_access_papers.jsonl')
    parser.add_argument('--repositories', nargs='+', 
                       default=['pmc', 'doaj', 'europe_pmc', 'core', 'zenodo'])
    parser.add_argument('--max-papers', type=int, default=100000)
    parser.add_argument('--delay', type=float, default=1.0)
    
    args = parser.parse_args()
    
    config = {
        'repositories': args.repositories,
        'max_papers': args.max_papers,
        'delay': args.delay,
        'timeout': 15,
        'user_agent': 'SO8T-DataCollector/1.0 (Research Purpose)'
    }
    
    crawler = OpenAccessPapersCrawler(config)
    crawler.crawl()
    crawler.save(Path(args.output))


if __name__ == '__main__':
    main()

