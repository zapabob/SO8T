#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
専用データソースクローラーモジュール

各データソース（4web版官報、e-Gov、zenn、Qiita）専用のクローラー実装
"""

import os
import sys
import json
import time
import logging
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse, parse_qs
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

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


class BaseSpecializedCrawler:
    """専用クローラーのベースクラス"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: クローラー設定
        """
        self.config = config
        self.base_url = config.get('base_url', '')
        self.domain = config.get('domain', '')
        self.language = config.get('language', 'ja')
        self.delay = config.get('delay', 1.0)
        self.timeout = config.get('timeout', 15)
        self.max_pages = config.get('max_pages', 1000)
        # Chromeに偽装
        self.user_agent = config.get('user_agent', get_chrome_user_agent())
        self.chrome_headers = get_chrome_headers()
        
        self.visited_urls: Set[str] = set()
        self.samples: List[Dict] = []
        
        # robots.txt確認
        self.robots_parser = None
        self._load_robots_txt()
    
    def _load_robots_txt(self):
        """robots.txt読み込み"""
        try:
            robots_url = urljoin(self.base_url, '/robots.txt')
            self.robots_parser = RobotFileParser()
            self.robots_parser.set_url(robots_url)
            self.robots_parser.read()
            logger.info(f"[ROBOTS] Loaded robots.txt from {robots_url}")
        except Exception as e:
            logger.warning(f"[ROBOTS] Failed to load robots.txt: {e}")
            self.robots_parser = None
    
    def can_fetch(self, url: str) -> bool:
        """robots.txtで許可されているか確認"""
        if not self.robots_parser:
            return True
        
        try:
            return self.robots_parser.can_fetch(self.user_agent, url)
        except Exception:
            return True
    
    def _fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """ページ取得"""
        if not self.can_fetch(url):
            logger.debug(f"[SKIP] robots.txt disallows: {url}")
            return None
        
        if url in self.visited_urls:
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
            return BeautifulSoup(response.content, 'lxml')
        
        except Exception as e:
            logger.debug(f"Failed to fetch {url}: {e}")
            return None
    
    def extract_text(self, soup: BeautifulSoup) -> str:
        """テキスト抽出"""
        # スクリプト・スタイル・ナビゲーション要素を削除
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()
        
        text = soup.get_text(separator='\n', strip=True)
        text = re.sub(r'\n\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()
    
    def create_sample(self, text: str, url: str, metadata: Dict = None) -> Dict:
        """サンプル作成"""
        if len(text) < 200:
            return None
        
        sample = {
            "text": text,
            "url": url,
            "domain": self.domain,
            "language": self.language,
            "source": self.__class__.__name__,
            "crawled_at": datetime.now().isoformat(),
            "text_length": len(text)
        }
        
        if metadata:
            sample.update(metadata)
        
        return sample
    
    def crawl(self) -> List[Dict]:
        """クロール実行（サブクラスで実装）"""
        raise NotImplementedError


class Kanpou4WebCrawler(BaseSpecializedCrawler):
    """4web版官報専用クローラー"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.base_url = "https://kanpou.4web.jp/"
    
    def crawl(self) -> List[Dict]:
        """官報クロール（日付ベース）"""
        logger.info(f"[KANPOU] Starting crawl from {self.base_url}")
        
        # トップページから開始
        soup = self._fetch_page(self.base_url)
        if soup:
            # リンクを取得してクロール
            links = soup.find_all('a', href=True)
            for link in links[:self.max_pages]:
                href = link.get('href', '')
                if not href:
                    continue
                
                url = urljoin(self.base_url, href)
                if url in self.visited_urls:
                    continue
                
                soup_page = self._fetch_page(url)
                if soup_page:
                    text = self.extract_text(soup_page)
                    sample = self.create_sample(text, url, {
                        'type': 'official_gazette'
                    })
                    if sample:
                        self.samples.append(sample)
                
                if len(self.samples) >= self.max_pages:
                    break
        
        logger.info(f"[KANPOU] Collected {len(self.samples)} samples")
        return self.samples
    
    def _extract_pdf_text(self, pdf_url: str) -> Optional[str]:
        """PDFテキスト抽出（簡易版）"""
        # 実際の実装ではPyPDF2やpdfplumberを使用
        # ここでは簡易実装
        try:
            response = requests.get(pdf_url, timeout=self.timeout)
            # PDF解析は別途実装が必要
            return None
        except Exception:
            return None


class EGovCrawler(BaseSpecializedCrawler):
    """e-Gov専用クローラー"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.base_url = "https://www.e-gov.go.jp/"
    
    def crawl(self) -> List[Dict]:
        """e-Govクロール（法令・告示・通知）"""
        logger.info(f"[EGOV] Starting crawl from {self.base_url}")
        
        # トップページから開始
        soup = self._fetch_page(self.base_url)
        if soup:
            # リンクを取得してクロール
            links = soup.find_all('a', href=True)
            for link in links[:self.max_pages * 2]:  # より多くのリンクを試行
                href = link.get('href', '')
                if not href:
                    continue
                
                url = urljoin(self.base_url, href)
                if url in self.visited_urls:
                    continue
                
                # 同一ドメインのみ
                if urlparse(url).netloc not in ['www.e-gov.go.jp', 'elaws.e-gov.go.jp']:
                    continue
                
                soup_page = self._fetch_page(url)
                if soup_page:
                    text = self.extract_text(soup_page)
                    sample = self.create_sample(text, url, {
                        'type': 'law'
                    })
                    if sample:
                        self.samples.append(sample)
                
                if len(self.samples) >= self.max_pages:
                    break
        
        logger.info(f"[EGOV] Collected {len(self.samples)} samples")
        return self.samples


class ZennCrawler(BaseSpecializedCrawler):
    """zenn専用クローラー"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.base_url = "https://zenn.dev/"
        self.api_enabled = config.get('api_enabled', False)
    
    def crawl(self) -> List[Dict]:
        """zennクロール（記事一覧）"""
        logger.info(f"[ZENN] Starting crawl from {self.base_url}")
        
        # API利用可能な場合はAPIを使用
        if self.api_enabled:
            return self._crawl_via_api()
        
        # 通常のHTMLクロール
        articles_url = f"{self.base_url}articles"
        soup = self._fetch_page(articles_url)
        
        if soup:
            # 記事リンク取得（より柔軟なパターン）
            article_links = soup.find_all('a', href=True)
            
            for link in article_links:
                href = link.get('href', '')
                if not href or '/articles/' not in href:
                    continue
                
                article_url = urljoin(self.base_url, href)
                if article_url in self.visited_urls:
                    continue
                
                soup_article = self._fetch_page(article_url)
                if soup_article:
                    text = self.extract_text(soup_article)
                    sample = self.create_sample(text, article_url, {
                        'type': 'article',
                        'platform': 'zenn'
                    })
                    if sample:
                        self.samples.append(sample)
                
                if len(self.samples) >= self.max_pages:
                    break
        
        logger.info(f"[ZENN] Collected {len(self.samples)} samples")
        return self.samples
    
    def _crawl_via_api(self) -> List[Dict]:
        """API経由でクロール"""
        # zenn APIは非公開のため、HTMLクロールにフォールバック
        logger.info("[ZENN] API not available, using HTML crawl")
        return self.crawl()


class QiitaCrawler(BaseSpecializedCrawler):
    """Qiita専用クローラー"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.base_url = "https://qiita.com/"
        self.api_enabled = config.get('api_enabled', False)
        self.api_token = config.get('api_token', None)
    
    def crawl(self) -> List[Dict]:
        """Qiitaクロール（記事一覧）"""
        logger.info(f"[QIITA] Starting crawl from {self.base_url}")
        
        # API利用可能な場合はAPIを使用
        if self.api_enabled and self.api_token:
            return self._crawl_via_api()
        
        # 通常のHTMLクロール
        articles_url = f"{self.base_url}articles"
        soup = self._fetch_page(articles_url)
        
        if soup:
            # 記事リンク取得（より柔軟なパターン）
            article_links = soup.find_all('a', href=True)
            
            for link in article_links:
                href = link.get('href', '')
                if not href or '/items/' not in href:
                    continue
                
                article_url = urljoin(self.base_url, href)
                if article_url in self.visited_urls:
                    continue
                
                soup_article = self._fetch_page(article_url)
                if soup_article:
                    text = self.extract_text(soup_article)
                    sample = self.create_sample(text, article_url, {
                        'type': 'article',
                        'platform': 'qiita'
                    })
                    if sample:
                        self.samples.append(sample)
                
                if len(self.samples) >= self.max_pages:
                    break
        
        logger.info(f"[QIITA] Collected {len(self.samples)} samples")
        return self.samples
    
    def _crawl_via_api(self) -> List[Dict]:
        """API経由でクロール"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_token}',
                'User-Agent': self.user_agent
            }
            
            # Qiita API v2
            api_url = "https://qiita.com/api/v2/items"
            params = {
                'page': 1,
                'per_page': 100
            }
            
            page = 1
            while len(self.samples) < self.max_pages:
                params['page'] = page
                response = requests.get(api_url, headers=headers, params=params, timeout=self.timeout)
                
                if response.status_code != 200:
                    break
                
                items = response.json()
                if not items:
                    break
                
                for item in items:
                    text = item.get('body', '')
                    url = item.get('url', '')
                    
                    sample = self.create_sample(text, url, {
                        'type': 'article',
                        'platform': 'qiita',
                        'article_id': item.get('id'),
                        'title': item.get('title'),
                        'tags': item.get('tags', [])
                    })
                    if sample:
                        self.samples.append(sample)
                
                page += 1
                time.sleep(self.delay)
            
            logger.info(f"[QIITA] Collected {len(self.samples)} samples via API")
            return self.samples
        
        except Exception as e:
            logger.warning(f"[QIITA] API crawl failed: {e}, falling back to HTML")
            return self.crawl()


class WikipediaCrawler(BaseSpecializedCrawler):
    """ウィキペディア専用クローラー"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.base_url = config.get('base_url', 'https://ja.wikipedia.org/wiki/')
    
    def crawl(self) -> List[Dict]:
        """ウィキペディアクロール"""
        logger.info(f"[WIKIPEDIA] Starting crawl from {self.base_url}")
        
        # トップページから開始
        soup = self._fetch_page(self.base_url)
        if soup:
            # 記事リンクを取得
            links = soup.find_all('a', href=True)
            
            for link in links:
                href = link.get('href', '')
                if not href or not href.startswith('/wiki/'):
                    continue
                
                # 特殊ページをスキップ
                if any(skip in href for skip in ['/wiki/Special:', '/wiki/File:', '/wiki/Category:', '/wiki/Template:']):
                    continue
                
                article_url = urljoin(self.base_url, href)
                if article_url in self.visited_urls:
                    continue
                
                soup_article = self._fetch_page(article_url)
                if soup_article:
                    text = self.extract_text(soup_article)
                    sample = self.create_sample(text, article_url, {
                        'type': 'wikipedia_article',
                        'platform': 'wikipedia'
                    })
                    if sample:
                        self.samples.append(sample)
                
                if len(self.samples) >= self.max_pages:
                    break
        
        logger.info(f"[WIKIPEDIA] Collected {len(self.samples)} samples")
        return self.samples

