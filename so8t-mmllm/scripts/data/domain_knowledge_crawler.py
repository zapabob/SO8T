#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ドメイン別知識サイト収集クローラー

日本語・英語のドメイン別知識が集積するサイトからデータを収集

日本語サイト:
- コトバンク (辞書・百科事典)
- Weblio (辞書・用語集)
- 日本大百科全書（ニッポニカ）
- ブリタニカ国際大百科事典
- 学術機関リポジトリ（各大学）
- J-STAGE (日本科学技術情報発信・流通総合システム)
- CiNii (学術情報ナビゲータ)

英語サイト:
- Wikipedia (既存統合)
- Britannica (百科事典)
- Khan Academy (教育コンテンツ)
- MIT OpenCourseWare (講義資料)
- Stanford Encyclopedia of Philosophy
- Internet Encyclopedia of Philosophy
- Project Gutenberg (古典文献)
"""

import os
import sys
import json
import time
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Optional
from urllib.parse import urljoin, urlparse, quote, urlencode
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


class DomainKnowledgeCrawler:
    """ドメイン別知識サイト収集クローラー"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: クローラー設定
        """
        self.config = config
        self.japanese_sites = config.get('japanese_sites', ['kotobank', 'weblio', 'jstage', 'cinii'])
        self.english_sites = config.get('english_sites', ['britannica', 'khan_academy', 'mit_ocw'])
        self.max_samples_per_site = config.get('max_samples_per_site', 50000)
        self.delay = config.get('delay', 1.0)
        self.timeout = config.get('timeout', 15)
        # Chromeに偽装
        self.user_agent = config.get('user_agent', get_chrome_user_agent())
        self.chrome_headers = get_chrome_headers()
        
        self.visited_urls: Set[str] = set()
        self.samples: List[Dict] = []
        
        # サイト別設定
        self.site_configs = {
            # 日本語サイト
            'kotobank': {
                'base_url': 'https://kotobank.jp/',
                'domain': 'encyclopedia',
                'language': 'ja'
            },
            'weblio': {
                'base_url': 'https://www.weblio.jp/',
                'domain': 'dictionary',
                'language': 'ja'
            },
            'jstage': {
                'base_url': 'https://www.jstage.jst.go.jp/',
                'domain': 'academic',
                'language': 'ja'
            },
            'cinii': {
                'base_url': 'https://ci.nii.ac.jp/',
                'domain': 'academic',
                'language': 'ja'
            },
            # 英語サイト
            'britannica': {
                'base_url': 'https://www.britannica.com/',
                'domain': 'encyclopedia',
                'language': 'en'
            },
            'khan_academy': {
                'base_url': 'https://www.khanacademy.org/',
                'domain': 'education',
                'language': 'en'
            },
            'mit_ocw': {
                'base_url': 'https://ocw.mit.edu/',
                'domain': 'education',
                'language': 'en'
            }
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
            return True
    
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
        
        # メインコンテンツを抽出
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile('content|main|article'))
        
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
        else:
            text = soup.get_text(separator='\n', strip=True)
        
        text = re.sub(r'\n\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()
    
    def crawl_kotobank(self) -> List[Dict]:
        """コトバンクからデータを収集"""
        logger.info("[KOTOBANK] Crawling コトバンク...")
        samples = []
        
        base_url = self.site_configs['kotobank']['base_url']
        
        try:
            # コトバンクの検索機能を使用
            # 人気キーワードやカテゴリから記事を取得
            search_keywords = [
                '人工知能', '機械学習', '深層学習', '自然言語処理',
                'プログラミング', 'アルゴリズム', 'データ構造',
                'コンピュータ', 'ソフトウェア', 'ハードウェア',
                'ネットワーク', 'セキュリティ', 'データベース',
                'オペレーティングシステム', 'コンパイラ', 'インタープリタ'
            ]
            
            for keyword in search_keywords[:20]:  # 最大20キーワード
                search_url = f"{base_url}word/{quote(keyword)}"
                
                soup = self._fetch_page(search_url)
                if not soup:
                    continue
                
                # 記事リンクを抽出
                article_links = soup.find_all('a', href=re.compile(r'/word/'))
                
                for link in article_links[:10]:  # キーワードあたり最大10記事
                    article_url = urljoin(base_url, link.get('href', ''))
                    if article_url in self.visited_urls:
                        continue
                    
                    article_soup = self._fetch_page(article_url)
                    if not article_soup:
                        continue
                    
                    text = self._extract_text(article_soup)
                    if len(text) < 200:
                        continue
                    
                    samples.append({
                        'url': article_url,
                        'content': text,
                        'title': article_soup.title.string if article_soup.title else article_url,
                        'domain': 'encyclopedia',
                        'subdomain': 'kotobank',
                        'language': 'ja',
                        'timestamp': datetime.now().isoformat(),
                        'type': 'encyclopedia_entry'
                    })
                    
                    if len(samples) >= self.max_samples_per_site:
                        break
                
                if len(samples) >= self.max_samples_per_site:
                    break
                
                time.sleep(self.delay)
        
        except Exception as e:
            logger.error(f"[KOTOBANK] Failed: {e}")
        
        logger.info(f"[KOTOBANK] Collected {len(samples)} entries")
        return samples
    
    def crawl_weblio(self) -> List[Dict]:
        """Weblioからデータを収集"""
        logger.info("[WEBLIO] Crawling Weblio...")
        samples = []
        
        base_url = self.site_configs['weblio']['base_url']
        
        try:
            # Weblioのカテゴリページから用語を収集
            category_urls = [
                f"{base_url}category/",
                f"{base_url}weblio/",
            ]
            
            for category_url in category_urls:
                soup = self._fetch_page(category_url)
                if not soup:
                    continue
                
                # 用語リンクを抽出
                term_links = soup.find_all('a', href=re.compile(r'/content/|/weblio/'))
                
                for link in term_links[:100]:  # カテゴリあたり最大100用語
                    term_url = urljoin(base_url, link.get('href', ''))
                    if term_url in self.visited_urls:
                        continue
                    
                    term_soup = self._fetch_page(term_url)
                    if not term_soup:
                        continue
                    
                    text = self._extract_text(term_soup)
                    if len(text) < 100:  # 用語説明は短い場合もある
                        continue
                    
                    samples.append({
                        'url': term_url,
                        'content': text,
                        'title': term_soup.title.string if term_soup.title else term_url,
                        'domain': 'dictionary',
                        'subdomain': 'weblio',
                        'language': 'ja',
                        'timestamp': datetime.now().isoformat(),
                        'type': 'dictionary_entry'
                    })
                    
                    if len(samples) >= self.max_samples_per_site:
                        break
                
                if len(samples) >= self.max_samples_per_site:
                    break
                
                time.sleep(self.delay)
        
        except Exception as e:
            logger.error(f"[WEBLIO] Failed: {e}")
        
        logger.info(f"[WEBLIO] Collected {len(samples)} entries")
        return samples
    
    def crawl_jstage(self) -> List[Dict]:
        """J-STAGEからデータを収集"""
        logger.info("[J-STAGE] Crawling J-STAGE...")
        samples = []
        
        base_url = self.site_configs['jstage']['base_url']
        
        try:
            # J-STAGEの検索機能を使用
            search_url = f"{base_url}search/ja"
            search_params = {
                'q': 'open access',
                'from': '2020',
                'to': '2024',
                'page': 1,
                'count': 20
            }
            
            for page in range(1, 51):  # 最大50ページ
                search_params['page'] = page
                
                soup = self._fetch_page(f"{search_url}?{urlencode(search_params)}")
                if not soup:
                    break
                
                # 論文リンクを抽出
                paper_links = soup.find_all('a', href=re.compile(r'/article/|/journal/'))
                
                if not paper_links:
                    break
                
                for link in paper_links[:20]:  # ページあたり最大20論文
                    paper_url = urljoin(base_url, link.get('href', ''))
                    if paper_url in self.visited_urls:
                        continue
                    
                    paper_soup = self._fetch_page(paper_url)
                    if not paper_soup:
                        continue
                    
                    text = self._extract_text(paper_soup)
                    if len(text) < 300:  # 論文は長いテキストが必要
                        continue
                    
                    samples.append({
                        'url': paper_url,
                        'content': text,
                        'title': paper_soup.title.string if paper_soup.title else paper_url,
                        'domain': 'academic',
                        'subdomain': 'jstage',
                        'language': 'ja',
                        'timestamp': datetime.now().isoformat(),
                        'type': 'academic_paper'
                    })
                    
                    if len(samples) >= self.max_samples_per_site:
                        break
                
                if len(samples) >= self.max_samples_per_site:
                    break
                
                time.sleep(self.delay)
        
        except Exception as e:
            logger.error(f"[J-STAGE] Failed: {e}")
        
        logger.info(f"[J-STAGE] Collected {len(samples)} papers")
        return samples
    
    def crawl_cinii(self) -> List[Dict]:
        """CiNiiからデータを収集"""
        logger.info("[CINII] Crawling CiNii...")
        samples = []
        
        base_url = self.site_configs['cinii']['base_url']
        
        try:
            # CiNii Articles APIを使用
            api_key = os.environ.get('CINII_API_KEY')
            if api_key:
                # API使用
                api_url = f"{base_url}api/opensearch"
                params = {
                    'q': 'open access',
                    'count': 100,
                    'start': 1,
                    'appid': api_key
                }
                
                for start in range(1, 1001, 100):  # 最大1000件
                    params['start'] = start
                    
                    # CiNii API呼び出し（Chromeヘッダー使用）
                    api_headers = {
                        'User-Agent': self.user_agent,
                        'Accept': 'application/xml, text/xml, */*',
                        'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8'
                    }
                    response = requests.get(
                        api_url,
                        params=params,
                        headers=api_headers,
                        timeout=self.timeout
                    )
                    
                    if response.status_code != 200:
                        break
                    
                    # XML解析
                    try:
                        import xml.etree.ElementTree as ET
                        root = ET.fromstring(response.content)
                        
                        items = root.findall('.//item')
                        if not items:
                            break
                        
                        for item in items:
                            title_elem = item.find('title')
                            description_elem = item.find('description')
                            link_elem = item.find('link')
                            
                            title = title_elem.text if title_elem is not None else ''
                            description = description_elem.text if description_elem is not None else ''
                            link = link_elem.text if link_elem is not None else ''
                            
                            if not title:
                                continue
                            
                            content = title
                            if description:
                                content += f"\n\n{description}"
                            
                            samples.append({
                                'url': link,
                                'content': content,
                                'title': title,
                                'domain': 'academic',
                                'subdomain': 'cinii',
                                'language': 'ja',
                                'timestamp': datetime.now().isoformat(),
                                'type': 'academic_paper'
                            })
                            
                            if len(samples) >= self.max_samples_per_site:
                                break
                        
                        if len(samples) >= self.max_samples_per_site:
                            break
                        
                        time.sleep(self.delay)
                    
                    except Exception as e:
                        logger.debug(f"Failed to parse CiNii XML: {e}")
                        break
            else:
                # Webスクレイピング（フォールバック）
                logger.warning("[CINII] CINII_API_KEY not set. Using web scraping.")
                search_url = f"{base_url}search"
                search_params = {
                    'q': 'open access',
                    'range': '0',
                    'count': '20'
                }
                
                for page in range(0, 500, 20):  # 最大500件
                    search_params['range'] = str(page)
                    
                    soup = self._fetch_page(f"{search_url}?{urlencode(search_params)}")
                    if not soup:
                        break
                    
                    paper_links = soup.find_all('a', href=re.compile(r'/naid/'))
                    
                    if not paper_links:
                        break
                    
                    for link in paper_links[:20]:
                        paper_url = urljoin(base_url, link.get('href', ''))
                        if paper_url in self.visited_urls:
                            continue
                        
                        paper_soup = self._fetch_page(paper_url)
                        if not paper_soup:
                            continue
                        
                        text = self._extract_text(paper_soup)
                        if len(text) < 300:
                            continue
                        
                        samples.append({
                            'url': paper_url,
                            'content': text,
                            'title': paper_soup.title.string if paper_soup.title else paper_url,
                            'domain': 'academic',
                            'subdomain': 'cinii',
                            'language': 'ja',
                            'timestamp': datetime.now().isoformat(),
                            'type': 'academic_paper'
                        })
                        
                        if len(samples) >= self.max_samples_per_site:
                            break
                    
                    if len(samples) >= self.max_samples_per_site:
                        break
                    
                    time.sleep(self.delay)
        
        except Exception as e:
            logger.error(f"[CINII] Failed: {e}")
        
        logger.info(f"[CINII] Collected {len(samples)} papers")
        return samples
    
    def crawl_britannica(self) -> List[Dict]:
        """Britannicaからデータを収集"""
        logger.info("[BRITANNICA] Crawling Britannica...")
        samples = []
        
        base_url = self.site_configs['britannica']['base_url']
        
        # Britannicaの記事ページをクロール
        # カテゴリページや検索結果から記事URLを取得
        try:
            # サンプル: カテゴリページから記事を取得
            category_urls = [
                f"{base_url}topic/science",
                f"{base_url}topic/technology",
                f"{base_url}topic/mathematics"
            ]
            
            for category_url in category_urls[:10]:  # 最大10カテゴリ
                soup = self._fetch_page(category_url)
                if not soup:
                    continue
                
                # 記事リンクを抽出
                article_links = soup.find_all('a', href=re.compile(r'/topic/|/biography/'))
                
                for link in article_links[:20]:  # カテゴリあたり最大20記事
                    article_url = urljoin(base_url, link.get('href', ''))
                    if not article_url or article_url in self.visited_urls:
                        continue
                    
                    article_soup = self._fetch_page(article_url)
                    if not article_soup:
                        continue
                    
                    text = self._extract_text(article_soup)
                    if len(text) < 200:
                        continue
                    
                    samples.append({
                        'url': article_url,
                        'content': text,
                        'title': article_soup.title.string if article_soup.title else article_url,
                        'domain': 'encyclopedia',
                        'subdomain': 'britannica',
                        'language': 'en',
                        'timestamp': datetime.now().isoformat(),
                        'type': 'encyclopedia_entry'
                    })
                    
                    if len(samples) >= self.max_samples_per_site:
                        break
                
                if len(samples) >= self.max_samples_per_site:
                    break
        
        except Exception as e:
            logger.error(f"[BRITANNICA] Failed: {e}")
        
        logger.info(f"[BRITANNICA] Collected {len(samples)} entries")
        return samples
    
    def crawl_khan_academy(self) -> List[Dict]:
        """Khan Academyからデータを収集"""
        logger.info("[KHAN ACADEMY] Crawling Khan Academy...")
        samples = []
        
        base_url = self.site_configs['khan_academy']['base_url']
        
        try:
            # Khan Academyのカテゴリページからコンテンツを収集
            category_urls = [
                f"{base_url}computing/computer-programming",
                f"{base_url}computing/computer-science",
                f"{base_url}math",
                f"{base_url}science"
            ]
            
            for category_url in category_urls:
                soup = self._fetch_page(category_url)
                if not soup:
                    continue
                
                # レッスンリンクを抽出
                lesson_links = soup.find_all('a', href=re.compile(r'/a/|/computing/|/math/|/science/'))
                
                for link in lesson_links[:50]:  # カテゴリあたり最大50レッスン
                    lesson_url = urljoin(base_url, link.get('href', ''))
                    if lesson_url in self.visited_urls or '/a/' not in lesson_url:
                        continue
                    
                    lesson_soup = self._fetch_page(lesson_url)
                    if not lesson_soup:
                        continue
                    
                    text = self._extract_text(lesson_soup)
                    if len(text) < 200:
                        continue
                    
                    samples.append({
                        'url': lesson_url,
                        'content': text,
                        'title': lesson_soup.title.string if lesson_soup.title else lesson_url,
                        'domain': 'education',
                        'subdomain': 'khan_academy',
                        'language': 'en',
                        'timestamp': datetime.now().isoformat(),
                        'type': 'educational_content'
                    })
                    
                    if len(samples) >= self.max_samples_per_site:
                        break
                
                if len(samples) >= self.max_samples_per_site:
                    break
                
                time.sleep(self.delay)
        
        except Exception as e:
            logger.error(f"[KHAN ACADEMY] Failed: {e}")
        
        logger.info(f"[KHAN ACADEMY] Collected {len(samples)} entries")
        return samples
    
    def crawl_mit_ocw(self) -> List[Dict]:
        """MIT OpenCourseWareからデータを収集"""
        logger.info("[MIT OCW] Crawling MIT OpenCourseWare...")
        samples = []
        
        base_url = self.site_configs['mit_ocw']['base_url']
        
        try:
            # MIT OCWのコース一覧ページから講義資料を収集
            courses_url = f"{base_url}courses/"
            
            soup = self._fetch_page(courses_url)
            if not soup:
                return samples
            
            # コースリンクを抽出
            course_links = soup.find_all('a', href=re.compile(r'/courses/'))
            
            for link in course_links[:100]:  # 最大100コース
                course_url = urljoin(base_url, link.get('href', ''))
                if course_url in self.visited_urls or '/courses/' not in course_url:
                    continue
                
                # コースページからシラバスや講義資料を取得
                course_soup = self._fetch_page(course_url)
                if not course_soup:
                    continue
                
                # シラバスや講義ノートのリンクを探す
                syllabus_links = course_soup.find_all('a', href=re.compile(r'syllabus|lecture|notes'))
                
                for syllabus_link in syllabus_links[:5]:  # コースあたり最大5資料
                    syllabus_url = urljoin(course_url, syllabus_link.get('href', ''))
                    if syllabus_url in self.visited_urls:
                        continue
                    
                    syllabus_soup = self._fetch_page(syllabus_url)
                    if not syllabus_soup:
                        continue
                    
                    text = self._extract_text(syllabus_soup)
                    if len(text) < 300:
                        continue
                    
                    samples.append({
                        'url': syllabus_url,
                        'content': text,
                        'title': syllabus_soup.title.string if syllabus_soup.title else syllabus_url,
                        'domain': 'education',
                        'subdomain': 'mit_ocw',
                        'language': 'en',
                        'timestamp': datetime.now().isoformat(),
                        'type': 'course_material'
                    })
                    
                    if len(samples) >= self.max_samples_per_site:
                        break
                
                if len(samples) >= self.max_samples_per_site:
                    break
                
                time.sleep(self.delay)
        
        except Exception as e:
            logger.error(f"[MIT OCW] Failed: {e}")
        
        logger.info(f"[MIT OCW] Collected {len(samples)} entries")
        return samples
    
    def crawl(self) -> List[Dict]:
        """全サイトからデータを収集"""
        logger.info("="*80)
        logger.info("Domain Knowledge Crawler")
        logger.info("="*80)
        
        all_samples = []
        
        # 日本語サイト
        for site in self.japanese_sites:
            if site == 'kotobank':
                samples = self.crawl_kotobank()
            elif site == 'weblio':
                samples = self.crawl_weblio()
            elif site == 'jstage':
                samples = self.crawl_jstage()
            elif site == 'cinii':
                samples = self.crawl_cinii()
            else:
                logger.warning(f"Unknown Japanese site: {site}")
                continue
            
            all_samples.extend(samples)
        
        # 英語サイト
        for site in self.english_sites:
            if site == 'britannica':
                samples = self.crawl_britannica()
            elif site == 'khan_academy':
                samples = self.crawl_khan_academy()
            elif site == 'mit_ocw':
                samples = self.crawl_mit_ocw()
            else:
                logger.warning(f"Unknown English site: {site}")
                continue
            
            all_samples.extend(samples)
        
        self.samples = all_samples
        logger.info(f"[TOTAL] Collected {len(self.samples)} samples")
        return self.samples
    
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
    
    parser = argparse.ArgumentParser(description='Domain Knowledge Crawler')
    parser.add_argument('--output', type=str, default='D:/webdataset/processed/domain_knowledge.jsonl')
    parser.add_argument('--japanese-sites', nargs='+', 
                       default=['kotobank', 'weblio', 'jstage', 'cinii'])
    parser.add_argument('--english-sites', nargs='+',
                       default=['britannica', 'khan_academy', 'mit_ocw'])
    parser.add_argument('--max-samples', type=int, default=50000)
    parser.add_argument('--delay', type=float, default=1.0)
    
    args = parser.parse_args()
    
    config = {
        'japanese_sites': args.japanese_sites,
        'english_sites': args.english_sites,
        'max_samples_per_site': args.max_samples,
        'delay': args.delay,
        'timeout': 15,
        'user_agent': 'SO8T-DataCollector/1.0 (Research Purpose)'
    }
    
    crawler = DomainKnowledgeCrawler(config)
    crawler.crawl()
    crawler.save(Path(args.output))


if __name__ == '__main__':
    main()

