#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
エンジニア向けサイトスクレイピングスクリプト

Stack Overflow、Qiita、Zenn、Medium、Dev.toなどの
エンジニア向けサイトから技術記事とコードスニペットを抽出します。

Usage:
    python scripts/data/engineer_site_scraper.py --output D:\webdataset\processed
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
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/engineer_site_scraper.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EngineerSiteScraper:
    """エンジニア向けサイトスクレイパー"""
    
    def __init__(
        self,
        output_dir: Path,
        delay_per_request: float = 2.0,
        max_articles_per_site: int = 100
    ):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリ
            delay_per_request: リクエスト間の遅延（秒）
            max_articles_per_site: サイトあたりの最大記事数
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.delay_per_request = delay_per_request
        self.max_articles_per_site = max_articles_per_site
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.all_samples: List[Dict] = []
        
        logger.info("="*80)
        logger.info("Engineer Site Scraper Initialized")
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Max articles per site: {self.max_articles_per_site}")
    
    def scrape_stackoverflow(self, queries: List[str]) -> List[Dict]:
        """Stack Overflowをスクレイピング"""
        samples = []
        
        for query in queries:
            try:
                url = "https://stackoverflow.com/search"
                params = {'q': query}
                
                response = self.session.get(url, params=params, timeout=30)
                if response.status_code != 200:
                    logger.warning(f"[STACKOVERFLOW] Failed to fetch: {response.status_code}")
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 質問のリンクを取得
                question_links = soup.select('a.question-hyperlink')[:self.max_articles_per_site]
                
                for link in question_links:
                    question_url = urljoin('https://stackoverflow.com', link.get('href', ''))
                    
                    try:
                        question_response = self.session.get(question_url, timeout=30)
                        if question_response.status_code == 200:
                            question_soup = BeautifulSoup(question_response.text, 'html.parser')
                            
                            # 質問のタイトルと本文を取得
                            title = question_soup.select_one('h1 a.question-hyperlink')
                            title_text = title.get_text(strip=True) if title else ''
                            
                            question_body = question_soup.select_one('.question .post-text')
                            question_text = question_body.get_text(strip=True) if question_body else ''
                            
                            # コードブロックを取得
                            code_blocks = question_soup.select('.question pre code')
                            code_snippets = [code.get_text(strip=True) for code in code_blocks]
                            
                            # 回答を取得
                            answers = question_soup.select('.answer .post-text')
                            answer_texts = [answer.get_text(strip=True) for answer in answers[:3]]  # 上位3つの回答
                            
                            # サンプルを作成
                            text = f"{title_text}\n\n{question_text}\n\n" + "\n\n".join(answer_texts)
                            
                            sample = {
                                'text': text,
                                'url': question_url,
                                'domain': 'stackoverflow.com',
                                'category': 'programming',
                                'language': 'en',
                                'metadata': {
                                    'code_snippets_count': len(code_snippets),
                                    'answers_count': len(answers)
                                },
                                'timestamp': datetime.now().isoformat()
                            }
                            
                            samples.append(sample)
                            time.sleep(self.delay_per_request)
                    
                    except Exception as e:
                        logger.debug(f"[STACKOVERFLOW] Failed to fetch question {question_url}: {e}")
                        continue
                
            except Exception as e:
                logger.error(f"[STACKOVERFLOW] Error: {e}")
                continue
        
        logger.info(f"[STACKOVERFLOW] Scraped {len(samples)} samples")
        return samples
    
    def scrape_qiita(self, queries: List[str]) -> List[Dict]:
        """Qiitaをスクレイピング"""
        samples = []
        
        for query in queries:
            try:
                url = "https://qiita.com/search"
                params = {'q': query}
                
                response = self.session.get(url, params=params, timeout=30)
                if response.status_code != 200:
                    logger.warning(f"[QIITA] Failed to fetch: {response.status_code}")
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 記事のリンクを取得
                article_links = soup.select('a[href^="/items/"]')[:self.max_articles_per_site]
                
                for link in article_links:
                    article_url = urljoin('https://qiita.com', link.get('href', ''))
                    
                    try:
                        article_response = self.session.get(article_url, timeout=30)
                        if article_response.status_code == 200:
                            article_soup = BeautifulSoup(article_response.text, 'html.parser')
                            
                            # タイトルと本文を取得
                            title = article_soup.select_one('h1.article-title')
                            title_text = title.get_text(strip=True) if title else ''
                            
                            body = article_soup.select_one('.it-MdContent')
                            body_text = body.get_text(strip=True) if body else ''
                            
                            # コードブロックを取得
                            code_blocks = article_soup.select('pre code')
                            code_snippets = [code.get_text(strip=True) for code in code_blocks]
                            
                            # サンプルを作成
                            sample = {
                                'text': f"{title_text}\n\n{body_text}",
                                'url': article_url,
                                'domain': 'qiita.com',
                                'category': 'programming',
                                'language': 'ja',
                                'metadata': {
                                    'code_snippets_count': len(code_snippets)
                                },
                                'timestamp': datetime.now().isoformat()
                            }
                            
                            samples.append(sample)
                            time.sleep(self.delay_per_request)
                    
                    except Exception as e:
                        logger.debug(f"[QIITA] Failed to fetch article {article_url}: {e}")
                        continue
                
            except Exception as e:
                logger.error(f"[QIITA] Error: {e}")
                continue
        
        logger.info(f"[QIITA] Scraped {len(samples)} samples")
        return samples
    
    def scrape_zenn(self, queries: List[str]) -> List[Dict]:
        """Zennをスクレイピング"""
        samples = []
        
        for query in queries:
            try:
                url = "https://zenn.dev/search"
                params = {'q': query}
                
                response = self.session.get(url, params=params, timeout=30)
                if response.status_code != 200:
                    logger.warning(f"[ZENN] Failed to fetch: {response.status_code}")
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 記事のリンクを取得
                article_links = soup.select('a[href^="/articles/"]')[:self.max_articles_per_site]
                
                for link in article_links:
                    article_url = urljoin('https://zenn.dev', link.get('href', ''))
                    
                    try:
                        article_response = self.session.get(article_url, timeout=30)
                        if article_response.status_code == 200:
                            article_soup = BeautifulSoup(article_response.text, 'html.parser')
                            
                            # タイトルと本文を取得
                            title = article_soup.select_one('h1')
                            title_text = title.get_text(strip=True) if title else ''
                            
                            body = article_soup.select_one('.znc')
                            body_text = body.get_text(strip=True) if body else ''
                            
                            # コードブロックを取得
                            code_blocks = article_soup.select('pre code')
                            code_snippets = [code.get_text(strip=True) for code in code_blocks]
                            
                            # サンプルを作成
                            sample = {
                                'text': f"{title_text}\n\n{body_text}",
                                'url': article_url,
                                'domain': 'zenn.dev',
                                'category': 'programming',
                                'language': 'ja',
                                'metadata': {
                                    'code_snippets_count': len(code_snippets)
                                },
                                'timestamp': datetime.now().isoformat()
                            }
                            
                            samples.append(sample)
                            time.sleep(self.delay_per_request)
                    
                    except Exception as e:
                        logger.debug(f"[ZENN] Failed to fetch article {article_url}: {e}")
                        continue
                
            except Exception as e:
                logger.error(f"[ZENN] Error: {e}")
                continue
        
        logger.info(f"[ZENN] Scraped {len(samples)} samples")
        return samples
    
    def scrape_all_sites(self, queries: List[str]) -> List[Dict]:
        """すべてのサイトをスクレイピング"""
        all_samples = []
        
        # Stack Overflow
        logger.info("[START] Scraping Stack Overflow...")
        stackoverflow_samples = self.scrape_stackoverflow(queries)
        all_samples.extend(stackoverflow_samples)
        
        # Qiita
        logger.info("[START] Scraping Qiita...")
        qiita_samples = self.scrape_qiita(queries)
        all_samples.extend(qiita_samples)
        
        # Zenn
        logger.info("[START] Scraping Zenn...")
        zenn_samples = self.scrape_zenn(queries)
        all_samples.extend(zenn_samples)
        
        return all_samples
    
    def save_samples(self, samples: List[Dict]):
        """サンプルを保存"""
        output_file = self.output_dir / f"engineer_sites_{self.session_id}.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"[SAVE] Saved {len(samples)} samples to {output_file}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Engineer Site Scraper')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--queries', type=str, nargs='+', default=['Python', 'JavaScript', 'programming'], help='Search queries')
    parser.add_argument('--delay', type=float, default=2.0, help='Delay between requests (seconds)')
    parser.add_argument('--max-articles', type=int, default=100, help='Max articles per site')
    
    args = parser.parse_args()
    
    scraper = EngineerSiteScraper(
        output_dir=args.output,
        delay_per_request=args.delay,
        max_articles_per_site=args.max_articles
    )
    
    samples = scraper.scrape_all_sites(args.queries)
    
    scraper.save_samples(samples)
    
    logger.info(f"[COMPLETE] Scraped {len(samples)} samples from engineer sites")


if __name__ == '__main__':
    main()

