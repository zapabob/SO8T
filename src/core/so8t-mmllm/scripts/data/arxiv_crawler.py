#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arxiv論文収集クローラー

Arxiv APIを使用して科学論文を収集

カテゴリ:
- cs.AI (Artificial Intelligence)
- cs.CL (Computation and Language)
- cs.LG (Machine Learning)
- math (Mathematics)
- physics (Physics)
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from tqdm import tqdm

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# arxivパッケージのインポート
try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("arxiv package not available. Install with: pip install arxiv")

logger = logging.getLogger(__name__)


class ArxivCrawler:
    """Arxiv論文収集クローラー"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: クローラー設定
        """
        self.config = config
        self.categories = config.get('categories', ['cs.AI', 'cs.CL', 'cs.LG', 'math', 'physics'])
        self.max_papers = config.get('max_papers', 50000)
        self.date_from = config.get('date_from', '2020-01-01')
        self.delay = config.get('delay', 1.0)  # Arxiv APIレート制限対応
        
        self.samples: List[Dict] = []
        
        if not ARXIV_AVAILABLE:
            raise ImportError("arxiv package is required. Install with: pip install arxiv")
    
    def _parse_date(self, date_str: str) -> datetime:
        """日付文字列をパース"""
        try:
            return datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            return datetime(2020, 1, 1)
    
    def _extract_text_from_summary(self, summary: str) -> str:
        """アブストラクトからテキストを抽出"""
        # LaTeXコマンドを簡易的に削除
        import re
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', summary)
        text = re.sub(r'\$[^$]+\$', '', text)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def crawl_category(self, category: str) -> List[Dict]:
        """特定カテゴリの論文を収集"""
        logger.info(f"[ARXIV] Crawling category: {category}")
        
        date_from = self._parse_date(self.date_from)
        samples = []
        
        # Arxiv検索クエリ構築
        query = f"cat:{category}"
        if date_from:
            query += f" AND submittedDate:[{date_from.strftime('%Y%m%d')}000000 TO {datetime.now().strftime('%Y%m%d')}235959]"
        
        try:
            search = arxiv.Search(
                query=query,
                max_results=min(self.max_papers, 10000),  # Arxiv API制限
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            for paper in tqdm(search.results(), desc=f"Fetching {category}"):
                try:
                    # 論文メタデータ取得
                    title = paper.title
                    summary = paper.summary
                    authors = [author.name for author in paper.authors]
                    published = paper.published.isoformat()
                    arxiv_id = paper.entry_id.split('/')[-1]
                    
                    # テキスト抽出
                    text = self._extract_text_from_summary(summary)
                    if len(text) < 100:
                        continue
                    
                    # サンプル作成
                    sample = {
                        'url': paper.entry_id,
                        'content': f"{title}\n\n{text}",
                        'title': title,
                        'authors': authors,
                        'published': published,
                        'arxiv_id': arxiv_id,
                        'category': category,
                        'domain': 'scientific_paper',
                        'subdomain': category,
                        'language': 'en',  # Arxivは主に英語
                        'timestamp': datetime.now().isoformat(),
                        'type': 'arxiv_paper'
                    }
                    
                    samples.append(sample)
                    
                    # レート制限対応
                    time.sleep(self.delay)
                    
                    if len(samples) >= self.max_papers:
                        break
                        
                except Exception as e:
                    logger.debug(f"Failed to process paper: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"[ARXIV] Failed to crawl category {category}: {e}")
        
        logger.info(f"[ARXIV] Collected {len(samples)} papers from {category}")
        return samples
    
    def crawl(self) -> List[Dict]:
        """全カテゴリの論文を収集"""
        logger.info("="*80)
        logger.info("Arxiv Paper Crawler")
        logger.info("="*80)
        
        all_samples = []
        papers_per_category = self.max_papers // len(self.categories)
        
        for category in self.categories:
            samples = self.crawl_category(category)
            all_samples.extend(samples)
            
            if len(all_samples) >= self.max_papers:
                break
        
        self.samples = all_samples[:self.max_papers]
        logger.info(f"[ARXIV] [TOTAL] Collected {len(self.samples)} papers")
        return self.samples
    
    def save(self, output_path: Path):
        """結果を保存"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in self.samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"[ARXIV] [SAVE] Saved {len(self.samples)} papers to {output_path}")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Arxiv Paper Crawler')
    parser.add_argument('--output', type=str, default='D:/webdataset/processed/arxiv_papers.jsonl')
    parser.add_argument('--categories', nargs='+', 
                       default=['cs.AI', 'cs.CL', 'cs.LG', 'math', 'physics'])
    parser.add_argument('--max-papers', type=int, default=50000)
    parser.add_argument('--date-from', type=str, default='2020-01-01')
    parser.add_argument('--delay', type=float, default=1.0)
    
    args = parser.parse_args()
    
    config = {
        'categories': args.categories,
        'max_papers': args.max_papers,
        'date_from': args.date_from,
        'delay': args.delay
    }
    
    crawler = ArxivCrawler(config)
    crawler.crawl()
    crawler.save(Path(args.output))


if __name__ == '__main__':
    main()







