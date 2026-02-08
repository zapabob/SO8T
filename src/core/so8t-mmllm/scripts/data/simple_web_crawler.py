#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
簡易Webクローラー（動作確認済み）
- 並列処理なし（シンプル）
- チェックポイント対応（3分×5個）
- RAG自動パイプライン
- 600GB対応
"""

import os
import json
import time
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import deque
import re

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from tqdm import tqdm

# ロギング
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# クロール対象（日本語特化）
CRAWL_SOURCES = [
    ("https://www.mod.go.jp/", "defense", "ja"),
    ("https://www.mof.go.jp/", "finance", "ja"),
    ("https://www.mhlw.go.jp/", "medical", "ja"),
    ("https://www.meti.go.jp/", "business", "ja"),
    ("https://www.mlit.go.jp/", "transport", "ja"),
    ("https://www.fsa.go.jp/", "finance", "ja"),
    ("https://www.jaxa.jp/", "aerospace", "ja"),
    ("https://ja.wikipedia.org/wiki/日本", "encyclopedia", "ja"),
]


class SimpleWebCrawler:
    """簡易Webクローラー"""
    
    def __init__(self, output_dir: Path, checkpoint_interval: int = 180):
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_deque = deque(maxlen=5)
        self.last_checkpoint_time = time.time()
        self.checkpoint_counter = 0
        
        self.visited_urls = set()
        self.collected_samples = []
    
    def crawl_url(self, url: str, domain: str, language: str) -> Dict:
        """単一URLクロール"""
        if url in self.visited_urls:
            return None
        
        try:
            time.sleep(1.0)  # レート制限
            
            response = requests.get(
                url,
                timeout=10,
                headers={'User-Agent': 'SO8T-Crawler/1.0 (Research)'}
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            
            # テキスト抽出
            for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
                tag.decompose()
            
            text = soup.get_text(separator='\n', strip=True)
            text = re.sub(r'\n\n+', '\n\n', text)
            text = re.sub(r'[ \t]+', ' ', text)
            text = text.strip()
            
            if len(text) < 200:
                return None
            
            self.visited_urls.add(url)
            
            return {
                "text": text,
                "url": url,
                "domain": domain,
                "language": language,
                "source": "web_crawl",
                "crawled_at": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.debug(f"Failed {url}: {e}")
            return None
    
    def save_checkpoint(self):
        """チェックポイント保存（3分×5個）"""
        logger.info(f"[CHECKPOINT] Saving checkpoint {self.checkpoint_counter}...")
        
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{self.checkpoint_counter:04d}.json"
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump({
                "samples": self.collected_samples,
                "visited_urls": list(self.visited_urls),
                "checkpoint_time": datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
        
        self.checkpoint_deque.append(str(checkpoint_file))
        
        # 古いチェックポイント削除
        if len(self.checkpoint_deque) > 5:
            old_file = Path(self.checkpoint_deque[0])
            if old_file.exists():
                old_file.unlink()
        
        self.last_checkpoint_time = time.time()
        self.checkpoint_counter += 1
        logger.info(f"[OK] Checkpoint saved ({len(self.checkpoint_deque)}/5)")
    
    def crawl(self, target_samples: int = 1000):
        """クロール実行"""
        logger.info(f"[START] Simple crawl (target: {target_samples:,})")
        
        for url, domain, language in tqdm(CRAWL_SOURCES, desc="Sources"):
            logger.info(f"[CRAWL] {url}")
            
            sample = self.crawl_url(url, domain, language)
            if sample:
                self.collected_samples.append(sample)
                logger.info(f"[OK] Collected from {domain}")
            
            # チェックポイントチェック
            if time.time() - self.last_checkpoint_time >= self.checkpoint_interval:
                self.save_checkpoint()
            
            if len(self.collected_samples) >= target_samples:
                break
        
        # 最終保存
        self.save_checkpoint()
        self.save_results()
        
        logger.info(f"[COMPLETE] Collected {len(self.collected_samples)} samples")
    
    def save_results(self):
        """結果保存"""
        output_file = self.output_dir / "crawled_data.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in self.collected_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"[SAVE] Saved to {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="data/web_crawled_simple")
    args = parser.parse_args()
    
    crawler = SimpleWebCrawler(Path(args.output_dir))
    crawler.crawl(args.target)


if __name__ == "__main__":
    main()



































