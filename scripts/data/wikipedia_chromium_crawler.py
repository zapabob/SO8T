#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wikipedia Chromium クローラー

Chromium（Playwright）を使用してWikipedia日本語・英語をクロールし、
指定ドメイン（防衛、航空宇宙、半導体、精密機器、インフラ、運輸）の知識を収集。

Usage:
    python scripts/data/wikipedia_chromium_crawler.py --output D:\webdataset --target 10000
"""

import os
import sys
import json
import time
import logging
import hashlib
import re
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, deque
from urllib.parse import urljoin, urlparse, quote
from urllib.robotparser import RobotFileParser

import asyncio
from playwright.async_api import async_playwright, Browser, Page, TimeoutError as PlaywrightTimeoutError
from bs4 import BeautifulSoup
from tqdm import tqdm

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.data.auto_labeler_thinking import ThinkingAutoLabeler

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/wikipedia_crawl.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 設定
CRAWL_CONFIG = {
    "max_depth": 3,
    "delay_per_request": 1.0,
    "timeout": 30000,
    "max_pages_per_domain": 1000,
    "checkpoint_interval": 180,
    "max_checkpoints": 5,
    "min_text_length": 200,
    "max_text_length": 5000,
}

# Wikipedia URL
WIKIPEDIA_BASE_URLS = {
    "ja": "https://ja.wikipedia.org/wiki/",
    "en": "https://en.wikipedia.org/wiki/",
}

# robots.txt キャッシュ
ROBOTS_CACHE: Dict[str, RobotFileParser] = {}


def load_domain_keywords() -> Dict:
    """ドメイン別キーワード定義を読み込み"""
    keywords_file = Path(__file__).parent / "wikipedia_domain_keywords.json"
    with open(keywords_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def check_robots_txt(url: str) -> bool:
    """robots.txtを確認してクロール可能かチェック"""
    parsed = urlparse(url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    
    if base_url not in ROBOTS_CACHE:
        rp = RobotFileParser()
        robots_url = f"{base_url}/robots.txt"
        try:
            rp.set_url(robots_url)
            rp.read()
            ROBOTS_CACHE[base_url] = rp
        except Exception as e:
            logger.warning(f"Failed to read robots.txt from {robots_url}: {e}")
            ROBOTS_CACHE[base_url] = None
    
    rp = ROBOTS_CACHE.get(base_url)
    if rp is None:
        return True  # robots.txtが読めない場合は許可
    
    return rp.can_fetch('*', url)


def extract_text_from_html(html: str) -> str:
    """HTMLからテキストを抽出"""
    soup = BeautifulSoup(html, 'lxml')
    
    # 不要な要素を削除
    for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'table']):
        tag.decompose()
    
    # メインコンテンツを取得
    content = soup.find('div', {'id': 'mw-content-text'}) or soup.find('body')
    if content is None:
        return ""
    
    text = content.get_text(separator='\n', strip=True)
    
    # 正規化
    text = re.sub(r'\n\n+', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()
    
    return text


def classify_domain(text: str, title: str, keywords_config: Dict) -> Optional[str]:
    """テキストとタイトルからドメインを分類"""
    text_lower = text.lower()
    title_lower = title.lower()
    combined = f"{title_lower} {text_lower}"
    
    domain_scores = {}
    
    for domain_key, domain_config in keywords_config["domains"].items():
        score = 0
        
        # 日本語キーワード
        for keyword in domain_config["keywords_ja"]:
            if keyword in combined:
                score += 2
            if keyword in title_lower:
                score += 3
        
        # 英語キーワード
        for keyword in domain_config["keywords_en"]:
            if keyword.lower() in combined:
                score += 2
            if keyword.lower() in title_lower:
                score += 3
        
        if score > 0:
            domain_scores[domain_key] = score
    
    if not domain_scores:
        return None
    
    # 最高スコアのドメインを返す
    return max(domain_scores.items(), key=lambda x: x[1])[0]


def get_wikipedia_links(page: Page, language: str) -> List[str]:
    """Wikipediaページから関連リンクを取得"""
    try:
        # メインコンテンツ内のリンクを取得
        links = page.query_selector_all('#mw-content-text a[href^="/wiki/"]')
        urls = []
        
        for link in links[:50]:  # 最大50リンク
            href = link.get_attribute('href')
            if href and not href.startswith('/wiki/Special:'):
                full_url = f"https://{language}.wikipedia.org{href}"
                urls.append(full_url)
        
        return urls
    except Exception as e:
        logger.debug(f"Failed to extract links: {e}")
        return []


class WikipediaChromiumCrawler:
    """Wikipedia Chromium クローラー"""
    
    def __init__(
        self,
        output_dir: Path,
        keywords_config: Dict,
        target_samples_per_domain: int = 1000,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keywords_config = keywords_config
        self.target_samples_per_domain = target_samples_per_domain
        
        self.visited_urls: Set[str] = set()
        self.collected_samples: List[Dict] = []
        self.domain_samples: Dict[str, List[Dict]] = defaultdict(list)
        
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_deque = deque(maxlen=CRAWL_CONFIG["max_checkpoints"])
        self.last_checkpoint_time = time.time()
        self.checkpoint_counter = 0
        
        # ラベル付け用
        self.labeler = ThinkingAutoLabeler(keywords_config)
    
    async def crawl_page(
        self,
        page: Page,
        url: str,
        language: str,
        depth: int = 0
    ) -> Optional[Dict]:
        """単一ページをクロール"""
        if url in self.visited_urls:
            return None
        
        if depth > CRAWL_CONFIG["max_depth"]:
            return None
        
        if not check_robots_txt(url):
            logger.debug(f"Robots.txt disallows: {url}")
            return None
        
        try:
            # レート制限
            await asyncio.sleep(CRAWL_CONFIG["delay_per_request"])
            
            # ページに移動
            await page.goto(url, timeout=CRAWL_CONFIG["timeout"], wait_until="networkidle")
            
            # タイトル取得
            title = await page.title()
            title = title.replace(" - Wikipedia", "").strip()
            
            # HTML取得
            html = await page.content()
            
            # テキスト抽出
            text = extract_text_from_html(html)
            
            if len(text) < CRAWL_CONFIG["min_text_length"]:
                return None
            
            if len(text) > CRAWL_CONFIG["max_text_length"]:
                text = text[:CRAWL_CONFIG["max_text_length"]]
            
            # ドメイン分類
            domain = classify_domain(text, title, self.keywords_config)
            if domain is None:
                return None
            
            # ドメイン別サンプル数チェック
            if len(self.domain_samples[domain]) >= self.target_samples_per_domain:
                return None
            
            # サンプル作成
            sample = {
                "instruction": f"{title}について教えてください" if language == "ja" else f"Tell me about {title}",
                "input": "",
                "output": text[:500] + "..." if len(text) > 500 else text,
                "thinking": f"<think>This is a Wikipedia article about {title}. The content covers general information about the topic.</think>",
                "title": title,
                "url": url,
                "domain": domain,
                "language": language,
                "source": "wikipedia",
                "crawled_at": datetime.now().isoformat(),
                "text_length": len(text),
                "depth": depth,
            }
            
            # 自動ラベル付け
            labeled_sample = self.labeler.label_sample(sample, domain)
            
            self.visited_urls.add(url)
            
            return labeled_sample
        
        except PlaywrightTimeoutError:
            logger.debug(f"Timeout: {url}")
            return None
        except Exception as e:
            logger.debug(f"Failed to crawl {url}: {e}")
            return None
    
    async def crawl_domain(
        self,
        browser: Browser,
        domain: str,
        language: str,
        seed_urls: List[str]
    ):
        """ドメイン別にクロール"""
        logger.info(f"[CRAWL] Domain: {domain}, Language: {language}")
        
        page = await browser.new_page()
        
        try:
            # シードURLから開始
            queue = deque(seed_urls)
            processed = 0
            
            with tqdm(total=self.target_samples_per_domain, desc=f"{domain} ({language})") as pbar:
                while queue and len(self.domain_samples[domain]) < self.target_samples_per_domain:
                    if processed >= CRAWL_CONFIG["max_pages_per_domain"]:
                        break
                    
                    url = queue.popleft()
                    
                    # ページクロール
                    sample = await self.crawl_page(page, url, language, depth=0)
                    
                    if sample:
                        self.domain_samples[domain].append(sample)
                        self.collected_samples.append(sample)
                        pbar.update(1)
                        
                        # 関連リンクを取得（深度1まで）
                        if sample.get("depth", 0) < 1:
                            links = get_wikipedia_links(page, language)
                            for link in links[:10]:  # 最大10リンク
                                if link not in self.visited_urls:
                                    queue.append(link)
                    
                    processed += 1
                    
                    # チェックポイントチェック
                    if time.time() - self.last_checkpoint_time >= CRAWL_CONFIG["checkpoint_interval"]:
                        await self.save_checkpoint()
        
        finally:
            await page.close()
    
    async def save_checkpoint(self):
        """チェックポイント保存"""
        logger.info(f"[CHECKPOINT] Saving checkpoint {self.checkpoint_counter}...")
        
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{self.checkpoint_counter:04d}.json"
        
        checkpoint_data = {
            "samples": self.collected_samples,
            "domain_samples": {k: v for k, v in self.domain_samples.items()},
            "visited_urls": list(self.visited_urls),
            "checkpoint_time": datetime.now().isoformat(),
            "checkpoint_counter": self.checkpoint_counter,
        }
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        
        self.checkpoint_deque.append(str(checkpoint_file))
        
        # 古いチェックポイント削除
        if len(self.checkpoint_deque) > CRAWL_CONFIG["max_checkpoints"]:
            old_file = Path(self.checkpoint_deque[0])
            if old_file.exists():
                old_file.unlink()
        
        self.last_checkpoint_time = time.time()
        self.checkpoint_counter += 1
        logger.info(f"[OK] Checkpoint saved ({len(self.checkpoint_deque)}/{CRAWL_CONFIG['max_checkpoints']})")
    
    def generate_seed_urls(self, domain: str, language: str) -> List[str]:
        """ドメイン別のシードURLを生成"""
        domain_config = self.keywords_config["domains"][domain]
        keywords = domain_config["keywords_ja"] if language == "ja" else domain_config["keywords_en"]
        
        base_url = WIKIPEDIA_BASE_URLS[language]
        seed_urls = []
        
        # キーワードからWikipedia URLを生成
        for keyword in keywords[:20]:  # 最大20キーワード
            encoded = quote(keyword.replace(' ', '_'))
            url = f"{base_url}{encoded}"
            seed_urls.append(url)
        
        return seed_urls
    
    async def crawl(self):
        """クロール実行"""
        logger.info("="*80)
        logger.info("Wikipedia Chromium Crawler")
        logger.info("="*80)
        logger.info(f"Target samples per domain: {self.target_samples_per_domain:,}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("="*80)
        
        async with async_playwright() as p:
            # Chromiumブラウザを起動
            browser = await p.chromium.launch(headless=True)
            
            try:
                domains = list(self.keywords_config["domains"].keys())
                languages = ["ja", "en"]
                
                for domain in domains:
                    for language in languages:
                        if len(self.domain_samples[domain]) >= self.target_samples_per_domain:
                            continue
                        
                        seed_urls = self.generate_seed_urls(domain, language)
                        
                        await self.crawl_domain(
                            browser,
                            domain,
                            language,
                            seed_urls
                        )
                        
                        # チェックポイント保存
                        await self.save_checkpoint()
            
            finally:
                await browser.close()
        
        # 最終保存
        await self.save_checkpoint()
        self.save_results()
        
        logger.info("="*80)
        logger.info(f"[COMPLETE] Collected {len(self.collected_samples):,} samples")
        logger.info("="*80)
    
    def save_results(self):
        """結果を保存"""
        # ドメイン×言語別に保存
        for domain in self.keywords_config["domains"].keys():
            for language in ["ja", "en"]:
                samples = [
                    s for s in self.collected_samples
                    if s.get("domain") == domain and s.get("language") == language
                ]
                
                if not samples:
                    continue
                
                filename = f"wikipedia_{domain}_{language}.jsonl"
                output_file = self.output_dir / filename
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    for sample in samples:
                        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                
                logger.info(f"[SAVE] Saved {len(samples):,} samples to {filename}")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Wikipedia Chromium Crawler for SO8T /thinking model training data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=r"D:\webdataset",
        help="Output directory (default: D:\\webdataset)"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=1000,
        help="Target samples per domain (default: 1000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # シード設定
    random.seed(args.seed)
    
    # キーワード設定読み込み
    keywords_config = load_domain_keywords()
    
    # クローラー作成
    crawler = WikipediaChromiumCrawler(
        output_dir=Path(args.output),
        keywords_config=keywords_config,
        target_samples_per_domain=args.target,
    )
    
    try:
        # クロール実行
        asyncio.run(crawler.crawl())
        
        logger.info("[SUCCESS] Crawling completed")
        return 0
    
    except Exception as e:
        logger.error(f"[FAILED] Crawling failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

