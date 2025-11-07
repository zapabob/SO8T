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
from collections import defaultdict, deque, Counter
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
from scripts.data.crawler_error_handler import CrawlerErrorHandler, ErrorType, classify_exception
from scripts.data.retry_handler import RetryHandler
from scripts.data.text_extractor import WikipediaTextExtractor
from scripts.data.domain_classifier import MLDomainClassifier

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


# テキスト抽出器（後方互換性のため関数も残す）
_text_extractor = None

def get_text_extractor() -> WikipediaTextExtractor:
    """テキスト抽出器を取得（シングルトン）"""
    global _text_extractor
    if _text_extractor is None:
        _text_extractor = WikipediaTextExtractor(
            min_text_length=CRAWL_CONFIG["min_text_length"],
            max_text_length=CRAWL_CONFIG["max_text_length"]
        )
    return _text_extractor

def extract_text_from_html(html: str) -> str:
    """HTMLからテキストを抽出（後方互換性のため）"""
    extractor = get_text_extractor()
    text, _ = extractor.extract_text(html)
    return text


# ドメイン分類器（後方互換性のため関数も残す）
_domain_classifier = None

def get_domain_classifier(keywords_config: Dict) -> MLDomainClassifier:
    """ドメイン分類器を取得（シングルトン）"""
    global _domain_classifier
    if _domain_classifier is None:
        _domain_classifier = MLDomainClassifier(
            keywords_config,
            use_ml=True,  # 機械学習を使用（訓練データがない場合はフォールバック）
            classifier_type="naive_bayes"
        )
    return _domain_classifier

def classify_domain(text: str, title: str, keywords_config: Dict) -> Optional[str]:
    """テキストとタイトルからドメインを分類（後方互換性のため）"""
    classifier = get_domain_classifier(keywords_config)
    domain, confidence = classifier.classify(text, title)
    
    # 確信度が低い場合はNoneを返す（後方互換性）
    if domain and confidence < 0.3:
        return None
    
    return domain


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
        enable_retry: bool = True,
        enable_ml_classification: bool = True,
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
        
        # エラーハンドラー
        error_log_dir = self.output_dir / "error_logs"
        self.error_handler = CrawlerErrorHandler(log_dir=error_log_dir)
        
        # リトライハンドラー
        self.retry_handler = RetryHandler(
            max_retries=3,
            initial_delay=1.0,
            backoff_factor=2.0,
            max_delay=10.0
        ) if enable_retry else None
        
        # テキスト抽出器
        self.text_extractor = WikipediaTextExtractor(
            min_text_length=CRAWL_CONFIG["min_text_length"],
            max_text_length=CRAWL_CONFIG["max_text_length"]
        )
        
        # ドメイン分類器
        self.domain_classifier = MLDomainClassifier(
            keywords_config,
            use_ml=enable_ml_classification,
            classifier_type="naive_bayes"
        )
    
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
            
            # ページに移動（リトライ機能付き）
            async def navigate_to_page():
                await page.goto(url, timeout=CRAWL_CONFIG["timeout"], wait_until="networkidle")
            
            if self.retry_handler:
                try:
                    await self.retry_handler.retry_async(
                        navigate_to_page,
                        operation_name="page_navigation",
                        retryable_exceptions=(PlaywrightTimeoutError, Exception)
                    )
                except Exception as e:
                    error_type = classify_exception(e)
                    should_continue = self.error_handler.handle_error(
                        error_type, url, e, context={"operation": "navigation"}
                    )
                    if not should_continue:
                        return None
            else:
                try:
                    await navigate_to_page()
                except Exception as e:
                    error_type = classify_exception(e)
                    should_continue = self.error_handler.handle_error(
                        error_type, url, e, context={"operation": "navigation"}
                    )
                    if not should_continue:
                        return None
            
            # タイトル取得
            try:
                title = await page.title()
                title = title.replace(" - Wikipedia", "").replace(" - ウィキペディア", "").strip()
                logger.debug(f"[PAGE] Loaded page: {title}")
            except Exception as e:
                error_type = classify_exception(e)
                self.error_handler.handle_error(error_type, url, e, context={"operation": "title_extraction"})
                return None
            
            # HTML取得
            try:
                html = await page.content()
            except Exception as e:
                error_type = classify_exception(e)
                self.error_handler.handle_error(error_type, url, e, context={"operation": "html_extraction"})
                return None
            
            # テキスト抽出（高度な抽出器を使用）
            try:
                text, quality_score = self.text_extractor.extract_text(html, url)
                logger.debug(f"[TEXT] Extracted {len(text)} characters from {title} (quality: {quality_score:.4f})")
                
                if not text or quality_score < 0.3:
                    logger.debug(f"[SKIP] Text quality too low: {quality_score:.4f}")
                    return None
            except Exception as e:
                error_type = classify_exception(e)
                self.error_handler.handle_error(error_type, url, e, context={"operation": "text_extraction"})
                # フォールバック: 従来の抽出方法
                text = extract_text_from_html(html)
                if not text:
                    return None
            
            # ドメイン分類（機械学習ベース）
            try:
                domain, confidence = self.domain_classifier.classify(text, title)
                if domain is None or confidence < 0.3:
                    logger.debug(f"[SKIP] Domain classification failed for {title} (confidence: {confidence:.4f})")
                    return None
                
                logger.debug(f"[DOMAIN] Classified '{title}' as {domain} (confidence: {confidence:.4f})")
            except Exception as e:
                error_type = classify_exception(e)
                self.error_handler.handle_error(error_type, url, e, context={"operation": "domain_classification"})
                # フォールバック: 従来の分類方法
                domain = classify_domain(text, title, self.keywords_config)
                if domain is None:
                    return None
                confidence = 0.5  # デフォルト確信度
            
            # ドメイン別サンプル数チェック
            if len(self.domain_samples[domain]) >= self.target_samples_per_domain:
                return None
            
            # サンプル作成
            sample = {
                "instruction": f"{title}について教えてください" if language == "ja" else f"Tell me about {title}",
                "input": "",
                "output": text[:500] + "..." if len(text) > 500 else text,
                "title": title,
                "url": url,
                "domain": domain,
                "language": language,
                "source": "wikipedia",
                "crawled_at": datetime.now().isoformat(),
                "text_length": len(text),
                "depth": depth,
                "quality_score": quality_score if 'quality_score' in locals() else 0.5,
                "domain_confidence": confidence if 'confidence' in locals() else 0.5,
            }
            
            # 自動ラベル付け
            try:
                labeled_sample = self.labeler.label_sample(sample, domain)
            except Exception as e:
                error_type = classify_exception(e)
                self.error_handler.handle_error(error_type, url, e, context={"operation": "labeling"})
                # ラベル付け失敗時はデフォルトラベルを使用
                labeled_sample = sample.copy()
                labeled_sample["safety_judgment"] = "ALLOW"
                labeled_sample["confidence"] = 0.5
            
            self.visited_urls.add(url)
            
            return labeled_sample
        
        except PlaywrightTimeoutError as e:
            error_type = ErrorType.TIMEOUT_ERROR
            should_continue = self.error_handler.handle_error(error_type, url, e)
            return None
        except Exception as e:
            error_type = classify_exception(e)
            should_continue = self.error_handler.handle_error(error_type, url, e)
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
                    
                    if not queue:
                        break
                    
                    url = queue.popleft()
                    
                    # ページクロール
                    sample = await self.crawl_page(page, url, language, depth=0)
                    
                    if sample:
                        self.domain_samples[domain].append(sample)
                        self.collected_samples.append(sample)
                        pbar.update(1)
                        logger.debug(f"[OK] Collected sample from {url} (domain: {domain})")
                        
                        # 関連リンクを取得（深度1まで）
                        if sample.get("depth", 0) < 1:
                            links = get_wikipedia_links(page, language)
                            logger.debug(f"[LINKS] Found {len(links)} links from {url}")
                            for link in links[:10]:  # 最大10リンク
                                if link not in self.visited_urls and link not in queue:
                                    queue.append(link)
                    else:
                        logger.debug(f"[SKIP] No sample collected from {url}")
                    
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
            # キーワードをエンコード（日本語はそのまま、英語はスペースをアンダースコアに）
            if language == "ja":
                encoded = keyword
            else:
                encoded = keyword.replace(' ', '_')
            encoded = quote(encoded, safe='')
            url = f"{base_url}{encoded}"
            seed_urls.append(url)
        
        logger.info(f"[SEED] Generated {len(seed_urls)} seed URLs for {domain} ({language})")
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
        
        # バランス調整（ラベル付けは既に完了している）
        logger.info("="*80)
        logger.info("Balancing dataset...")
        logger.info("="*80)
        
        self.collected_samples = self.labeler.balance_complete(self.collected_samples)
        
        # 最終保存
        await self.save_checkpoint()
        self.save_results()
        
        # エラーレポート保存
        if self.error_handler.error_stats:
            error_report = self.output_dir / "error_report.json"
            self.error_handler.save_error_report(error_report)
        
        # リトライレポート保存
        if self.retry_handler and self.retry_handler.retry_stats:
            retry_report = self.output_dir / "retry_report.json"
            self.retry_handler.save_retry_report(retry_report)
        
        logger.info("="*80)
        logger.info(f"[COMPLETE] Collected {len(self.collected_samples):,} samples")
        
        # エラー統計
        if self.error_handler.error_stats:
            error_summary = self.error_handler.get_error_summary()
            logger.info(f"[ERROR_STATS] Total errors: {error_summary['total_errors']}")
            for error_type, count in error_summary['error_types'].items():
                logger.info(f"  - {error_type}: {count}")
        
        # リトライ統計
        if self.retry_handler and self.retry_handler.retry_stats:
            retry_summary = self.retry_handler.get_retry_summary()
            logger.info(f"[RETRY_STATS] Total operations: {retry_summary['total_operations']}")
            logger.info(f"  - Successful retries: {retry_summary['total_successful_retries']}")
            logger.info(f"  - Failed retries: {retry_summary['total_failed_retries']}")
        
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
        
        # 統合ファイルも保存
        all_samples_file = self.output_dir / "wikipedia_all_samples.jsonl"
        with open(all_samples_file, 'w', encoding='utf-8') as f:
            for sample in self.collected_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"[SAVE] Saved all {len(self.collected_samples):,} samples to {all_samples_file.name}")
        
        # 統計レポート
        self._save_statistics()
    
    def _save_statistics(self):
        """統計情報を保存"""
        stats = {
            "total_samples": len(self.collected_samples),
            "by_domain": Counter(s.get("domain", "unknown") for s in self.collected_samples),
            "by_language": Counter(s.get("language", "unknown") for s in self.collected_samples),
            "by_label": Counter(s.get("safety_judgment", "unknown") for s in self.collected_samples),
        }
        
        stats_file = self.output_dir / "wikipedia_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump({
                "total_samples": stats["total_samples"],
                "by_domain": dict(stats["by_domain"]),
                "by_language": dict(stats["by_language"]),
                "by_label": dict(stats["by_label"]),
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[STATS] Statistics saved to {stats_file.name}")


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

